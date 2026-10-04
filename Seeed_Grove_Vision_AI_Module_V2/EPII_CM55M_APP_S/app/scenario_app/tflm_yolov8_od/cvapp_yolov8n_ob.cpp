/*
 * cvapp.cpp
 *
 *  Created on: 2018�~12��4��
 *      Author: 902452
 */

#include "cvapp_yolov8n_ob.h"
#include "WE2_device.h"
#include "board.h"
#include "cisdp_sensor.h"
#include <assert.h>
#include <cstdio>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "WE2_core.h"

#include "ethosu_driver.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#if TFLM2209_U55TAG2205
#include "tensorflow/lite/micro/micro_error_reporter.h"
#endif
#include "img_proc_helium.h"
#include "yolo_postprocessing.h"

#include "cisdp_cfg.h"
#include "memory_manage.h"
#include "spi_master_protocol.h"
#include "xprintf.h"
#include <send_result.h>

#define CHANGE_YOLOV8_OB_OUPUT_SHAPE 1

#define INPUT_IMAGE_CHANNELS 3

// ── 模型輸入尺寸 ────────────────────────────────────────────────────────────
// ★ 必須與 .tflite 的 input tensor 完全一致，也就是訓練/匯出端的 IMAGE_SIZE。
//   DFL 模式下韌體會用它推算 anchor 數（stride 8/16/32）並自動對帳：
//       192 -> 24^2 + 12^2 + 6^2 = 756
//        96 -> 12^2 +  6^2 + 3^2 = 189   ← 目前模型（2026-09-08 改為 96 訓練）
//   對不上的話會每幀印 ERROR 並且完全不輸出任何偵測框。
//   這兩個 define 同時決定相機影像 resize 的目標尺寸（見 cv_yolov8n_ob_run）。
#if 1
#define YOLOV8_OB_INPUT_TENSOR_WIDTH 192
#define YOLOV8_OB_INPUT_TENSOR_HEIGHT 192
#define YOLOV8_OB_INPUT_TENSOR_CHANNEL INPUT_IMAGE_CHANNELS
#else
#define YOLOV8_OB_INPUT_TENSOR_WIDTH 224
#define YOLOV8_OB_INPUT_TENSOR_HEIGHT 224
#define YOLOV8_OB_INPUT_TENSOR_CHANNEL INPUT_IMAGE_CHANNELS
#endif

#define YOLOV8N_OB_DBG_APP_LOG 1

// ── Letterbox（等比縮放 + 補邊）─────────────────────────────────────────────
// 相機 raw 是 320x240 (4:3)，模型輸入是 192x192 (1:1)。
// 原本的做法是直接把 4:3 硬拉成 1:1 —— 水平方向被額外壓縮 25%
//   (寬 192/320 = 0.600，高 192/240 = 0.800)
// 但訓練圖片是 1440x1440 的正方形、沒有變形，ultralytics 訓練/推論也都是
// letterbox（等比縮放後補灰邊）。模型從沒看過橫向壓扁的手。
//
// 影響最大的正是「往水平方向伸出去」的特徵 —— 拇指。實測 cls7/8/9（拇指外張）
// 全數失敗、cls4/5（手指橫向張開）也失敗，而 cls0/1/2/3（垂直方向）正常，
// 與這個推論完全吻合。
//
// ── 2026-09-10 更新：letterbox 修好了幾何，但帶來兩個新問題 ────────────────
//
//   1) 灰邊是訓練時從沒出現過的東西。訓練圖是 1440x1440 的完整正方形，
//      上下不會有 114 的灰帶。letterbox 讓 192x192 裡有 48 列（25%）是灰色。
//   2) 有效區只剩 192x144，手的垂直解析度被壓到 144 px。
//
//   中央裁切（center-crop）同時解掉這兩點：
//      從 320x240 取中央 240x240 的正方形 -> 縮到 192x192
//      幾何忠實（正方形對正方形）、沒有灰邊、整個 192x192 都是真影像，
//      手的垂直解析度從 144 -> 192 px（+33%）。
//   代價是左右各裁掉 40 px（水平 FOV 從 100% 變 75%），手要大致置中。
//
// YOLOV8_INPUT_MODE
//   0 = stretch    直接硬拉（原廠行為，幾何失真，僅供 A/B 對照）
//   1 = letterbox  等比縮放 + 上下補灰邊 114
//   2 = center-crop 中央裁切成正方形再縮放  ← 目前預設
// ⚠️ A/B 對照進行中（09-11）：三連測 A→B→A
//    第 1 輪 = 2 (crop)       ✅ 已完成 16:02，平均信心 59.8%、≥70% 44%
//    第 2 輪 = 1 (letterbox)  ✅ 已完成 16:18
//    第 3 輪 = 2 (crop)       ← 目前這輪，回去驗證第 1 輪不是運氣
//    ※ letterbox 那輪跑 PC 工具時要加 --crop-guide 0（沒有裁切，黃框會誤導）
#define YOLOV8_INPUT_MODE 2
#define YOLOV8_LETTERBOX_PAD_VALUE 114

// mode 2 專用：要從 raw 中央取多大的正方形（像素，會自動夾到 raw 的短邊）
//   240 = 垂直用滿、水平取 75%（縮放 0.80x）  ← 預設
//   224 = 垂直 93%、水平 70%（縮放 0.86x）
//   208 = 垂直 87%、水平 65%（縮放 0.92x）
//   192 = 垂直 80%、水平 60%（縮放 1.00x，完全不重採樣，最銳利但 FOV 最窄）
// 數字越小 = 視角越窄、手在模型眼中越大。可用來調「手的表觀大小」而不動幾何。
#define YOLOV8_CROP_SIZE 240

// mode 2 需要把來源指標位移到裁切原點。相機 raw 的記憶體排列會影響位移量：
//   1 = planar（[[BBB..][GGG..][RRR..]]，img_proc_helium.h 的文件如此描述）
//       -> 位移 = sy * img_w + sx（三個平面會一起正確位移）
//   0 = interleaved（[BGR][BGR]...）-> 位移 = (sy * img_w + sx) * ch
// 若切到 mode 2 後畫面明顯錯位／變成雜訊，先把這個值改成 0 再試一次。
// 建議第一次切換時把下面的 YOLOV8_DBG_DUMP_INPUT_FRAME 設成 60，
// 直接把送進 NPU 的張量傾印出來看，一次就能確認排列對不對。
#define YOLOV8_CROP_SRC_PLANAR 1

// 舊名稱相容（其他檔案若有引用不會壞）
#define YOLOV8_USE_LETTERBOX (YOLOV8_INPUT_MODE == 1)

// 一次性把「真正送進模型的輸入張量」以 base64 傾印到 UART，用來確認
// raw_addr -> resize -> int8 這條路徑產生的畫面是否正常（數值統計分不出
// 「正確縮小的圖」與「錯位的垃圾」，必須把圖還原出來看）。
// 每 N 幀傾印一次；設 0 = 關閉。
// 用週期性而非「只在第 N 幀」，是因為板子開機後就開始推論，PC 端要等 2.5 秒
// 才連上，只印一次很容易在記錄開始前就發生掉。
// 27648 bytes -> 約 485 行 base64，921600 baud 下每次約 0.5 秒。
// 60 幀 @ 5.3FPS 約 11 秒一次，抓 30 秒的 log 會有 2~3 次。
// 日常運作設 0；要再檢查「模型實際收到什麼畫面」時改成 60（每 60 幀傾印一次）
// 2026-09-10：切換到 center-crop 後曾暫時設 60 用來確認裁切位置，已確認正確
// （258 個框全部落在 x∈[40,280]，零例外）。
// 2026-09-11：關閉。它每 60 幀會停約 0.5 秒傾印 27648 bytes，實測在 09-11
// 那場造成 19 次 FPS 從 5.3 掉到 1.7~1.9 的停頓。做 A/B 對照時這會吃掉
// 每個手勢的有效幀數，必須關掉。要再檢查輸入畫面時改回 60。
#define YOLOV8_DBG_DUMP_INPUT_FRAME 0

// #define EACH_STEP_TICK
#define TOTAL_STEP_TICK
#define YOLOV8_POST_EACH_STEP_TICK 0
uint32_t systick_1, systick_2;
uint32_t loop_cnt_1, loop_cnt_2;
#define CPU_CLK 0xffffff + 1
static uint32_t capture_image_tick = 0;
#ifdef TRUSTZONE_SEC
#define U55_BASE BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#ifndef TRUSTZONE
#define U55_BASE BASE_ADDR_APB_U55_CTRL_ALIAS
#else
#define U55_BASE BASE_ADDR_APB_U55_CTRL
#endif
#endif

using namespace std;

namespace {

constexpr int tensor_arena_size = 1053 * 1024;

static uint32_t tensor_arena = 0;

struct ethosu_driver ethosu_drv; /* Default Ethos-U device driver */
tflite::MicroInterpreter *yolov8n_ob_int_ptr = nullptr;
TfLiteTensor *yolov8n_ob_input, *yolov8n_ob_output, *yolov8n_ob_output2;
}; // namespace

#if YOLOV8N_OB_DBG_APP_LOG
// Gesture classes: 0-9 digits + B (blank/background)
std::string gesture_classes[] = {"0", "1", "2", "3", "4",
                                  "5", "6", "7", "8", "9", "B"};
#endif

#if (YOLOV8_DBG_DUMP_INPUT_FRAME && YOLOV8N_OB_DBG_APP_LOG)
/**
 * @brief 把輸入張量 base64 印出來（分行送，不需要大緩衝區）
 *
 * 張量此時已是 int8（原本的 uint8 減 128），所以 +128 還原成 resize 寫入的
 * RGB24 位元組順序。
 */
static void dbg_dump_input_base64(const int8_t *p, int n) {
  static const char B64[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  char line[84];
  int li = 0;
  xprintf("[DUMP] BEGIN %d bytes  %dx%dx3 RGB interleaved\r\n", n,
          YOLOV8_OB_INPUT_TENSOR_WIDTH, YOLOV8_OB_INPUT_TENSOR_HEIGHT);
  for (int i = 0; i < n; i += 3) {
    int rem = n - i;
    uint32_t v = ((uint32_t)(uint8_t)(p[i] + 128)) << 16;
    if (rem > 1) v |= ((uint32_t)(uint8_t)(p[i + 1] + 128)) << 8;
    if (rem > 2) v |= ((uint32_t)(uint8_t)(p[i + 2] + 128));
    line[li++] = B64[(v >> 18) & 0x3F];
    line[li++] = B64[(v >> 12) & 0x3F];
    line[li++] = (rem > 1) ? B64[(v >> 6) & 0x3F] : '=';
    line[li++] = (rem > 2) ? B64[v & 0x3F] : '=';
    if (li >= 76) {
      line[li] = 0;
      xprintf("[DUMP] %s\r\n", line);
      li = 0;
    }
  }
  if (li) {
    line[li] = 0;
    xprintf("[DUMP] %s\r\n", line);
  }
  xprintf("[DUMP] END\r\n");
}
#endif

/**
 * @brief 計算 letterbox 參數：等比縮放後的尺寸與上下左右的補邊量
 *
 * 320x240 -> 192x192 會得到 fit=192x144、pad_y=24（上下各 24 列灰邊）
 */
// ── 輸入幾何 ───────────────────────────────────────────────────────────────
// 前處理與後處理必須用同一份幾何，否則框會畫錯位置。這個 struct 就是那份
// 唯一真相：src = 要從 camera raw 取用的矩形；dst = 它會被寫進張量的哪個矩形。
//   stretch   : src=整張 raw          dst=整個張量
//   letterbox : src=整張 raw          dst=張量中間 fit_w x fit_h（其餘是灰邊）
//   crop      : src=raw 中央正方形    dst=整個張量
typedef struct {
  int mode;                  // 實際採用的模式（可能因參數不合理而退回 0）
  int sx, sy, sw, sh;        // camera raw 上的來源矩形
  int dx, dy, dw, dh;        // 模型張量上的目的矩形
} yolov8_geom_t;

static inline void yolov8_input_geom(int img_w, int img_h, int in_w, int in_h,
                                     int mode, yolov8_geom_t *g) {
  g->mode = mode;
  g->sx = 0; g->sy = 0; g->sw = img_w; g->sh = img_h;
  g->dx = 0; g->dy = 0; g->dw = in_w;  g->dh = in_h;

  if (img_w <= 1 || img_h <= 1) { g->mode = 0; return; }

  if (mode == 1) {                       // letterbox
    int fw = in_w;
    int fh = (img_h * in_w) / img_w;
    if (fh > in_h) {                     // 來源比目標更「高瘦」時改以高度為準
      fh = in_h;
      fw = (img_w * in_h) / img_h;
    }
    g->dw = fw; g->dh = fh;
    g->dx = (in_w - fw) / 2;
    g->dy = (in_h - fh) / 2;
  } else if (mode == 2) {                // center-crop
    int c = YOLOV8_CROP_SIZE;
    if (c > img_w) c = img_w;
    if (c > img_h) c = img_h;
    if (c < 8) { g->mode = 0; return; }  // 參數不合理 -> 退回 stretch
    g->sw = c; g->sh = c;
    g->sx = (img_w - c) / 2;
    g->sy = (img_h - c) / 2;
  }
}

// 模型座標 -> camera raw 座標（後處理畫框用）
static inline void yolov8_model_to_raw(const yolov8_geom_t *g,
                                       float mx, float my,
                                       float *rx, float *ry) {
  const float fx = (g->dw > 0) ? (float)g->sw / (float)g->dw : 1.0f;
  const float fy = (g->dh > 0) ? (float)g->sh / (float)g->dh : 1.0f;
  *rx = (float)g->sx + (mx - (float)g->dx) * fx;
  *ry = (float)g->sy + (my - (float)g->dy) * fy;
}

static inline const char *yolov8_geom_mode_name(int mode) {
  return (mode == 2) ? "center-crop" : (mode == 1) ? "letterbox" : "stretch";
}

static void _arm_npu_irq_handler(void) {
  /* Call the default interrupt handler from the NPU driver */
  ethosu_irq_handler(&ethosu_drv);
}

/**
 * @brief  Initialises the NPU IRQ
 **/
static void _arm_npu_irq_init(void) {
  const IRQn_Type ethosu_irqnum = (IRQn_Type)U55_IRQn;

  /* Register the EthosU IRQ handler in our vector table.
   * Note, this handler comes from the EthosU driver */
  EPII_NVIC_SetVector(ethosu_irqnum, (uint32_t)_arm_npu_irq_handler);

  /* Enable the IRQ */
  NVIC_EnableIRQ(ethosu_irqnum);
}

static int _arm_npu_init(bool security_enable, bool privilege_enable) {
  int err = 0;

  /* Initialise the IRQ */
  _arm_npu_irq_init();

  /* Initialise Ethos-U55 device */
#if TFLM2209_U55TAG2205
  const void *ethosu_base_address = (void *)(U55_BASE);
#else
  void *const ethosu_base_address = (void *)(U55_BASE);
#endif

  if (0 !=
      (err = ethosu_init(&ethosu_drv, /* Ethos-U driver device pointer */
                         ethosu_base_address, /* Ethos-U NPU's base address. */
                         NULL, /* Pointer to fast mem area - NULL for U55. */
                         0,    /* Fast mem region size. */
                         security_enable,      /* Security enable. */
                         privilege_enable))) { /* Privilege enable. */
    xprintf("failed to initalise Ethos-U device\n");
    return err;
  }

  xprintf("Ethos-U55 device initialised\n");

  return 0;
}

int cv_yolov8n_ob_init(bool security_enable, bool privilege_enable,
                       uint32_t model_addr) {
  int ercode = 0;

  // set memory allocation to tensor_arena
  tensor_arena = mm_reserve_align(tensor_arena_size, 0x20); // 1mb
  xprintf("TA[%x]\r\n", tensor_arena);

  if (_arm_npu_init(security_enable, privilege_enable) != 0)
    return -1;

  if (model_addr != 0) {
    static const tflite::Model *yolov8n_ob_model =
        tflite::GetModel((const void *)model_addr);

    if (yolov8n_ob_model->version() != TFLITE_SCHEMA_VERSION) {
      xprintf("[ERROR] yolov8n_ob_model's schema version %d is not equal "
              "to supported version %d\n",
              yolov8n_ob_model->version(), TFLITE_SCHEMA_VERSION);
      return -1;
    } else {
      xprintf("yolov8n_ob model's schema version %d\n",
              yolov8n_ob_model->version());
    }
#if TFLM2209_U55TAG2205
    static tflite::MicroErrorReporter yolov8n_ob_micro_error_reporter;
#endif
    static tflite::MicroMutableOpResolver<2> yolov8n_ob_op_resolver;

    yolov8n_ob_op_resolver.AddTranspose();
    if (kTfLiteOk != yolov8n_ob_op_resolver.AddEthosU()) {
      // 原本 return false —— false 等於 0，跟成功的回傳值完全一樣，
      // 呼叫端無從分辨。改成 -1。
      xprintf("[FATAL] Failed to add Arm NPU support to op resolver.\r\n");
      return -1;
    }
#if TFLM2209_U55TAG2205
    static tflite::MicroInterpreter yolov8n_ob_static_interpreter(
        yolov8n_ob_model, yolov8n_ob_op_resolver, (uint8_t *)tensor_arena,
        tensor_arena_size, &yolov8n_ob_micro_error_reporter);
#else
    static tflite::MicroInterpreter yolov8n_ob_static_interpreter(
        yolov8n_ob_model, yolov8n_ob_op_resolver, (uint8_t *)tensor_arena,
        tensor_arena_size);
#endif

    if (yolov8n_ob_static_interpreter.AllocateTensors() != kTfLiteOk) {
      // 原本是無聲的 return false（= 0 = 成功碼），而且 yolov8n_ob_int_ptr
      // 會停在 nullptr，導致 cv_yolov8n_ob_run() 整段推論被跳過、
      // 一行 log 都不印、框數永遠 0 —— 極難查。改成大聲失敗。
      xprintf("[FATAL] AllocateTensors() failed!\r\n");
      xprintf("[FATAL]   tensor_arena_size = %d bytes\r\n",
              (int)tensor_arena_size);
      xprintf("[FATAL]   arena addr        = 0x%x\r\n",
              (unsigned int)tensor_arena);
      xprintf("[FATAL]   model addr        = 0x%x\r\n",
              (unsigned int)model_addr);
      xprintf("[FATAL]   -> arena 不足、或模型含未註冊的 CPU op\r\n");
      return -1;
    }
    yolov8n_ob_int_ptr = &yolov8n_ob_static_interpreter;
    yolov8n_ob_input = yolov8n_ob_static_interpreter.input(0);
    yolov8n_ob_output = yolov8n_ob_static_interpreter.output(0);
#if CHANGE_YOLOV8_OB_OUPUT_SHAPE
    yolov8n_ob_output2 = yolov8n_ob_static_interpreter.output(1);
#endif
  }

  xprintf("initial done\n");
  return ercode;
}

typedef struct detection_cls_yolov8 {
  box bbox;
  float confidence;
  float index;

} detection_cls_yolov8;

static bool yolov8_det_comparator(detection_cls_yolov8 &pa,
                                  detection_cls_yolov8 &pb) {
  return pa.confidence > pb.confidence;
}

static void yolov8_NMSBoxes(std::vector<box> &boxes,
                            std::vector<float> &confidences,
                            float modelScoreThreshold, float modelNMSThreshold,
                            std::vector<int> &nms_result) {
  detection_cls_yolov8 yolov8_bbox;
  std::vector<detection_cls_yolov8> yolov8_bboxes{};
  for (int i = 0; i < boxes.size(); i++) {
    yolov8_bbox.bbox = boxes[i];
    yolov8_bbox.confidence = confidences[i];
    yolov8_bbox.index = i;
    yolov8_bboxes.push_back(yolov8_bbox);
  }
  sort(yolov8_bboxes.begin(), yolov8_bboxes.end(), yolov8_det_comparator);
  int updated_size = yolov8_bboxes.size();
  for (int k = 0; k < updated_size; k++) {
    if (yolov8_bboxes[k].confidence < modelScoreThreshold) {
      continue;
    }

    nms_result.push_back(yolov8_bboxes[k].index);
    for (int j = k + 1; j < updated_size; j++) {
      float iou = box_iou(yolov8_bboxes[k].bbox, yolov8_bboxes[j].bbox);
      // float iou = box_diou(yolov8_bboxes[k].bbox, yolov8_bboxes[j].bbox);
      if (iou > modelNMSThreshold) {
        yolov8_bboxes.erase(yolov8_bboxes.begin() + j);
        updated_size = yolov8_bboxes.size();
        j = j - 1;
      }
    }
  }
}

#if CHANGE_YOLOV8_OB_OUPUT_SHAPE
// ===========================================================================
//  YOLOv8 雙輸出後處理（支援兩種匯出格式，開機時自動偵測，不需改 define）
//
//  [Layout A] 已解碼雙輸出 —— export 端有套 split-output patch
//      bbox : [1, 4,  N]   已是 cx,cy,w,h 像素值（0~input_size）
//      cls  : [1, N,  C]   已過 sigmoid，值域 0~1
//      例：best_full_int8_vela_new_RGB.tflite
//
//  [Layout B] 原始 Detect head 輸出 —— export 端未套 patch
//      bbox : [1, 64, N]   DFL raw logits（4 邊 x reg_max 16 bins，未解碼）
//      cls  : [1, C,  N]   raw logits（未過 sigmoid）
//      例：best_int8_vela.tflite（2026-09-08 訓練）
//
//  N = anchor 總數；192 輸入 → 24^2 + 12^2 + 6^2 = 756
// ===========================================================================

#define YOLOV8_DFL_REG_MAX 16              // DFL 每邊的 bin 數
#define YOLOV8_DFL_BOX_CH  (4 * YOLOV8_DFL_REG_MAX)   // = 64
#define YOLOV8_NUM_LEVELS  3

// DFL 退化框過濾：softmax 接近均勻分布時期望值 -> 7.5 格（reg_max 16 的中點），
// 乘上 stride 後往往超出輸入邊界 -> 夾回後就是一個「整張畫面」的框。這種框
// 信心度可以高到 88%，且與真實手部框的 IoU 約 0.52 > NMS 門檻 0.45，會在
// class-agnostic NMS 裡把真正的框吃掉，所以在進 NMS 之前先丟掉。
//
// 門檻怎麼定（依 2026-09-08 的 192px session，256 筆偵測的面積佔比分布）：
//   P10 56% | P25 66% | P50 77% | P75 90% | P90 97% | P95 99%
//   真實手部框在目前拍攝距離下就佔畫面 67~80%（未被過濾者最大 79.7%）。
//   原本設 0.80f 幾乎沒有餘裕，很可能連正確的框一起丟掉；改用 0.95f，
//   仍能擋掉真正的整張畫面框（>=99% 者），但保留貼近邊界的正常框。
//
// 注意面積「比例」不隨輸入尺寸改變（w、h 同時等比縮放），所以 96 與 192
// 用同一個比例即可；但 stride 8 的退化框在 192 下是 120x120（39%，會通過），
// 在 96 下會被夾成 96x96（100%，被丟）—— 這是兩種尺寸的實質差異。
//
// 調整方式：想更嚴格就往 0.85 調；1.0f = 完全不過濾（僅供除錯驗證）。
#define YOLOV8_MAX_BOX_AREA_RATIO 0.95f

// 每幀印出「分數最高的那個 anchor」的完整 11 類分數，用來診斷某個手勢是
// 「完全沒反應」還是「有反應但輸給別的類別」。不需要時設 0 可省 UART 頻寬。
#define YOLOV8_DBG_CLASS_SCORES 1


static const int yolov8_strides[YOLOV8_NUM_LEVELS] = {8, 16, 32};

/**
 * @brief 由 anchor 序號反推它在哪一層 feature map、中心座標與 stride
 *
 * anchor 排列順序與 ultralytics make_anchors() 一致：
 *   stride 8 那層先排完（row-major），再 stride 16，最後 stride 32
 *   每格中心 = (col + 0.5, row + 0.5)，單位為 grid cell
 */
static inline bool yolov8_anchor_from_index(int idx, int input_w, int input_h,
                                            float *ax, float *ay, int *stride) {
  int base = 0;
  for (int lv = 0; lv < YOLOV8_NUM_LEVELS; lv++) {
    int s = yolov8_strides[lv];
    int gw = input_w / s;
    int gh = input_h / s;
    int cnt = gw * gh;
    if (idx < base + cnt) {
      int local = idx - base;
      *ax = (float)(local % gw) + 0.5f;
      *ay = (float)(local / gw) + 0.5f;
      *stride = s;
      return true;
    }
    base += cnt;
  }
  return false;
}

/** @brief 依 input 尺寸推算 anchor 應有的總數，用來對帳模型是否相符 */
static inline int yolov8_expected_anchors(int input_w, int input_h) {
  int total = 0;
  for (int lv = 0; lv < YOLOV8_NUM_LEVELS; lv++) {
    int s = yolov8_strides[lv];
    total += (input_w / s) * (input_h / s);
  }
  return total;
}

/**
 * @brief DFL 解碼：把單一 anchor 的 64 個 logit 還原成 ltrb 四個距離
 *
 * 每一邊 16 個 bin 先做 softmax（減 max 保持數值穩定），
 * 再取期望值 sum(b * p[b])，得到以 grid cell 為單位的距離。
 */
static inline void yolov8_dfl_decode(const int8_t *box_data, int num_anchors,
                                     int anchor_idx, float scale, int zp,
                                     float *ltrb) {
  for (int side = 0; side < 4; side++) {
    float bins[YOLOV8_DFL_REG_MAX];
    float maxv = -1e30f;

    for (int b = 0; b < YOLOV8_DFL_REG_MAX; b++) {
      int ch = side * YOLOV8_DFL_REG_MAX + b;
      int q = (int)box_data[ch * num_anchors + anchor_idx];
      float v = ((float)q - (float)zp) * scale;
      bins[b] = v;
      if (v > maxv) maxv = v;
    }

    float sum = 0.0f;
    for (int b = 0; b < YOLOV8_DFL_REG_MAX; b++) {
      float e = expf(bins[b] - maxv);
      bins[b] = e;
      sum += e;
    }

    float acc = 0.0f;
    for (int b = 0; b < YOLOV8_DFL_REG_MAX; b++) {
      acc += (float)b * bins[b];
    }
    ltrb[side] = (sum > 0.0f) ? (acc / sum) : 0.0f;
  }
}

static void
yolov8_ob_post_processing(tflite::MicroInterpreter *static_interpreter,
                          float modelScoreThreshold, float modelNMSThreshold,
                          struct_yolov8_ob_algoResult *alg,
                          std::forward_list<el_box_t> &el_algo) {
  uint32_t img_w = app_get_raw_width();
  uint32_t img_h = app_get_raw_height();

  const int input_w = YOLOV8_OB_INPUT_TENSOR_WIDTH;
  const int input_h = YOLOV8_OB_INPUT_TENSOR_HEIGHT;

  // 與前處理共用同一份幾何（見 yolov8_input_geom）。有效影像只佔張量的
  // dst 矩形，其餘是灰邊；座標換算與面積過濾都必須以有效區為準。
  yolov8_geom_t geom;
  yolov8_input_geom((int)img_w, (int)img_h, input_w, input_h,
                    YOLOV8_INPUT_MODE, &geom);
  // 前處理在左右需要補邊時會退回 stretch，這裡要跟著退，否則框會錯位
  if (geom.mode == 1 && (geom.dx != 0 || geom.dw != input_w || geom.dh <= 1)) {
    yolov8_input_geom((int)img_w, (int)img_h, input_w, input_h, 0, &geom);
  }
  const int lb_fit_w = geom.dw, lb_fit_h = geom.dh;
  const int lb_pad_x = geom.dx, lb_pad_y = geom.dy;

  TfLiteTensor *o0 = static_interpreter->output(0);
  TfLiteTensor *o1 = static_interpreter->output(1);

  // ── 自動判斷哪個 tensor 是 bbox、哪個是 class ─────────────────────────────
  // bbox tensor 的 dims[1] 必為 4（已解碼）或 64（DFL raw）；
  // class tensor 的 dims[1] 是類別數(11) 或 anchor 數(756)，不會撞號。
  TfLiteTensor *output = nullptr;   // bbox
  TfLiteTensor *output_2 = nullptr; // class
  if (o0->dims->data[1] == 4 || o0->dims->data[1] == YOLOV8_DFL_BOX_CH) {
    output = o0;
    output_2 = o1;
  } else {
    output = o1;
    output_2 = o0;
  }

  const int box_ch = output->dims->data[1];
  const int num_anchors = output->dims->data[2];
  const bool dfl_mode = (box_ch == YOLOV8_DFL_BOX_CH);

  // class 佈局：[1, C, N] 為 class-major、[1, N, C] 為 anchor-major
  const bool cls_class_major = (output_2->dims->data[2] == num_anchors);
  const int num_classes =
      cls_class_major ? output_2->dims->data[1] : output_2->dims->data[2];

  if (box_ch != 4 && !dfl_mode) {
    xprintf("[ERROR] 未知的 bbox 輸出通道數 %d（預期 4 或 %d），放棄本幀\r\n",
            box_ch, YOLOV8_DFL_BOX_CH);
    return;
  }
  if (num_classes <= 0 || num_classes > 128) {
    xprintf("[ERROR] 類別數異常 %d，放棄本幀\r\n", num_classes);
    return;
  }

  const float output_scale =
      ((TfLiteAffineQuantization *)(output->quantization.params))
          ->scale->data[0];
  const int output_zeropoint =
      ((TfLiteAffineQuantization *)(output->quantization.params))
          ->zero_point->data[0];
  const float output_2_scale =
      ((TfLiteAffineQuantization *)(output_2->quantization.params))
          ->scale->data[0];
  const int output_2_zeropoint =
      ((TfLiteAffineQuantization *)(output_2->quantization.params))
          ->zero_point->data[0];

  // class 是否為未過 sigmoid 的 logit：已過 sigmoid 的張量值域必落在 0~1，
  // 用「量化能表示的最大值」判斷最可靠（logit 模型這裡會是十幾）。
  const float cls_max_repr = (127.0f - (float)output_2_zeropoint) * output_2_scale;
  const bool cls_is_logit = (cls_max_repr > 1.5f);

  // Layout A 的 bbox 有兩種慣例，都實際存在，必須分辨：
  //   原廠 COCO 模型 (yolov8n_od_192_delete_transpose)：
  //       scale 0.006194, zp -108 -> 值域 [-0.12, 1.46]  = 歸一化 0~1
  //       -> 原廠 cvapp 才會乘 input_w / input_h
  //   本專案手勢模型 (best_full_int8_vela_new_RGB)：
  //       scale 1.281372, zp -120 -> 值域 [-10.3, 316.5] = 已經是像素值
  //       -> 不可再乘，否則溢位
  // 用量化能表示的最大值判斷，門檻 2.0 遠離兩者，不會誤判。
  // DFL 模式（Layout B）不適用，那條路徑自己算像素。
  const float box_max_repr = (127.0f - (float)output_zeropoint) * output_scale;
  const bool box_is_normalized = (!dfl_mode) && (box_max_repr <= 2.0f);

  // 省 expf：logit 模式先把門檻換算到 logit 空間，比大小過關了才算 sigmoid
  float score_thr_cmp = modelScoreThreshold;
  if (cls_is_logit) {
    float t = modelScoreThreshold;
    if (t <= 0.0f) t = 1e-6f;
    if (t >= 1.0f) t = 1.0f - 1e-6f;
    score_thr_cmp = logf(t / (1.0f - t));
  }

#if YOLOV8N_OB_DBG_APP_LOG
  {
    static bool layout_logged = false;
    if (!layout_logged) {
      layout_logged = true;
      xprintf("=== YOLOv8 OUTPUT LAYOUT (auto-detected) ===\r\n");
      xprintf("  mode      : %s\r\n",
              dfl_mode ? "B / DFL raw (64ch, 需韌體解碼)"
                       : "A / decoded (4ch, export 已解碼)");
      xprintf("  bbox  dims: [%d, %d, %d]  scale(x1000)=%d zp=%d\r\n",
              output->dims->data[0], output->dims->data[1],
              output->dims->data[2], (int)(output_scale * 1000),
              output_zeropoint);
      xprintf("  class dims: [%d, %d, %d]  scale(x1000)=%d zp=%d\r\n",
              output_2->dims->data[0], output_2->dims->data[1],
              output_2->dims->data[2], (int)(output_2_scale * 1000),
              output_2_zeropoint);
      xprintf("  num_classes=%d  num_anchors=%d  cls_layout=%s  sigmoid=%s\r\n",
              num_classes, num_anchors,
              cls_class_major ? "class-major" : "anchor-major",
              cls_is_logit ? "YES (韌體補算)" : "NO (模型已算)");
      if (!dfl_mode) {
        xprintf("  bbox 座標    : %s (box_max_repr=%d/1000)\r\n",
                box_is_normalized ? "normalized 0~1 (韌體乘 input size)"
                                  : "pixel (直接使用)",
                (int)(box_max_repr * 1000));
      }
      if (dfl_mode) {
        int expect = yolov8_expected_anchors(input_w, input_h);
        if (expect != num_anchors) {
          xprintf("  [ERROR] anchor 數不符！模型=%d, 依 %dx%d 推算=%d\r\n",
                  num_anchors, input_w, input_h, expect);
          xprintf("  [ERROR] 請確認 YOLOV8_OB_INPUT_TENSOR_WIDTH/HEIGHT 與模型一致\r\n");
        }
      }
    }
  }
#endif

  // DFL 模式下 anchor 數必須對得上，否則座標會全錯 —— 直接擋掉比較安全。
  // 這裡一定要出聲：之前這個 return 是靜音的，症狀就是「畫面正常但永遠沒框、
  // 也沒有任何訊息」，非常難查。
  if (dfl_mode && yolov8_expected_anchors(input_w, input_h) != num_anchors) {
    static int anchor_warn_cnt = 0;
    if (anchor_warn_cnt < 5) {
      anchor_warn_cnt++;
      xprintf("[FATAL] anchor 數不符：模型=%d，依 %dx%d 推算=%d -> 跳過整幀\r\n",
              num_anchors, input_w, input_h,
              yolov8_expected_anchors(input_w, input_h));
      xprintf("[FATAL]   -> 韌體 YOLOV8_OB_INPUT_TENSOR_WIDTH/HEIGHT 與模型不一致\r\n");
    }
    return;
  }

  ///////////////////////
  // start postprocessing
  std::vector<uint16_t> class_idxs;
  std::vector<float> confidences;
  std::vector<box> boxes;

  const int8_t *box_data = output->data.int8;
  const int8_t *cls_data = output_2->data.int8;

  // 各關卡計數器：用來一眼看出偵測是在哪一關被濾光的
  int n_pass_thr = 0;   // 通過信心度門檻的 anchor 數
  int n_drop_deg = 0;   // 因退化（w/h <= 1）丟棄
  int n_drop_big = 0;   // 因面積過大丟棄

#if (YOLOV8_DBG_CLASS_SCORES && YOLOV8N_OB_DBG_APP_LOG)
  // 全幀分數最高的 anchor（即使沒過門檻也記錄，這樣「完全沒偵測」時仍看得到）
  int dbg_best_anchor = -1;
  int dbg_best_q = -129;
#endif

  for (int a = 0; a < num_anchors; a++) {
    // ── 1) 先找最高分的類別；只比量化後的整數，不做 expf ──────────────────
    int maxq = -129;
    uint16_t maxClassIndex = 0;
    for (int c = 0; c < num_classes; c++) {
      int q = cls_class_major ? (int)cls_data[c * num_anchors + a]
                              : (int)cls_data[a * num_classes + c];
      if (q > maxq) {
        maxq = q;
        maxClassIndex = (uint16_t)c;
      }
    }

#if (YOLOV8_DBG_CLASS_SCORES && YOLOV8N_OB_DBG_APP_LOG)
    if (maxq > dbg_best_q) {
      dbg_best_q = maxq;
      dbg_best_anchor = a;
    }
#endif

    float maxRaw = ((float)maxq - (float)output_2_zeropoint) * output_2_scale;
    if (maxRaw < score_thr_cmp) {
      continue; // 沒過門檻 → 連 bbox 都不用解，省下 64 次 expf
    }
    float maxScore = cls_is_logit ? sigmoid(maxRaw) : maxRaw;
    if (maxScore < modelScoreThreshold) {
      continue;
    }
    n_pass_thr++;

    // ── 2) 解 bbox ────────────────────────────────────────────────────────
    box bbox;
    if (dfl_mode) {
      float ax, ay;
      int stride;
      if (!yolov8_anchor_from_index(a, input_w, input_h, &ax, &ay, &stride)) {
        continue;
      }
      float ltrb[4];
      yolov8_dfl_decode(box_data, num_anchors, a, output_scale,
                        output_zeropoint, ltrb);
      // ltrb 是「距離 anchor 中心幾格」，乘 stride 換回模型輸入的像素座標
      float x1 = (ax - ltrb[0]) * (float)stride;
      float y1 = (ay - ltrb[1]) * (float)stride;
      float x2 = (ax + ltrb[2]) * (float)stride;
      float y2 = (ay + ltrb[3]) * (float)stride;
      bbox.x = x1;
      bbox.y = y1;
      bbox.w = x2 - x1;
      bbox.h = y2 - y1;
    } else {
      // 已解碼格式：[cx, cy, w, h]
      float d[4];
      for (int k = 0; k < 4; k++) {
        int q = (int)box_data[k * num_anchors + a];
        d[k] = ((float)q - (float)output_zeropoint) * output_scale;
        if (box_is_normalized) {
          // 偶數 index (cx, w) 走寬軸；奇數 index (cy, h) 走高軸
          d[k] *= (k % 2 == 0) ? (float)input_w : (float)input_h;
        }
      }
      bbox.x = d[0] - 0.5f * d[2];
      bbox.y = d[1] - 0.5f * d[3];
      bbox.w = d[2];
      bbox.h = d[3];
    }

    // ── 3) 夾回「有效影像區」，避免負值之後被轉成 uint32 變成天文數字 ──────
    //     letterbox 之下有效區是 [pad, pad+fit]，灰邊不算數
    const float vx0 = (float)lb_pad_x, vy0 = (float)lb_pad_y;
    const float vx1 = (float)(lb_pad_x + lb_fit_w);
    const float vy1 = (float)(lb_pad_y + lb_fit_h);
    float x1 = bbox.x, y1 = bbox.y;
    float x2 = bbox.x + bbox.w, y2 = bbox.y + bbox.h;
    if (x1 < vx0) x1 = vx0;
    if (y1 < vy0) y1 = vy0;
    if (x2 > vx1) x2 = vx1;
    if (y2 > vy1) y2 = vy1;
    bbox.x = x1;
    bbox.y = y1;
    bbox.w = x2 - x1;
    bbox.h = y2 - y1;
    if (bbox.w <= 1.0f || bbox.h <= 1.0f) {
      n_drop_deg++;
      continue; // 退化框
    }

    // ── 4) 丟掉「整張畫面」的框（DFL 均勻輸出的產物，見檔案上方 define 說明）
    if (bbox.w * bbox.h >
        YOLOV8_MAX_BOX_AREA_RATIO * (float)lb_fit_w * (float)lb_fit_h) {
#if YOLOV8N_OB_DBG_APP_LOG
      xprintf("[filter] drop oversized box cls=%d conf=%d%% (%dx%d)\r\n",
              (int)maxClassIndex, (int)(maxScore * 100), (int)bbox.w,
              (int)bbox.h);
#endif
      n_drop_big++;
      continue;
    }

    boxes.push_back(bbox);
    class_idxs.push_back(maxClassIndex);
    confidences.push_back(maxScore);
  }

#if (YOLOV8_DBG_CLASS_SCORES && YOLOV8N_OB_DBG_APP_LOG)
  if (dbg_best_anchor >= 0) {
    float bax = 0.0f, bay = 0.0f;
    int bstride = 0;
    if (!yolov8_anchor_from_index(dbg_best_anchor, input_w, input_h, &bax, &bay,
                                  &bstride)) {
      bstride = 0;
    }
    xprintf("[CLS-DBG] anchor=%d stride=%d center=(%d,%d) scores%%:",
            dbg_best_anchor, bstride, (int)(bax * bstride),
            (int)(bay * bstride));
    for (int c = 0; c < num_classes; c++) {
      int q = cls_class_major
                  ? (int)cls_data[c * num_anchors + dbg_best_anchor]
                  : (int)cls_data[dbg_best_anchor * num_classes + c];
      float raw = ((float)q - (float)output_2_zeropoint) * output_2_scale;
      float s = cls_is_logit ? sigmoid(raw) : raw;
      xprintf(" %s:%d", (c < 11) ? gesture_classes[c].c_str() : "?",
              (int)(s * 100));
    }
    xprintf("\r\n");
  }
#endif

#if YOLOV8N_OB_DBG_APP_LOG
  {
    // 每幀一行總結：偵測在哪一關被濾光，一眼可見。
    // best 是全幀最高的類別分數（即使沒過門檻也算），用來分辨
    //   「模型完全沒反應」(best 很低) vs 「有反應但被後處理擋掉」(best 高但 kept=0)
    int best_pct = 0;
    int best_cls = -1;
#if YOLOV8_DBG_CLASS_SCORES
    if (dbg_best_anchor >= 0) {
      int bq = -129;
      for (int c = 0; c < num_classes; c++) {
        int q = cls_class_major
                    ? (int)cls_data[c * num_anchors + dbg_best_anchor]
                    : (int)cls_data[dbg_best_anchor * num_classes + c];
        if (q > bq) { bq = q; best_cls = c; }
      }
      float braw = ((float)bq - (float)output_2_zeropoint) * output_2_scale;
      float bs = cls_is_logit ? sigmoid(braw) : braw;
      best_pct = (int)(bs * 100);
    }
#endif
    xprintf("[POST] anchors=%d passThr=%d dropDegen=%d dropBig=%d kept=%d "
            "best=%d%%(cls%d)\r\n",
            num_anchors, n_pass_thr, n_drop_deg, n_drop_big, (int)boxes.size(),
            best_pct, best_cls);
  }
#endif

  /**
   * do nms
   **/
  std::vector<int> nms_result;
  yolov8_NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold,
                  nms_result);
#if YOLOV8N_OB_DBG_APP_LOG
  xprintf("nms_result.size(): %d\r\n", nms_result.size());
#endif

  for (int i = 0; i < nms_result.size(); i++) {
    if (!(MAX_TRACKED_YOLOV8_ALGO_RES - i))
      break;
    int idx = nms_result[i];

    // 用共用幾何把模型座標換算回 camera raw 座標。
    // 三種模式都走同一條路：crop 會加回裁切原點，letterbox 會扣掉補邊，
    // stretch 則 src=整張、dst=整個張量，結果與原本完全一致。
    float ox, oy, bx2, by2;
    yolov8_model_to_raw(&geom, boxes[idx].x, boxes[idx].y, &ox, &oy);
    yolov8_model_to_raw(&geom, boxes[idx].x + boxes[idx].w,
                        boxes[idx].y + boxes[idx].h, &bx2, &by2);
    if (ox < 0.0f) ox = 0.0f;
    if (oy < 0.0f) oy = 0.0f;
    if (bx2 > (float)img_w) bx2 = (float)img_w;
    if (by2 > (float)img_h) by2 = (float)img_h;
    float ow = (bx2 > ox) ? (bx2 - ox) : 0.0f;
    float oh = (by2 > oy) ? (by2 - oy) : 0.0f;

    alg->obr[i].confidence = confidences[idx];
    alg->obr[i].bbox.x = (uint32_t)ox;
    alg->obr[i].bbox.y = (uint32_t)oy;
    alg->obr[i].bbox.width = (uint32_t)ow;
    alg->obr[i].bbox.height = (uint32_t)oh;
    alg->obr[i].class_idx = class_idxs[idx];

    el_box_t temp_el_box;
    temp_el_box.score = confidences[idx] * 100;
    temp_el_box.target = class_idxs[idx];
    temp_el_box.x = (uint32_t)ox;
    temp_el_box.y = (uint32_t)oy;
    temp_el_box.w = (uint32_t)ow;
    temp_el_box.h = (uint32_t)oh;
    el_algo.emplace_front(temp_el_box);

#if YOLOV8N_OB_DBG_APP_LOG
    xprintf("detect object[%d]: cls=%d conf=%d%%  box=(%d,%d,%d,%d)\r\n", i,
            (int)class_idxs[idx], (int)(confidences[idx] * 100),
            (int)(boxes[idx].x), (int)(boxes[idx].y), (int)(boxes[idx].w),
            (int)(boxes[idx].h));
#endif
  }
}
#else
static void
yolov8_ob_post_processing(tflite::MicroInterpreter *static_interpreter,
                          float modelScoreThreshold, float modelNMSThreshold,
                          struct_yolov8_ob_algoResult *alg,
                          std::forward_list<el_box_t> &el_algo) {
  uint32_t img_w = app_get_raw_width();
  uint32_t img_h = app_get_raw_height();
  TfLiteTensor *output = static_interpreter->output(0);
  // init postprocessing
  int num_classes = output->dims->data[1] - 4;

  // end init
  ///////////////////////
  // start postprocessing
  int nboxes = 0;
  int input_w = YOLOV8_OB_INPUT_TENSOR_WIDTH;
  int input_h = YOLOV8_OB_INPUT_TENSOR_HEIGHT;

  std::vector<uint16_t> class_idxs;
  std::vector<float> confidences;
  std::vector<box> boxes;

  float output_scale =
      ((TfLiteAffineQuantization *)(output->quantization.params))
          ->scale->data[0];
  int output_zeropoint =
      ((TfLiteAffineQuantization *)(output->quantization.params))
          ->zero_point->data[0];
  int output_size = output->bytes;

#if YOLOV8N_OB_DBG_APP_LOG
  xprintf("=== YOLOv8 POST-PROCESSING DEBUG ===\r\n");
  xprintf("output dims size: %d\r\n", output->dims->size);
  xprintf("output->dims->data[0]: %d\r\n", output->dims->data[0]); // batch=1
  xprintf("output->dims->data[1]: %d\r\n", output->dims->data[1]); // expected: 4+num_cls OR num_anchors
  xprintf("output->dims->data[2]: %d\r\n", output->dims->data[2]); // expected: num_anchors OR 4+num_cls
  printf("output_scale: %f\r\n", output_scale);
  xprintf("output_zeropoint: %d\r\n", output_zeropoint);
  xprintf("output_size (bytes): %d\r\n", output_size);
  xprintf("num_classes (dims[1]-4): %d\r\n", num_classes);
  xprintf("input_w: %d  input_h: %d\r\n", input_w, input_h);
#endif
  /***
   * dequantize the output result
   *
   *
   ******/
  float globalMaxScore = -1.0f;
  int   globalMaxCls   = -1;
  for (int dims_cnt_2 = 0; dims_cnt_2 < output->dims->data[2]; dims_cnt_2++) {
    float outputs_bbox_data[4];
    float maxScore = (-1); // the first four indexes are bbox information
    uint16_t maxClassIndex = 0;
    for (int dims_cnt_1 = 0; dims_cnt_1 < output->dims->data[1]; dims_cnt_1++) {
      int value =
          output->data.int8[dims_cnt_2 + dims_cnt_1 * output->dims->data[2]];

      float deq_value = ((float)value - (float)output_zeropoint) * output_scale;
      if (dims_cnt_1 < 4) {
        /***
         * fix big score
         * ****/
        if (dims_cnt_1 % 2) //==1
        {
          deq_value *= (float)input_h;
        } else {
          deq_value *= (float)input_w;
        }
        outputs_bbox_data[dims_cnt_1] = deq_value;
      } else {
        /***
         * find maximum Score and correspond Class idx
         * **/
        if (maxScore < deq_value) {
          maxScore = deq_value;
          maxClassIndex = dims_cnt_1 - 4;
        }
      }
    }
    // Track global maximum score for diagnostic
    if (maxScore > globalMaxScore) {
      globalMaxScore = maxScore;
      globalMaxCls   = (int)maxClassIndex;
    }
    if (maxScore >= modelScoreThreshold) {
      box bbox;

      bbox.x = (outputs_bbox_data[0] - (0.5 * outputs_bbox_data[2]));
      bbox.y = (outputs_bbox_data[1] - (0.5 * outputs_bbox_data[3]));
      bbox.w = (outputs_bbox_data[2]);
      bbox.h = (outputs_bbox_data[3]);
      boxes.push_back(bbox);
      class_idxs.push_back(maxClassIndex);
      confidences.push_back(maxScore);
    }
  }
#if YOLOV8N_OB_DBG_APP_LOG
  xprintf("boxes.size() before NMS: %d\r\n", (int)boxes.size());
  printf("GlobalMaxScore=%.4f  GlobalMaxCls=%d\r\n", globalMaxScore, globalMaxCls);
  xprintf("Threshold=%.2f\r\n", modelScoreThreshold);
  if (boxes.empty()) {
    xprintf("[WARN] No boxes passed threshold!\r\n");
    xprintf("[HINT] If cls_id was 190-250 in JSON, dims[1]/dims[2] may be swapped.\r\n");
  }
#endif
  /**
   * do nms
   *
   * **/

  std::vector<int> nms_result;
  yolov8_NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold,
                  nms_result);
  for (int i = 0; i < nms_result.size(); i++) {
    if (!(MAX_TRACKED_YOLOV8_ALGO_RES - i))
      break;
    int idx = nms_result[i];

    float scale_factor_w = (float)img_w / (float)YOLOV8_OB_INPUT_TENSOR_WIDTH;
    float scale_factor_h = (float)img_h / (float)YOLOV8_OB_INPUT_TENSOR_HEIGHT;
    alg->obr[i].confidence = confidences[idx];
    alg->obr[i].bbox.x = (uint32_t)(boxes[idx].x * scale_factor_w);
    alg->obr[i].bbox.y = (uint32_t)(boxes[idx].y * scale_factor_h);
    alg->obr[i].bbox.width = (uint32_t)(boxes[idx].w * scale_factor_w);
    alg->obr[i].bbox.height = (uint32_t)(boxes[idx].h * scale_factor_h);
    alg->obr[i].class_idx = class_idxs[idx];

    // Populate el_algo for UART JSON serialization
    el_box_t temp_el_box;
    temp_el_box.score  = (uint32_t)(confidences[idx] * 100);
    temp_el_box.target = class_idxs[idx];
    temp_el_box.x      = alg->obr[i].bbox.x;
    temp_el_box.y      = alg->obr[i].bbox.y;
    temp_el_box.w      = alg->obr[i].bbox.width;
    temp_el_box.h      = alg->obr[i].bbox.height;
    el_algo.emplace_front(temp_el_box);

#if YOLOV8N_OB_DBG_APP_LOG
    printf("detect object[%d]: confidences: %f cls:%d\r\n", i,
           confidences[idx], class_idxs[idx]);
#endif
  }
}

#endif

int cv_yolov8n_ob_run(struct_yolov8_ob_algoResult *algoresult_yolov8n_ob) {
  int ercode = 0;
  float w_scale;
  float h_scale;
  uint32_t img_w = app_get_raw_width();
  uint32_t img_h = app_get_raw_height();
  uint32_t ch = app_get_raw_channels();
  uint32_t raw_addr = app_get_raw_addr();
  uint32_t expand = 0;
  std::forward_list<el_box_t> el_algo;

#if YOLOV8N_OB_DBG_APP_LOG
  xprintf("raw info: w[%d] h[%d] ch[%d] addr[%x]\n", img_w, img_h, ch,
          raw_addr);
#endif

  if (yolov8n_ob_int_ptr != nullptr) {
#ifdef TOTAL_STEP_TICK
    SystemGetTick(&systick_1, &loop_cnt_1);
#endif
#ifdef EACH_STEP_TICK
    SystemGetTick(&systick_1, &loop_cnt_1);
#endif
    // ── get image from sensor and resize（依 YOLOV8_INPUT_MODE 決定幾何）──
    {
      const int IN_W = YOLOV8_OB_INPUT_TENSOR_WIDTH;
      const int IN_H = YOLOV8_OB_INPUT_TENSOR_HEIGHT;
      yolov8_geom_t g;
      yolov8_input_geom((int)img_w, (int)img_h, IN_W, IN_H,
                        YOLOV8_INPUT_MODE, &g);

      // resize 函式每列連續寫 dw*3 bytes，只有在 dw == 張量寬時才能靠
      // 「位移目的指標」補上下邊。dw != IN_W（左右要補邊）無法這樣做，
      // 退回 stretch 以免寫壞記憶體。320x240 -> 192x192 不會走到這裡。
      if (g.mode == 1 && (g.dx != 0 || g.dw != IN_W || g.dh <= 1)) {
        static int lb_warn = 0;
        if (lb_warn < 3) {
          lb_warn++;
          xprintf("[WARN] letterbox 不適用 (dst=%d,%d %dx%d)，改用 stretch\r\n",
                  g.dx, g.dy, g.dw, g.dh);
        }
        yolov8_input_geom((int)img_w, (int)img_h, IN_W, IN_H, 0, &g);
      }

      // 只印一次，確認實際採用的幾何（切換 mode 後對照用）
      static int geom_logged = 0;
      if (!geom_logged) {
        geom_logged = 1;
        xprintf("[GEOM] mode=%s  raw=%dx%d -> src=(%d,%d %dx%d) "
                "dst=(%d,%d %dx%d)  有效像素=%d%%\r\n",
                yolov8_geom_mode_name(g.mode), (int)img_w, (int)img_h,
                g.sx, g.sy, g.sw, g.sh, g.dx, g.dy, g.dw, g.dh,
                (IN_W * IN_H) ? (g.dw * g.dh * 100) / (IN_W * IN_H) : 0);
      }

      uint8_t *src = (uint8_t *)raw_addr;
      uint8_t *dst = (uint8_t *)yolov8n_ob_input->data.data;

      if (g.dw != IN_W || g.dh != IN_H) {
        // 有補邊：先整片填成 letterbox 灰（與 ultralytics 的 114 相同）
        memset(dst, YOLOV8_LETTERBOX_PAD_VALUE, yolov8n_ob_input->bytes);
        dst += (size_t)g.dy * IN_W * 3;   // 目的地往下位移 dy 列（RGB24 交錯）
      }

      if (g.sx != 0 || g.sy != 0) {
        // 中央裁切：把來源指標移到裁切原點。
        // planar 時三個平面的起點都是 base + p*W*H，一起加同一個位移即正確；
        // interleaved 時要再乘上 channel 數。見 YOLOV8_CROP_SRC_PLANAR 說明。
#if YOLOV8_CROP_SRC_PLANAR
        src += (size_t)g.sy * img_w + (size_t)g.sx;
#else
        src += ((size_t)g.sy * img_w + (size_t)g.sx) * ch;
#endif
      }

      // align-corners：取樣範圍剛好是 [0, sw-1] x [0, sh-1]
      w_scale = (g.dw > 1) ? (float)(g.sw - 1) / (float)(g.dw - 1) : 0.0f;
      h_scale = (g.dh > 1) ? (float)(g.sh - 1) / (float)(g.dh - 1) : 0.0f;

      // input_w / input_h 一律傳「完整 raw 尺寸」，因為函式用它算列間距與
      // 平面間距；取用範圍是靠 src 指標 + w_scale/h_scale 決定的。
      hx_lib_image_resize_BGR8U3C_to_RGB24_helium(
          src, dst, img_w, img_h, ch, g.dw, g.dh, w_scale, h_scale);
    }
#ifdef EACH_STEP_TICK
    SystemGetTick(&systick_2, &loop_cnt_2);
    dbg_printf(
        DBG_LESS_INFO,
        "Tick for resize image BGR8U3C_to_RGB24_helium for yolov8 OB:[%d]\r\n",
        (loop_cnt_2 - loop_cnt_1) * CPU_CLK + (systick_1 - systick_2));
#endif

#ifdef EACH_STEP_TICK
    SystemGetTick(&systick_1, &loop_cnt_1);
#endif

    // //uint8 to int8
    for (int i = 0; i < yolov8n_ob_input->bytes; ++i) {
      *((int8_t *)yolov8n_ob_input->data.data + i) =
          *((int8_t *)yolov8n_ob_input->data.data + i) - 128;
    }

#if YOLOV8N_OB_DBG_APP_LOG
    {
      // 檢查真正送進模型的影像是否正常。這一關能分辨「相機/縮放壞掉」與
      // 「模型看不懂畫面」：
      //   min=max        -> 整張純色（相機沒資料、raw_addr 無效、resize 沒寫入）
      //   min=-128 max=-128 -> 全黑
      //   min≈-128 max≈127，mean 在中間 -> 正常影像
      const int8_t *pin = (const int8_t *)yolov8n_ob_input->data.data;
      const int n = (int)yolov8n_ob_input->bytes;
      int vmin = 127, vmax = -128;
      long sum = 0;
      for (int i = 0; i < n; ++i) {
        int v = (int)pin[i];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
        sum += v;
      }
      xprintf("[INPUT] %dx%dx3 bytes=%d min=%d max=%d mean=%d\r\n",
              YOLOV8_OB_INPUT_TENSOR_WIDTH, YOLOV8_OB_INPUT_TENSOR_HEIGHT, n,
              vmin, vmax, (int)(n ? (sum / n) : 0));

// 守衛條件必須與 dbg_dump_input_base64() 的定義完全一致，否則
// YOLOV8N_OB_DBG_APP_LOG=0 時會「函式被編掉、呼叫還在」導致編譯失敗
#if (YOLOV8_DBG_DUMP_INPUT_FRAME && YOLOV8N_OB_DBG_APP_LOG)
      {
        static int dump_frame_no = 0;
        dump_frame_no++;
        if ((dump_frame_no % YOLOV8_DBG_DUMP_INPUT_FRAME) == 0) {
          dbg_dump_input_base64(pin, n);
        }
      }
#endif
    }
#endif

#ifdef EACH_STEP_TICK
    SystemGetTick(&systick_2, &loop_cnt_2);
    dbg_printf(DBG_LESS_INFO,
               "Tick for Invoke for uint8toint8 for YOLOV8_OB:[%d]\r\n\n",
               (loop_cnt_2 - loop_cnt_1) * CPU_CLK + (systick_1 - systick_2));
#endif

#ifdef EACH_STEP_TICK
    SystemGetTick(&systick_1, &loop_cnt_1);
#endif
    TfLiteStatus invoke_status = yolov8n_ob_int_ptr->Invoke();

#ifdef EACH_STEP_TICK
    SystemGetTick(&systick_2, &loop_cnt_2);
#endif
    if (invoke_status != kTfLiteOk) {
      xprintf("yolov8 object detect invoke fail\n");
      return -1;
    } else {
#if YOLOV8N_OB_DBG_APP_LOG
      xprintf("yolov8 object detect  invoke pass\n");
#endif
    }
#ifdef EACH_STEP_TICK
    dbg_printf(DBG_LESS_INFO, "Tick for Invoke for YOLOV8_OB:[%d]\r\n\n",
               (loop_cnt_2 - loop_cnt_1) * CPU_CLK + (systick_1 - systick_2));
#endif

#ifdef EACH_STEP_TICK
    SystemGetTick(&systick_1, &loop_cnt_1);
#endif
    // retrieve output data
    yolov8_ob_post_processing(yolov8n_ob_int_ptr, 0.20, 0.45,  // score threshold: 暫定 0.20（模型重訓前的折衷值，原 0.15 太低，0.35 目前模型達不到）
                              algoresult_yolov8n_ob, el_algo);
#ifdef EACH_STEP_TICK
    SystemGetTick(&systick_2, &loop_cnt_2);
    dbg_printf(DBG_LESS_INFO,
               "Tick for Invoke for YOLOV8_OB_post_processing:[%d]\r\n\n",
               (loop_cnt_2 - loop_cnt_1) * CPU_CLK + (systick_1 - systick_2));
#endif
#if YOLOV8N_OB_DBG_APP_LOG
    xprintf("yolov8_ob_post_processing done\r\n");
#endif
#ifdef TOTAL_STEP_TICK
    SystemGetTick(&systick_2, &loop_cnt_2);
    // dbg_printf(DBG_LESS_INFO,"Tick for TOTAL YOLOV8
    // OB:[%d]\r\n",(loop_cnt_2-loop_cnt_1)*CPU_CLK+(systick_1-systick_2));
#endif
  } else {
    // interpreter 是 nullptr 代表 cv_yolov8n_ob_init() 沒跑完（多半是
    // AllocateTensors 失敗）。原本這裡什麼都不做，症狀就是「畫面正常但
    // 永遠沒有框、也沒有任何 log」。印出來才查得到。
    static int null_warn_cnt = 0;
    if (null_warn_cnt < 5) {
      null_warn_cnt++;
      xprintf("[FATAL] interpreter is NULL -- init failed, skipping inference\r\n");
    }
  }

#ifdef UART_SEND_ALOGO_RESEULT
  algoresult_yolov8n_ob->algo_tick = (loop_cnt_2 - loop_cnt_1) * CPU_CLK +
                                     (systick_1 - systick_2) +
                                     capture_image_tick;
  uint32_t judge_case_data;
  uint32_t g_trans_type;
  hx_drv_swreg_aon_get_appused1(&judge_case_data);
  g_trans_type = (judge_case_data >> 16);
  if (g_trans_type == 0 ||
      g_trans_type == 2) // transfer type is (UART) or (UART & SPI)
  {
    // invalid dcache to let uart can send the right jpeg img out
    hx_InvalidateDCache_by_Addr((volatile void *)app_get_jpeg_addr(),
                                sizeof(uint8_t) * app_get_jpeg_sz());

    el_img_t temp_el_jpg_img = el_img_t{};
    temp_el_jpg_img.data = (uint8_t *)app_get_jpeg_addr();
    temp_el_jpg_img.size = app_get_jpeg_sz();
    temp_el_jpg_img.width = app_get_raw_width();
    temp_el_jpg_img.height = app_get_raw_height();
    temp_el_jpg_img.format = EL_PIXEL_FORMAT_JPEG;
    temp_el_jpg_img.rotate = EL_PIXEL_ROTATE_0;

    send_device_id();
    // event_reply(concat_strings(", ", box_results_2_json_str(el_algo), ", ",
    // img_2_json_str(&temp_el_jpg_img)));
    event_reply(concat_strings(
        ", ", algo_tick_2_json_str(algoresult_yolov8n_ob->algo_tick), ", ",
        box_results_2_json_str(el_algo), ", ",
        img_2_json_str(&temp_el_jpg_img)));
  }
  set_model_change_by_uart();
#endif

  SystemGetTick(&systick_1, &loop_cnt_1);
  // recapture image
  sensordplib_retrigger_capture();

  SystemGetTick(&systick_2, &loop_cnt_2);
  capture_image_tick =
      (loop_cnt_2 - loop_cnt_1) * CPU_CLK + (systick_1 - systick_2);
  return ercode;
}

int cv_yolov8n_ob_deinit() { return 0; }
