> ## ⚠️ 這份文件的結論已經過時
>
> 2026-09-29 的 `docs/根因報告_權重訓練尺寸不符_20260929.md` 查出：
> 從 2026-06 到 2026-09，匯出腳本一直讀到一顆 **imgsz=640 訓練的舊權重**，
> 而韌體以 192 運作。**這段期間所有的實機數據與推論都建立在錯誤的模型上。**
>
> 保留本文是為了記錄除錯過程，**不要拿它的結論當依據**。
> 現行有效的文件請看 `docs/` 根目錄。

> **特別註記**：本文是 imgsz **96** 版的規格。96 已結案放棄，前處理也已從 stretch 改為 center-crop，整份規格都已過時。
>
---

# 手勢辨識 Pipeline — Part 3／3：韌體端規格

> 撰寫日期：2026-09-08（imgsz 96 修正版）
> 負責範圍：Himax WiseEye2 韌體後處理 + 編譯燒錄流程
> 對應原始碼：`Seeed_Grove_Vision_AI_Module_V2/EPII_CM55M_APP_S/app/scenario_app/tflm_yolov8_od/`

---

## 0. 這份文件要解決什麼

我們三段分工：**① Colab 訓練 → best.pt ② export_full_int8.py → Vela tflite ③ 韌體推論 + 燒錄**。

三段之間有幾個「必須完全一致」的介面，只要有一個對不上，板子上就會出現「跑得動但結果全錯」、「完全不輸出」或直接 hardfault。這份文件把**韌體端的硬性要求**寫清楚，前兩段照著對即可。

**最重要的是第 2 節（韌體對模型的要求）和第 8 節（三方對齊檢查清單）。**

---

## 1. 硬體與環境

| 項目 | 內容 |
|---|---|
| 開發板 | Seeed Grove Vision AI Module V2 |
| SoC | Himax WiseEye2 (WE2) — Cortex-M55 + Ethos-U55 NPU |
| 相機 | OV5647（`tflm_yolov8_od.mk` 的 `CIS_SUPPORT_INAPP_MODEL = cis_ov5647`） |
| Camera raw 解析度 | **320 × 240**（實測 log 中 bbox 上界 319 / 239） |
| 推論框架 | TFLite Micro + Ethos-U55 delegate |
| Tensor arena | 1053 KB |
| UART | 921600 8N1 |
| SDK app | `makefile` → `APP_TYPE = tflm_yolov8_od` |

---

## 2. ★ 韌體對模型的硬性要求

### 2.1 輸入 tensor

| 項目 | 要求值 | 說明 |
|---|---|---|
| shape | `[1, 96, 96, 3]` | NHWC，RGB。**2026-09-08 起由 192 改為 96** |
| dtype | **int8** | 必須 full integer quantization，NPU 才會全速 |
| scale / zero_point | `1/256` / `-128` | 等價於 uint8 0~255 直接搬過來 |

韌體端對應 `cvapp_yolov8n_ob.cpp` 的 `YOLOV8_OB_INPUT_TENSOR_WIDTH / HEIGHT`。這兩個 define 同時決定：

1. 相機影像 resize 的目標尺寸（`hx_lib_image_resize_BGR8U3C_to_RGB24_helium`）
2. DFL 模式下 anchor 數的推算基準
3. bbox 座標換算回 camera raw 的比例

**改 imgsz 一定要同步改這裡**，否則每幀都會被對帳擋掉。

### 2.2 輸出 tensor — 兩種格式都支援

韌體在執行期**自動偵測**，不需事先指定，換模型也不用重編。`N` = anchor 數（見 2.4，目前 189）：

| | **Layout A**（export 有套 split-output patch） | **Layout B**（原始 Detect head 輸出） |
|---|---|---|
| bbox tensor | `[1, 4, N]`，已解碼成 cx,cy,w,h 像素值 | `[1, 64, N]`，DFL raw logits（4 邊 × 16 bins） |
| class tensor | `[1, N, 11]` anchor-major，**已過 sigmoid**（0~1） | `[1, 11, N]` class-major，**raw logits** |
| 解碼工作 | 幾乎不用 | 韌體做 DFL softmax-integral + sigmoid |
| 量化精度 | 較好（兩輸出各自 scale） | 較差（兩輸出共用 scale） |
| 代表檔 | `best_full_int8_vela_new_RGB.tflite` | `best_int8_vela.tflite`（**目前在用**） |

**自動偵測規則**（寫出來是為了讓你們知道什麼情況會誤判）：

```
bbox tensor  = 兩個輸出中 dims[1] 為 4 或 64 的那個
dfl_mode     = (dims[1] == 64)
class-major  = (cls->dims[2] == num_anchors)
num_classes  = class-major ? dims[1] : dims[2]
需要 sigmoid = ((127 - zp) * scale > 1.5)   ← 已 sigmoid 的張量值域必在 0~1
```

> ⚠️ 只有在 `num_classes` 剛好等於 4 或 64 時才會誤判。目前 11 類，安全。

**Layout A 量化精度較好**，若之後訓練端願意在 export 加上雙輸出 patch（見 `yolov8_quantization_analysis.md` §A），**韌體完全不用改**，會自動走 A 分支。

### 2.3 類別定義 — 順序必須完全一致

```
index :  0    1    2    3    4    5    6    7    8    9    10
label : "0"  "1"  "2"  "3"  "4"  "5"  "6"  "7"  "8"  "9"  "B"
```

- 韌體端：`cvapp_yolov8n_ob.cpp` 的 `gesture_classes[]`
- 訓練端：`train_yolo.py` 的 `CLASSES`、`auto_label.py` 的 `CLASS_MAP`

實際手勢語意（依 09-01 分析報告）：

| idx | 手勢 | idx | 手勢 |
|---|---|---|---|
| 0 | fist 全握拳頭 | 6 | Shaka 大拇指＋小拇指 |
| 1 | one 食指 | 7 | L形 大拇指＋食指 |
| 2 | two 食指＋中指 | 8 | 三指 拇＋食＋中 |
| 3 | three 食＋中＋無名 | 9 | 四指 拇＋食＋中＋無名 |
| 4 | four 四指（拇指彎） | 10 | B 手掌側面 |
| 5 | five 全開手掌 | | |

### 2.4 Anchor 數量

```
imgsz 192 → 24² + 12² + 6² = 756
imgsz  96 → 12² +  6² + 3² = 189   ← 目前使用
```

排序與 ultralytics `make_anchors()` 一致：stride 8 → 16 → 32，層內 row-major，每格中心 `(col+0.5, row+0.5)`。

韌體在 DFL 模式下會**自動對帳**：anchor 數與 `YOLOV8_OB_INPUT_TENSOR_WIDTH/HEIGHT` 推算不符時印 ERROR 並跳過每一幀 —— 不會默默算出錯誤座標，但也代表**畫面上會一個框都沒有**。

---

## 3. 韌體後處理流程

```
NPU invoke
  ↓
自動偵測輸出格式（Layout A / B）— 開機印一次
  ↓
逐 anchor 掃 class：只比量化後的 int8，找最高分類別
  ↓
門檻過濾（logit 模式先在 logit 空間比，過了才算 sigmoid — 省 expf）
  ↓
過門檻的 anchor 才解 bbox
    Layout A：直接反量化 cx,cy,w,h
    Layout B：DFL softmax-integral → ltrb（grid 單位）→ ×stride → x1y1x2y2
  ↓
夾回 [0, imgsz]，丟棄 w/h ≤ 1 的退化框
  ↓
★ 丟棄面積 > 畫面 80% 的框（DFL 均勻輸出的產物，見下）
  ↓
class-agnostic NMS (IoU 0.45)
  ↓
座標從 imgsz×imgsz 換算回 camera raw 320×240
  ↓
JPEG(Base64) + JSON 經 UART 送出
```

### 為什麼要丟大框

DFL 每邊是 16 bin 取期望值。**模型沒把握時 softmax 趨近均勻，期望值 → 7.5 格**；在最粗的 stride 那層就等於一個遠超輸入尺寸的距離 → 夾回後就是「整張畫面」的框。

這種框信心度可以高到 88%，而全畫面框 vs 真實手部框的 IoU ≈ 0.52 > NMS 門檻 0.45 —— **它會在 class-agnostic NMS 裡把真正的手部框整個吃掉**。實測 2026-09-08 的 192px 模型：256 筆偵測中有 109 筆（43%）是這種框。

---

## 4. 可調參數一覽

全部在 `cvapp_yolov8n_ob.cpp` 或 `common_config.h`，用 define 名稱搜尋即可：

| 參數 | 檔案 | 目前值 | 說明 |
|---|---|---|---|
| score threshold | `cvapp_yolov8n_ob.cpp`（`yolov8_ob_post_processing()` 呼叫處） | `0.20` | 模型重訓前的折衷值。0.15 太吵，0.35 目前模型達不到 |
| NMS IoU | 同上，同一行第二個參數 | `0.45` | class-agnostic |
| `YOLOV8_OB_INPUT_TENSOR_WIDTH/HEIGHT` | `cvapp_yolov8n_ob.cpp` | `96` | ★ 必須與模型一致 |
| `YOLOV8_MAX_BOX_AREA_RATIO` | `cvapp_yolov8n_ob.cpp` | `0.80f` | 大框過濾；`1.0f` = 關閉 |
| `YOLOV8_DBG_CLASS_SCORES` | `cvapp_yolov8n_ob.cpp` | `1` | 每幀印最佳 anchor 的 11 類分數 |
| `YOLOV8N_OB_DBG_APP_LOG` | `cvapp_yolov8n_ob.cpp` | `1` | 總 debug 開關 |
| `CHANGE_YOLOV8_OB_OUPUT_SHAPE` | `cvapp_yolov8n_ob.cpp` | `1` | 必須為 1（雙輸出路徑） |
| `FRAME_CHECK_DEBUG` | `common_config.h` | `0` | **必須為 0**，否則不輸出 JPEG（PC 端全黑） |
| `YOLOV8_OBJECT_DETECTION_FLASH_ADDR` | `common_config.h` | `0x3AB7B000` | 模型在 flash 的完整位址 |

任何一項改動都要 `make clean && make` 重編。

---

## 5. ★ 模型交付格式（給匯出端）

| 項目 | 要求 |
|---|---|
| 格式 | Vela 編譯後的 `.tflite`（Ethos-U55 custom op） |
| Vela 設定 | `model_export_tools/himax_vela.ini`（`const_mem_area=Axi1`, `arena_mem_area=Axi0`, `core_clock=400e6`） |
| 輸入尺寸 | **96×96×3**，int8 —— 與訓練 `IMAGE_SIZE` 一致 |
| 檔案大小 | 目前 2,786,288 bytes。放在 flash 0xB7B000，往後推不要超過 flash 容量 |
| 交付路徑 | 放到 `model_output/weights/best_saved_model/best_int8_vela.tflite` 即可 |
| 板上檔名 | `model_zoo/tflm_yolov8_od/gesture_yolov8n_96_vela_0xB7B000.tflite`（`deploy.py` 自動複製改名） |

> **Vela 不會把 flash 位址寫進 tflite** —— `himax_vela.ini` 只有 memory mode 設定，檔名裡的 `_0xB7B000` 純屬命名慣例。所以匯出端不需要知道燒錄位址。
>
> **匯出端改了 imgsz 請務必通知韌體端**，這是目前唯一會讓板子「完全沒反應」的介面。

---

## 6. 編譯燒錄流程

```powershell
# 一鍵（推薦）
python deploy.py                        # 同步最新模型到 model_zoo 並改成正確檔名
python build_and_flash.py --port COM3   # make clean + make + image gen + xmodem

# 或分步
cd Seeed_Grove_Vision_AI_Module_V2\EPII_CM55M_APP_S
make clean && make
cd ..\we2_image_gen_local
copy ..\EPII_CM55M_APP_S\obj_epii_evb_icv30_bdv10\gnu_epii_evb_WLCSP65\EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf .\input_case1_secboot\
.\we2_local_image_gen.exe project_case1_blp_wlcsp.json
python ..\xmodem\xmodem_send.py --port=COM3 --baudrate=921600 --protocol=xmodem ^
    --file=...\output.img ^
    --model="...\gesture_yolov8n_96_vela_0xB7B000.tflite 0xB7B000 0x00000"
```

燒錄時看到 `Please press reset button!!` 後：按住 **BOOT** → 點一下 **RST** → 放開 BOOT。

> ⚠️ `build_and_flash.py` 只認 `model_zoo` 底下那個固定檔名，**不會**自動抓最新的 `best_int8_vela.tflite`。拿到新模型要先跑 `deploy.py`，否則會燒到舊快照。

---

## 7. UART 輸出格式（給 PC 端）

一幀一則 JSON，JPEG 影像與偵測結果包在同一則 `INVOKE` 訊息：

```json
{"type": 1, "name": "INVOKE", "code": 0, "data": {
    "count": 0,
    "perf": [...],
    "boxes": [[x, y, w, h, score, cls_id], ...],
    "image": "<base64 JPEG>"
}}
```

| 欄位 | 說明 |
|---|---|
| `x, y` | bbox **左上角**像素座標，基於 **camera raw 320×240** |
| `w, h` | 寬高，同樣基於 camera raw |
| `score` | 0~100 整數（百分比） |
| `cls_id` | 0~10，對應第 2.3 節 |
| `image` | Base64 JPEG |

> 韌體已把座標從模型輸入尺寸換算回 camera raw，**PC 端不需要再做位元移位或縮放**。

### Debug log

開機印一次（**這幾行是排查的第一站**）：

```
=== YOLOv8 OUTPUT LAYOUT (auto-detected) ===
  mode      : B / DFL raw (64ch, 需韌體解碼)
  bbox  dims: [1, 64, 189]  scale(x1000)=169 zp=4
  class dims: [1, 11, 189]  scale(x1000)=169 zp=4
  num_classes=11  num_anchors=189  cls_layout=class-major  sigmoid=YES
```

每幀：

```
[filter] drop oversized box cls=1 conf=62% (96x96)
[CLS-DBG] anchor=112 stride=16 center=(88,56) scores%: 0:3 1:12 2:47 3:8 4:31 5:2 6:5 7:9 8:4 9:6 B:1
detect object[0]: cls=2 conf=61%  box=(30,9,96,102)
```

**`[CLS-DBG]` 是給訓練端最有用的一行** —— 即使該幀完全沒有偵測通過門檻也會印。判讀方式：

| 看到 | 代表 | 該找誰 |
|---|---|---|
| `7:9 8:4 9:6` 全個位數 | 模型對這些手勢根本沒學到 | 訓練端補資料 |
| `4:31 2:47` 兩者都不低 | 有看到，只是輸給 cls2 | 訓練端補對比樣本 / 調 class weight |
| `7:55` 但畫面上沒框 | 被門檻或 NMS 擋掉 | 韌體端調參數 |

---

## 8. ★ 三方對齊檢查清單

| # | 項目 | 訓練端 (Colab) | 匯出端 (export_full_int8.py) | 韌體端 | 目前值 |
|---|---|---|---|---|---|
| 1 | **影像尺寸** | `IMAGE_SIZE` / `imgsz` | `IMAGE_SIZE` | `YOLOV8_OB_INPUT_TENSOR_WIDTH/HEIGHT` | **96**（三處必須一致） |
| 2 | 類別數與順序 | `CLASSES` / `data.yaml` `names` | 沿用 best.pt | `gesture_classes[]` | 11 類，`0`~`9`,`B` |
| 3 | 通道 | RGB | RGB | RGB | 3ch |
| 4 | 量化 | — | `inference_input/output_type = int8` | 要求 int8 | full int8 |
| 5 | 輸入 scale/zp | — | 由校正資料決定 | 不檢查，但應為 1/256 / −128 | ✅ |
| 6 | 是否套 split-output patch | — | 目前**未套** → Layout B | 兩種都支援 | Layout B |
| 7 | Vela 設定 | — | `himax_vela.ini` | — | ✅ |
| 8 | Flash 位址 | — | 不需要知道 | `common_config.h` + `deploy.py` + `build_and_flash.py` 三處 | `0x3AB7B000` / offset `0xB7B000` |

> **第 1 項最容易出事，而且已經出事過一次。**
>
> 2026-09-08 訓練端把 `IMAGE_SIZE` 從 192 改成 96 並重新匯出，韌體端沒同步 → 模型 189 anchors vs 韌體預期 756 → 對帳檢查每幀 `return`，**板子完全不畫框、也沒有明顯錯誤訊息**（只在 UART 印 ERROR）。
>
> 燒錄後第一件事：看 UART 開機那段的 `num_anchors=` 是否符合預期。

---

## 9. 目前狀態

### 韌體端已完成

- 支援 Layout A / B 自動偵測，DFL 解碼邏輯已用真實量化參數做過獨立單元測試
  （anchor 對映邊界、softmax-integral 還原 ltrb、均勻分布期望值 7.5，全數通過）
- NMS 前的大框過濾
- `[CLS-DBG]` 類別分數診斷輸出
- 座標夾回範圍、退化框過濾、anchor 數自動對帳
- imgsz 已同步為 96

### 上一版模型（192px）的實測結論

256 筆偵測 / 83 秒，亮度充足、純色背景：

- **可用**：cls0 / 1 / 2 / 3 / 6 / 10(B)
- **不可用**：cls4、cls5 真實偵測數 = 0（原始那幾筆全是大框幻覺）；cls8 平均僅 22–26%；cls9 有 97% 是大框幻覺
- **cls2 是黑洞**：套用大框過濾後仍佔剩餘偵測的 41%，比 4 常被讀成 2

> 96px 模型尚未取得有效實測數據（第一次燒錄因 imgsz 不一致而完全無輸出）。

### 要靠訓練端解決的

1. cls4 / cls5 補資料（在 192 模型上等同不存在）
2. cls2 過度偏斜 —— 降權或補負樣本
3. cls8 vs cls9 對比樣本（差一根無名指，高混淆對）
4. cls6 (Shaka) vs cls7 (L形) 區分樣本（都是兩指張開）
5. 若可行，在 export 加上 split-output patch 改用 Layout A，量化精度會更好，**韌體不用改**

---

## 附：韌體端主要檔案

| 檔案 | 內容 |
|---|---|
| `cvapp_yolov8n_ob.cpp` | 模型載入、NPU invoke、**後處理（主要改動處）** |
| `common_config.h` | flash 位址、`FRAME_CHECK_DEBUG` |
| `tflm_yolov8_od.c` | 主流程、相機資料路徑、事件迴圈 |
| `send_result.cpp` | JPEG Base64 + JSON 封裝、UART 送出 |
| `yolo_postprocessing.cc` | `sigmoid()`、`box_iou()` 等共用工具 |
| `cis_sensor/cis_ov5647/` | OV5647 相機設定 |
| `tflm_yolov8_od.mk` | 選 sensor、編譯來源清單 |
