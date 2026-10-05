# GraduationProject — Himax WiseEye2 手勢辨識

11 類手勢（數字 0–9 與手勢 B）即時辨識。模型為 YOLOv8n，以 INT8 全整數量化後經 Vela
編譯，跑在 **Seeed Grove Vision AI Module V2**（Himax WiseEye2 / Cortex-M55 + Ethos-U55 NPU）。

| 分工 | 負責 |
|---|---|
| **Kevin** | 韌體、前後處理、編譯燒錄流程、實機測試與量測工具 |
| **Johnson** | 訓練、匯出（ONNX → TFLite INT8 → Vela） |

**目前狀態（2026-10-04）**：11 個類別在板子上全部可辨識，平均信心度約 68%。
`5` 與 `9` 仍不穩定，待優化。

---

## 1. 環境需求

| 項目 | 版本／說明 |
|---|---|
| OS | Windows 10/11（腳本內的路徑是 Windows 格式） |
| 編譯器 | Arm GNU Toolchain（`arm-none-eabi-gcc`），需在 PATH |
| Make | GNU Make（MSYS2 或 Git Bash 內附皆可） |
| Python | 3.9+，`pip install pyserial opencv-python numpy` |
| 板子 | Seeed Grove Vision AI Module V2，USB 接上後會出現 COM 埠 |

> 腳本預設專案根目錄是 `C:\GraduationProject`，COM 埠是 `COM3`。
> 路徑不同的話改 `build_and_flash.py` 最上面的 `PROJECT_ROOT`，COM 埠用 `--port` 指定。

---

## 2. 目錄結構

```
GraduationProject/
├─ Seeed_Grove_Vision_AI_Module_V2/      Himax/Seeed SDK（含我改過的韌體）
│  └─ EPII_CM55M_APP_S/
│     ├─ app/scenario_app/tflm_yolov8_od/   ★ 韌體主體，我改的東西都在這
│     ├─ prebuilt_libs/                     TFLM 預編譯靜態庫（連結時必要）
│     └─ library/                           Himax 驅動與推論函式庫
├─ model_export_tools/                   匯出與量化腳本（Johnson 維護）
├─ data handling/                        訓練資料處理腳本與標註檔
├─ docs/                                 ★ 所有除錯報告、交接文件、發現紀錄
├─ Analysis/                             測試結果分析圖
├─ build_and_flash.py                    ★ 一鍵編譯 + 燒錄
├─ deploy.py                             印出完整部署指令（不自動執行）
├─ grove_vision_capture.py               ★ PC 端即時預覽與逐幀存檔
├─ check_label_scale.py                  量訓練集標註框的尺度分布
├─ check_scale_invariance.py             量模型對物件大小的敏感度
└─ switch_input_size.py                  一個指令切換全鏈路輸入尺寸
```

---

## 3. 韌體：我改過的檔案

全部位於
`Seeed_Grove_Vision_AI_Module_V2/EPII_CM55M_APP_S/app/scenario_app/tflm_yolov8_od/`

| 檔案 | 改了什麼 |
|---|---|
| **`cvapp_yolov8n_ob.cpp`** | 主要戰場。輸入前處理（stretch / letterbox / center-crop 三種可切換）、YOLOv8 anchor 解碼、DFL、NMS、Layout A/B 自動偵測、大框過濾、座標映射回原始畫面 |
| `yolo_postprocessing.cc` | 後處理輔助 |
| `common_config.h` | Flash 位址、JPEG 輸出開關 |
| `send_result.cpp` | 透過 UART 把偵測結果送回 PC |

### 關鍵參數（都在 `cvapp_yolov8n_ob.cpp` 最上方）

```c
#define YOLOV8_OB_INPUT_TENSOR_WIDTH   192
#define YOLOV8_OB_INPUT_TENSOR_HEIGHT  192
#define YOLOV8_INPUT_MODE              2      // 0=stretch 1=letterbox 2=center-crop
#define YOLOV8_CROP_SIZE               240    // center-crop 的來源邊長
#define YOLOV8_MAX_BOX_AREA_RATIO      0.95f  // 超過畫面這個比例的框視為退化框，丟棄
```

**`YOLOV8_INPUT_MODE` 是這個專案最重要的一個數字。**
相機輸出 320×240（4:3），模型吃 192×192（1:1），兩者不合，怎麼轉決定了成敗：

| 模式 | 做法 | 問題 |
|---|---|---|
| 0 stretch | 直接硬拉成 1:1 | **水平壓扁 25%**，往旁邊張開的手勢（4/5/7/8/9）全部認不出來 |
| 1 letterbox | 等比例縮成 192×144 + 上下補灰邊 114 | 形狀正確，但 25% 畫面是灰色，而且手只有 144px 垂直解析度 |
| 2 center-crop | 取中央 240×240 再縮到 192×192 | **目前採用。** 無灰邊，手的垂直解析度 +33%，代價是水平視野只剩 75% |

> 模式 2 之下手必須放在畫面中央才進得了取樣範圍。
> `grove_vision_capture.py --crop-guide` 會在即時預覽上畫出這個範圍。

改完上面任何一個 define，**必須 `make clean` 再 `make`**。

---

## 4. 編譯與燒錄

```bash
python build_and_flash.py                 # 完整流程
python build_and_flash.py --no-clean      # 增量編譯
python build_and_flash.py --build-only    # 只編譯
python build_and_flash.py --flash-only    # 只燒錄
python build_and_flash.py --port COM5     # 指定 COM 埠
```

燒錄到 xmodem 那一步會停下來，**這時要按住板子上的 BOOT 再按一下 RST**，才會進入燒錄模式。

模型本身是獨立燒進 flash 的，不包在韌體裡：

```
Flash 完整位址   0x3AB7B000
xmodem 偏移位址  0xB7B000
```

換模型只要把新的 `*_vela.tflite` 放進
`Seeed_Grove_Vision_AI_Module_V2/model_zoo/tflm_yolov8_od/` 再燒一次，
**韌體不用重編**（Layout A/B 會自動偵測）。詳細步驟見 `command.txt` 或跑 `python deploy.py`。

---

## 5. PC 端即時預覽

```bash
python grove_vision_capture.py --port COM3 --no-vote --crop-guide
```

會開一個視窗顯示板子回傳的畫面與偵測框，同時把每一幀存成 JPEG、偵測結果寫成
`captures/session_YYYYMMDD_HHMMSS/detections_log.csv`。

常用參數：

| 參數 | 說明 |
|---|---|
| `--no-vote` | 關閉多幀投票。**做測試時一定要加**，否則看到的是平滑後的結果 |
| `--threshold 20` | 信心度門檻（%），預設 20，與韌體一致 |
| `--max-box-area 0.95` | 框佔畫面比例上限，與韌體一致 |
| `--crop-guide 240` | 在預覽上畫出韌體實際取樣的中央區域 |

---

## 6. 測試方式（請照這個做，不然數據沒辦法比較）

這幾條是踩過坑之後訂下來的，寫在 `docs/交接_韌體端結案_20260911_v6.md`：

1. **不要用**「類別出現筆數 / 平均信心度 / 最高信心度」下結論 —— 這三個指標在 09-10 曾經給出互相矛盾的答案。模型退化成「滿版框 + 單一類別」時會產生虛高的信心度。
2. 依序比 `0 → 1 → … → 9 → B`，每個手勢停 8–10 秒，中間手放下 2 秒當分隔。
3. **逐幀人工核對**算真實準確率。
4. 每輪記錄畫面亮度（工具會印），**三輪亮度相差 5 以內才能互相比較**。
5. 要比較任何改動，一律跑 **A → B → A 三輪**。第三輪是用來量「什麼都不改也會有多少誤差」的，沒有它就無從判斷差異是真的還是雜訊。

---

## 7. 這個 repo **沒有**包含什麼

為了讓 clone 不要太痛苦，以下東西不進版控。需要的話跟 Kevin 要雲端硬碟連結：

| 不在 repo 裡 | 原因 |
|---|---|
| `data handling/dataset/*.MOV`（約 360 MB） | 原始拍攝影片，單檔超過 GitHub 100 MB 上限 |
| 資料集影像 `*.jpg`（3,792 張，約 200 MB） | 體積大；標註 `*.txt` **有**保留 |
| 訓練權重 `*.pt` / `*.onnx` | 由訓練端產出，走雲端硬碟交付 |
| `captures/` | 實機測試的逐幀影像，每次測試都會重新產生 |
| `library/inference/torchtag1_0_0_u55tag2502/` | SDK 內的 ExecuTorch / PyTorch 推論庫，9,304 個檔案 / 150 MB 的 LLM tokenizer 程式碼。**本專案走 TFLM 路徑**（`tflmtag2209` 與 `tflmtag2412` 兩個版本都有保留）。需要時可從 Seeed 官方 repo 取回同一路徑 |

---

## 8. SDK 的巢狀版本庫已停用

`Seeed_Grove_Vision_AI_Module_V2/` 原本自己是一個獨立的 git repo（還帶一個 CMSIS-CV
子模組）。巢狀 repo 會讓外層 git 只記錄一個 submodule 指標、**收不到裡面改過的韌體**，
所以把它們的 `.git` 改名停用了：

```
Seeed_Grove_Vision_AI_Module_V2/_git_disabled_20261004
Seeed_Grove_Vision_AI_Module_V2/EPII_CM55M_APP_S/library/cmsis_cv/CMSIS-CV/_git_disabled_20261004
```

這兩個資料夾不進版控。要還原成獨立 repo，改名回 `.git` 即可。

---

## 9. 文件

`docs/` 底下是現行有效的文件：

| 檔案 | 內容 |
|---|---|
| **`根因報告_權重訓練尺寸不符_20260929.md`** | **最重要，先看這份。** 三個月所有症狀的共同根因：匯出腳本一直讀到一顆 imgsz=640 訓練的舊權重，而韌體以 192 運作 |
| `發現紀錄_尺度不變性量測_20260929.md` | 手佔畫面 45%~100% 判對率 99%，30% 以下崩掉。據此撤回了「補拍 1,300 張」的建議 |
| `交接_韌體端結案_20260911_v6.md` | A→B→A 對照實驗與**評估規範**（哪些指標不能用、怎麼比才算數）。這部分現在仍然有效 |
| `電梯輸入介面_給組員_20261005.md` | 手勢 → 兩位數樓層的事件介面 |

韌體本身的現行參數看本文件第 3 節，那是最新的；`cvapp_yolov8n_ob.cpp` 檔頭的註解也維持同步。

### `docs/archive_錯誤權重時期/`

2026-06 到 2026-09 之間的除錯文件。**這段期間所有實機數據都建立在錯誤的權重上**，
所以裡面的結論不要當依據 —— 每一份檔首都加了警語，其中幾份的核心主張已被明確推翻：

| 文件 | 它主張的 | 現在知道的 |
|---|---|---|
| `yolov8_quantization_analysis.md` | 框閃爍源自量化精度 | 量化 MAE 僅 0.00007，不是瓶頸 |
| `Layout.md` | 切開匯出的位置決定精度差 14 倍 | 實機 A 43% vs B 45%，沒有差異 |
| `pipeline_3_firmware.md` | imgsz 96 版規格 | 96 已放棄，前處理也換成 center-crop |
| `實機測試紀錄_20260910.md` | 七場實機分析 | 全部在錯誤權重下取得，需重測 |

保留它們是因為**除錯的過程本身就是這個專題的成果**，不是因為結論還有效。

更早期的幾份（專案架構總覽、韌體說明文件 09-08 版、量化校正偏斜等）同樣過時，
留在 Claude 專案裡沒有落地到這個 repo。
