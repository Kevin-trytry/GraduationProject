# 🤖 Hand Gesture YOLOv8 Model — 使用說明

> 訓練完成時間：2026-04-29  
> 模型：YOLOv8n（GPU 訓練，RTX 3060 Laptop）  
> 類別：`0` `1` `2` `3` `4` `5` `6` `7` `8` `9` `B`（共 11 類）  
> Test set 成績：**mAP@0.5 = 99.5%，Recall = 100%**

---

## 📁 輸出檔案一覽

| 檔案 | 位置 | 用途 |
|------|------|------|
| `best.pt` | `weights/best.pt` | Python 推論（主力模型） |
| `best.onnx` | `weights/best.onnx` | 跨平台部署（通用格式） |
| `results.csv` | `training_plots/results.csv` | 每個 epoch 的訓練數據 |
| `results.png` | `training_plots/results.png` | 訓練曲線圖（一眼看趨勢） |
| `confusion_matrix_normalized.png` | `test_plots/` | 各類別辨識混淆矩陣 |
| `data.yaml` | `data.yaml` | 資料集類別定義 |

---

## `best.pt` — Python 推論模型

**是什麼：**  
PyTorch 格式的模型權重。記錄了訓練過程中**表現最好的那個 epoch** 的所有神經網路參數。

**怎麼使用：**

```python
from ultralytics import YOLO

# 1. 載入模型
model = YOLO('path/to/best.pt')

# 2. 對一張圖片跑辨識
results = model.predict(source='your_image.jpg', conf=0.4)

# 3. 取出辨識結果
for box in results[0].boxes:
    cls  = int(box.cls[0])    # 類別編號（0~10，10 = B）
    conf = float(box.conf[0]) # 信心分數（0.0 ~ 1.0）
    x1, y1, x2, y2 = box.xyxy[0]  # 框的座標（像素）
    print(f"偵測到手勢：{cls}，信心：{conf:.0%}")
```

**`source` 支援多種輸入：**

| 輸入類型 | 寫法 |
|---------|------|
| 單張圖片 | `source='image.jpg'` |
| 整個資料夾 | `source='./images/'` |
| 影片檔案 | `source='video.mp4'` |
| 電腦鏡頭 | `source=0` |

**`conf` 參數說明：**  
- `conf=0.4` → 只顯示信心 ≥ 40% 的偵測結果  
- 太低（如 0.1）會出現很多錯誤框；太高（如 0.8）可能漏掉正確偵測  
- 建議範圍：`0.4 ~ 0.6`

**適合誰：** 用 Python 做測試或驗證的組員

---

## 🌐 `best.onnx` — 跨平台通用模型

**是什麼：**  
將 `best.pt` 轉換成 ONNX（Open Neural Network Exchange）格式，**不依賴 PyTorch**，幾乎任何語言和平台都能讀取。

**適用平台：**

| 平台 | 說明 |
|------|------|
| Python（onnxruntime） | 不需安裝 PyTorch |
| C++ / Java / C# | 透過 onnxruntime 函式庫 |
| Android / iOS | 透過 ONNX Mobile Runtime |
| OpenCV DNN | 直接讀取 |
| Edge Impulse | 支援 ONNX 匯入 |

> ⚠️ **Himax WiseEye2（WE2）注意事項**  
> WE2 開發板通常使用 **TFLite INT8** 格式，ONNX 無法直接燒錄。  
> 需要額外轉換步驟：  
> `best.pt` → `best.onnx` → `best.tflite（INT8 量化）`  
> 可使用 Ultralytics 的轉換指令：
> ```bash
> yolo export model=best.pt format=tflite int8=True imgsz=640
> ```
> 或透過 **Edge Impulse** 的線上工具上傳 ONNX 再轉換。

---

## 📊 `results.csv` — 訓練數據

每一行 = 一個 epoch（這次共跑了 100 個 epoch）。

**重要欄位說明：**

| 欄位名稱 | 意思 | 理想趨勢 |
|---------|------|---------|
| `train/box_loss` | 訓練集 Bounding Box 位置誤差 | ↓ 持續下降 |
| `train/cls_loss` | 訓練集分類誤差 | ↓ 持續下降 |
| `metrics/mAP50(B)` | Val set mAP@IoU=0.5（最常用指標） | ↑ 趨近 1.0 |
| `metrics/mAP50-95(B)` | 嚴格版 mAP（多個 IoU 門檻平均） | ↑ 越高越好 |
| `metrics/precision(B)` | 精確率（找到的框是否正確） | ↑ 趨近 1.0 |
| `metrics/recall(B)` | 召回率（有沒有漏掉） | ↑ 趨近 1.0 |

**本次訓練最終結果（Epoch 100）：**

| 指標 | 數值 |
|------|------|
| mAP@0.5 | **0.995** |
| mAP@0.5:0.95 | **0.952** |
| Precision | 0.987 |
| Recall | **1.000** |

---

## 📄 `data.yaml` — 資料集定義

訓練時使用的設定檔，定義類別數量與名稱。**推論時不需要**，但重新訓練或執行 `model.val()` 時需要。

```yaml
nc: 11
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B']
```

---

## 快速開始（給組員的 Copy-Paste 指令）

### 安裝環境
```bash
pip install ultralytics
```

### 單張圖片測試
```python
from ultralytics import YOLO

model   = YOLO('model_output/weights/best.pt')
results = model.predict(source='test.jpg', conf=0.4, show=True)
```

### 開鏡頭即時辨識
```python
from ultralytics import YOLO

model = YOLO('model_output/weights/best.pt')
model.predict(source=0, conf=0.4, show=True, stream=True)
```

---

## 外接鏡頭設定

`source=0` 是電腦**內建鏡頭**。若要使用外接 USB 鏡頭，需改成對應編號。

### 鏡頭編號規則

| `source` 值 | 對應鏡頭 |
|-------------|---------|
| `0` | 筆電內建鏡頭（預設） |
| `1` | 第一個外接 USB 鏡頭 |
| `2` | 第二個外接 USB 鏡頭 |

### 步驟 1：確認外接鏡頭編號

執行 `test_source.py`，程式會自動掃描 0~4 號看哪個有效：

```python
import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ source={i} 有效")
        cap.release()
    else:
        print(f"❌ source={i} 無效")
```

> 本機測試結果（2026-04-29）：目前只有 `source=0`（內建鏡頭）有效，外接鏡頭插入後再重新測試。

### 步驟 2：使用外接鏡頭辨識

確認編號後，修改 `fast_test.py` 的 `source` 參數：

```python
from ultralytics import YOLO

model = YOLO('weights/best.pt')
model.predict(source=1, conf=0.4, show=True, stream=True)  # 改成你的鏡頭編號
```

### 常見問題排解

| 問題 | 解法 |
|------|------|
| 鏡頭畫質低、推論慢 | 加 `imgsz=320` 降低輸入解析度 |
| 偵測 FPS 太低 | 加 `vid_stride=2`（每隔一幀辨識一次） |
| 視窗沒有跳出來 | 執行 `pip install opencv-python` |
| 鏡頭被其他程式佔用 | 關閉 Teams、Zoom 等佔用鏡頭的程式 |
| `source=1` 無效 | 確認 USB 鏡頭已插上，或改試 `source=2` |

---

### 查看詳細結果
```python
from ultralytics import YOLO

CLASSES = ['0','1','2','3','4','5','6','7','8','9','B']
model   = YOLO('model_output/weights/best.pt')
results = model.predict(source='test.jpg', conf=0.4)

for box in results[0].boxes:
    cls_id = int(box.cls[0])
    conf   = float(box.conf[0])
    coords = [round(x) for x in box.xyxy[0].tolist()]
    print(f"手勢：{CLASSES[cls_id]}  信心：{conf:.1%}  位置：{coords}")
```

---

## 📝 補充：給報告用的一句話摘要

> 本專案使用 YOLOv8n 模型在 11 類手勢資料集上進行訓練（GPU: RTX 3060 Laptop, 100 epochs），  
> 於 Test Set 達到 **mAP@0.5 = 99.5%、Recall = 100%**，模型以 `.pt`（Python）及 `.onnx`（跨平台）格式輸出。
