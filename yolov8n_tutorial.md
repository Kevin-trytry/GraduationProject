# YOLOv8n 完整使用教學

本教學涵蓋從安裝、資料集準備、訓練、驗證、測試到模型匯出的完整流程。範例以 `yolov8n` (nano 版,參數最少、速度最快) 為主,但所有指令對 `yolov8s/m/l/x` 都通用。

---

## 1. 環境安裝

### 1.1 基本需求
- Python ≥ 3.8
- PyTorch ≥ 1.8 (建議搭配 CUDA 以使用 GPU)

### 1.2 安裝 Ultralytics

```bash
pip install ultralytics
```

這一行會自動安裝 `ultralytics` 套件本身以及所有依賴 (torch, opencv, numpy, matplotlib 等)。

### 1.3 驗證安裝

```bash
yolo checks
```

正常會印出 Python、PyTorch、CUDA、GPU 等環境資訊。

### 1.4 (選擇性) 確認 GPU 可用

```python
import torch
print(torch.cuda.is_available())      # True 代表 GPU OK
print(torch.cuda.get_device_name(0))  # 顯示 GPU 型號
```

---

## 2. 資料集準備

### 2.1 資料夾結構

```
my_dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   ├── val/
│   │   ├── img101.jpg
│   │   └── img102.jpg
│   └── test/           ← 選擇性
├── labels/
│   ├── train/
│   │   ├── img001.txt
│   │   └── img002.txt
│   ├── val/
│   │   ├── img101.txt
│   │   └── img102.txt
│   └── test/
└── data.yaml
```

**重點**:`images/` 和 `labels/` 必須**平行存在**,且檔名一一對應 (`img001.jpg` ↔ `img001.txt`)。YOLOv8 內部會自動把路徑中的 `images` 替換成 `labels` 來找標註檔。

### 2.2 標註檔格式 (`.txt`)

每行一個物件:
```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: 整數,從 0 開始
- 後四個值:**相對於圖片寬高的比例 (0~1)**

範例 (`img001.txt`):
```
0 0.521875 0.487500 0.156250 0.225000
2 0.812500 0.345833 0.087500 0.116667
```

### 2.3 像素座標 → YOLO 格式的轉換

如果你的標註是像素座標 (例如 `x_min, y_min, x_max, y_max`),轉換公式如下:

```python
def pixel_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h):
    x_center = (x_min + x_max) / 2 / img_w
    y_center = (y_min + y_max) / 2 / img_h
    width    = (x_max - x_min) / img_w
    height   = (y_max - y_min) / img_h
    return x_center, y_center, width, height
```

### 2.4 `data.yaml` 設定檔

```yaml
# 資料集根目錄 (絕對路徑或相對於目前工作目錄)
path: /home/user/my_dataset

# 三個 split 的路徑 (相對於 path)
train: images/train
val:   images/val
test:  images/test     # 選擇性

# 類別數量與名稱
nc: 3
names:
  0: cat
  1: dog
  2: bird
```

---

## 3. 訓練 (Training)

YOLOv8 提供兩種介面:**CLI** 與 **Python API**。功能完全相同,參數名稱也一樣。

### 3.1 CLI 方式 (最快上手)

```bash
yolo detect train \
    model=yolov8n.pt \
    data=my_dataset/data.yaml \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0 \
    name=exp1
```

第一次執行會自動下載 `yolov8n.pt` 預訓練權重 (COCO 80 類預訓練)。

### 3.2 Python API 方式 (彈性較高)

```python
from ultralytics import YOLO

# 載入預訓練模型 (建議從 .pt 開始,套用 COCO 預訓練權重)
model = YOLO('yolov8n.pt')

# 開始訓練
results = model.train(
    data='my_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,            # GPU 編號,多卡可寫 [0,1]; CPU 寫 'cpu'
    workers=8,           # DataLoader 工作執行緒數
    project='runs/train',
    name='exp1',
    optimizer='auto',    # 'SGD', 'Adam', 'AdamW', 'auto'
    lr0=0.01,            # 初始學習率
    patience=50,         # early stopping 容忍 epochs
    save=True,
    save_period=-1,      # 每隔幾個 epoch 存一次 checkpoint (-1 表只存 best/last)
    pretrained=True,
    verbose=True,
    seed=0,
)
```

### 3.3 從零開始訓練 (不使用預訓練)

```python
# 載入結構檔 (.yaml) 而不是權重檔 (.pt)
model = YOLO('yolov8n.yaml')
model.train(data='my_dataset/data.yaml', epochs=100, imgsz=640)
```

### 3.4 訓練輸出

訓練完成後,結果存放於 `runs/train/exp1/`:

```
runs/train/exp1/
├── weights/
│   ├── best.pt          ← 最佳模型 (驗證 mAP 最高)
│   └── last.pt          ← 最後一個 epoch 的模型
├── results.csv          ← 各 epoch 的 loss/mAP 數值
├── results.png          ← 訓練曲線圖
├── confusion_matrix.png ← 混淆矩陣
├── val_batch0_pred.jpg  ← 驗證集預測視覺化
├── args.yaml            ← 本次訓練使用的所有參數
└── ...
```

### 3.5 從中斷處續訓

```python
model = YOLO('runs/train/exp1/weights/last.pt')
model.train(resume=True)
```

---

## 4. 驗證 (Validation)

訓練過程中每個 epoch 都會自動跑驗證,但你也可以手動對任何 split 跑驗證。

### 4.1 CLI

```bash
yolo detect val model=runs/train/exp1/weights/best.pt data=my_dataset/data.yaml
```

### 4.2 Python API

```python
from ultralytics import YOLO

model = YOLO('runs/train/exp1/weights/best.pt')
metrics = model.val(
    data='my_dataset/data.yaml',
    split='val',         # 'val' 或 'test'
    imgsz=640,
    batch=16,
    conf=0.001,          # 計算 mAP 用的低 confidence threshold
    iou=0.6,             # NMS IoU threshold
)

# 取得指標
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP75:    {metrics.box.map75:.4f}")
print(f"各類別 mAP50-95: {metrics.box.maps}")  # 每個類別的 mAP
```

### 4.3 常用指標解釋

| 指標 | 意義 |
|------|------|
| `mAP50` | IoU 門檻 = 0.5 時的平均精度 (寬鬆) |
| `mAP50-95` | IoU 門檻從 0.5 到 0.95 (步長 0.05) 平均的 mAP (COCO 主要指標) |
| `Precision` | 預測為正中真的是正的比例 |
| `Recall` | 真實為正中被找到的比例 |

---

## 5. 測試 / 推論 (Inference)

> 🎨 **找畫 bounding box 的程式碼?** 直接看 [5.4 推論結果視覺化](#54-推論結果視覺化--畫-bounding-box-的核心區塊-),裡面列出 4 種畫法。本節其他地方用 `★` 符號標示出觸發 bbox 繪製的程式行。

### 5.1 CLI

```bash
# 單張圖片
yolo detect predict model=best.pt source=test.jpg

# 整個資料夾
yolo detect predict model=best.pt source=test_images/

# 影片
yolo detect predict model=best.pt source=test_video.mp4

# 攝影機 (0 號裝置)
yolo detect predict model=best.pt source=0

# 加參數
yolo detect predict model=best.pt source=test.jpg \
    conf=0.5 iou=0.45 imgsz=640 save=True save_txt=True
#                                ↑★ save=True 會自動畫 bbox 並存檔到 runs/detect/predict/ ★
```

### 5.2 Python API (推薦)

```python
from ultralytics import YOLO

model = YOLO('runs/train/exp1/weights/best.pt')

# 對單張圖推論
results = model.predict(
    source='test.jpg',
    conf=0.25,       # 信心門檻
    iou=0.45,        # NMS IoU 門檻
    imgsz=640,
    device=0,
    save=True,       # ★★★ 自動畫 bounding box 並儲存到 runs/detect/predict/ ★★★
    save_txt=True,   # 同時輸出 YOLO 格式 .txt (這個不會畫 bbox,只存座標)
    save_conf=True,  # .txt 中包含 confidence
)

# 解析結果
for r in results:
    boxes = r.boxes
    print(f"偵測到 {len(boxes)} 個物件")
    
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # 像素座標
        
        print(f"  {cls_name}: conf={conf:.3f}, "
              f"bbox=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
```

### 5.3 批次推論多張圖

```python
import glob

image_paths = glob.glob('test_images/*.jpg')

# 一次傳一個 list,內部會 batch 處理
results = model.predict(source=image_paths, conf=0.25)

for img_path, r in zip(image_paths, results):
    print(f"{img_path}: 偵測到 {len(r.boxes)} 個物件")
```

### 5.4 推論結果視覺化 ★ 畫 bounding box 的核心區塊 ★

**在 YOLOv8 中,有 4 種畫 bounding box 的方法,請看下面對照表與範例:**

| 方法 | 程式碼 | 自訂程度 | 適用情境 |
|------|--------|---------|----------|
| ① `save=True` | `model.predict(..., save=True)` | 低 | 快速看結果 |
| ② `r.plot()` | `annotated = r.plot()` | 中 | 需要拿到 numpy 陣列做後處理 |
| ③ `r.show()` | `r.show()` | 低 | 在腳本中直接彈窗顯示 |
| ④ 手動 cv2 | `cv2.rectangle(...)` | 高 | 自訂顏色、字型、樣式 |

---

#### 方法 ① `save=True` (最簡單,YOLO 全自動畫)

```python
# 推論時直接讓 YOLO 把 bbox 畫好並存檔
results = model.predict('test.jpg', save=True)
# ↑ 畫好的圖會出現在 runs/detect/predict/test.jpg
```

#### 方法 ② `r.plot()` ★ 最常用 ★

```python
import cv2
from ultralytics import YOLO

model = YOLO('runs/train/exp1/weights/best.pt')
results = model.predict('test.jpg')
r = results[0]

# ★★★ 這一行就是畫 bounding box 的關鍵 ★★★
annotated = r.plot()   # 回傳已畫上所有 bbox 與標籤的 BGR ndarray

# 你可以選擇儲存或顯示
cv2.imwrite('output.jpg', annotated)
# cv2.imshow('Result', annotated); cv2.waitKey(0)
```

`r.plot()` 也接受參數客製化:
```python
annotated = r.plot(
    conf=True,        # 顯示 confidence 數值
    labels=True,      # 顯示類別名稱
    boxes=True,       # ★ 是否畫 bounding box (預設 True) ★
    line_width=2,     # bbox 邊框粗細
    font_size=None,   # 字型大小
)
```

#### 方法 ③ `r.show()` (直接彈窗)

```python
results = model.predict('test.jpg')
results[0].show()   # ★ 內部也是用 plot() 畫 bbox,再用 PIL 開窗顯示 ★
```

#### 方法 ④ 手動用 OpenCV 畫 (最自由,適合整合到自己的 pipeline)

```python
import cv2
from ultralytics import YOLO

model = YOLO('best.pt')
img = cv2.imread('test.jpg')
results = model.predict(img, conf=0.25)
r = results[0]

# 自訂每個類別的顏色 (BGR)
colors = {
    0: (0, 255, 0),    # cat → 綠色
    1: (255, 0, 0),    # dog → 藍色
    2: (0, 0, 255),    # bird → 紅色
}

# ★★★ 以下迴圈就是手動畫 bounding box 的程式碼 ★★★
for box in r.boxes:
    # 取出座標與類別
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    label = f"{model.names[cls_id]} {conf:.2f}"
    color = colors.get(cls_id, (255, 255, 255))

    # ★ 畫矩形 bounding box ★
    cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)

    # ★ 畫文字標籤背景 + 文字 ★
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
# ★★★ 畫 bbox 的核心區塊到這裡結束 ★★★

cv2.imwrite('output_custom.jpg', img)
```

> 💡 **重點**:不論用哪種方法,bbox 的座標都來自 `box.xyxy` (像素左上+右下) 或 `box.xywh` (中心+寬高)。其他方法只是 YOLO 幫你包好的便利函式;真正的「畫圖」動作就是 `cv2.rectangle()` 或它背後的等價邏輯。

### 5.5 串流模式 (處理影片或大量圖片時節省記憶體)

```python
# stream=True 回傳 generator,逐張處理而非一次全載入
results = model.predict(source='video.mp4', stream=True)

for r in results:
    # 在這裡逐 frame 處理
    boxes = r.boxes
    
    # ★ 若要逐 frame 畫 bbox,在迴圈內呼叫 plot() ★
    annotated_frame = r.plot()
    # cv2.imshow('YOLO', annotated_frame)
    # cv2.waitKey(1)
```

---

## 6. 模型匯出 (Export)

訓練好的 `.pt` 可以匯出成多種部署格式。

### 6.1 CLI

```bash
yolo export model=best.pt format=onnx
```

### 6.2 Python API

```python
from ultralytics import YOLO

model = YOLO('runs/train/exp1/weights/best.pt')

# 匯出 ONNX
model.export(format='onnx', imgsz=640, opset=12, simplify=True)

# 其他常用格式
model.export(format='torchscript')  # PyTorch TorchScript
model.export(format='engine')       # TensorRT (需安裝 TensorRT)
model.export(format='coreml')       # Apple CoreML
model.export(format='tflite')       # TensorFlow Lite
model.export(format='openvino')     # Intel OpenVINO
```

匯出後的檔案會放在原 `.pt` 同一個資料夾。

---

## 7. 常用訓練參數速查

| 參數 | 預設 | 說明 |
|------|------|------|
| `model` | - | 起始權重檔 (`yolov8n.pt`) 或結構檔 (`yolov8n.yaml`) |
| `data` | - | 資料集 YAML 路徑 |
| `epochs` | 100 | 訓練回合數 |
| `imgsz` | 640 | 訓練/推論影像尺寸 (必為 32 的倍數) |
| `batch` | 16 | Batch size (寫 -1 自動依 VRAM 調整) |
| `device` | None | GPU 編號 (`0`, `[0,1]`, `'cpu'`) |
| `workers` | 8 | DataLoader 子程序數 |
| `optimizer` | 'auto' | `SGD`, `Adam`, `AdamW`, `auto` |
| `lr0` | 0.01 | 初始學習率 |
| `lrf` | 0.01 | 最終學習率比例 (= lr0 × lrf) |
| `momentum` | 0.937 | SGD momentum / Adam beta1 |
| `weight_decay` | 0.0005 | 權重衰減 |
| `warmup_epochs` | 3.0 | Warm-up 回合 |
| `patience` | 50 | Early stopping 容忍 epochs (mAP 連 N 輪沒進步就停) |
| `cos_lr` | False | 是否用 cosine learning rate scheduler |
| `augment` | False | 推論時是否啟用 TTA |
| `mosaic` | 1.0 | Mosaic 資料增強機率 |
| `mixup` | 0.0 | MixUp 資料增強機率 |
| `flipud` | 0.0 | 上下翻轉機率 |
| `fliplr` | 0.5 | 左右翻轉機率 |
| `freeze` | None | 凍結前 N 層 (用於遷移學習) |

完整列表見 [Ultralytics 官方訓練文件](https://docs.ultralytics.com/modes/train/)。

---

## 8. 完整流程範例 (一個自定義資料集從頭到尾)

```python
from ultralytics import YOLO

# === 1. 訓練 ===
model = YOLO('yolov8n.pt')
model.train(
    data='my_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    name='my_exp',
)

# === 2. 驗證 (用 best.pt) ===
best_model = YOLO('runs/detect/my_exp/weights/best.pt')
metrics = best_model.val()
print(f"mAP50-95: {metrics.box.map:.4f}")

# === 3. 推論 ===
results = best_model.predict('test.jpg', conf=0.25, save=True)
#                                                   ↑★ save=True 會自動畫 bbox 並存檔 ★
for box in results[0].boxes:
    print(best_model.names[int(box.cls[0])], 
          float(box.conf[0]), 
          box.xyxy[0].tolist())   # ← 這裡是「印出」bbox 座標,不是畫圖

# 若想自己畫 bbox,改用 r.plot() 或手動 cv2.rectangle (見 5.4)
# annotated = results[0].plot()
# cv2.imwrite('out.jpg', annotated)

# === 4. 匯出部署模型 ===
best_model.export(format='onnx', imgsz=640)
```

---

## 9. 常見問題

**Q1. 訓練時 `CUDA out of memory` 怎麼辦?**
降低 `batch` (例如 16 → 8 → 4),或降低 `imgsz` (例如 640 → 416)。設 `batch=-1` 讓 YOLO 自動找最大可用 batch size。

**Q2. 自定義資料集 mAP 一直很低?**
檢查:(a) `data.yaml` 中 `nc` 與 `names` 是否正確;(b) `.txt` 標註座標是否真的是正規化值 (不是像素);(c) 訓練圖數量是否太少 (建議每類至少 100+ 張);(d) `class_id` 是否從 0 開始。

**Q3. 想凍結 backbone 做遷移學習?**
加參數 `freeze=10` (YOLOv8 backbone 約 10 層),會凍結前 10 層只訓練 head。

**Q4. CPU 也能訓練嗎?**
可以,設 `device='cpu'`,但會非常慢。實務上建議至少租用 Colab 或 Kaggle 的免費 GPU。

**Q5. 推論的 `results[0].boxes.xyxy` 是什麼座標系?**
是**原圖的像素座標**,左上 (x1, y1)、右下 (x2, y2)。`xywh` 是中心點+寬高;加 `n` (`xyxyn`, `xywhn`) 則為正規化版本。

---

## 10. 參考資源

- **Ultralytics 官方文件**: https://docs.ultralytics.com/
- **YOLOv8 模型頁**: https://docs.ultralytics.com/models/yolov8/
- **訓練模式**: https://docs.ultralytics.com/modes/train/
- **驗證模式**: https://docs.ultralytics.com/modes/val/
- **推論模式**: https://docs.ultralytics.com/modes/predict/
- **匯出模式**: https://docs.ultralytics.com/modes/export/
- **GitHub Repo**: https://github.com/ultralytics/ultralytics

> 注意:YOLOv8 沒有正式發表的研究論文,引用時請使用 Ultralytics GitHub repo 的 software citation 格式。
