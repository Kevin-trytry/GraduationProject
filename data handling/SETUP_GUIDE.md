# 🎯 YOLO Hand Detection Pipeline - Setup & Usage Guide

## 📋 Overview

This guide walks you through the complete setup and execution of an end-to-end YOLO hand detection pipeline:

1. **Auto-Labeling Script** (`auto_label.py`) - Automatically detects hands and generates YOLO format labels
2. **Training Script** (`train_yolo.py`) - Trains YOLOv8 model on your labeled dataset

---

## 🔧 Prerequisites & Installation

### 1. Install Required Packages

```bash
# Create virtual environment (recommended)
python -m venv yolo_env
yolo_env\Scripts\activate  # On Windows
# source yolo_env/bin/activate  # On macOS/Linux

# Install dependencies
pip install ultralytics mediapipe opencv-python numpy pyyaml
pip install torch torchvision  # GPU version: pip install torch torchvision torcuda=cu118
```

### 2. Verify MediaPipe Model

Ensure you have `hand_landmarker.task` in the `data handling/` directory:
- Download from: [Google MediaPipe](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index)
- Or use the path provided in your workout data

### 3. Directory Structure

Your workspace should look like:
```
data handling/
├── hand_landmarker.task          # MediaPipe hand detection model
├── New data set/                 # Raw image dataset
│   ├── 0/, 1/, ..., B/           # Subdirectories with class images
│   └── image files (.jpg, .png)
├── auto_label.py                 # ⭐ Auto-labeling script
├── train_yolo.py                 # ⭐ Training script
└── labels/                       # (Auto-created) YOLO format labels

output/
└── hand_detection_model/         # (Auto-created) Training results
    ├── weights/
    │   ├── best.pt               # Best trained weights
    │   └── last.pt               # Last checkpoint
    ├── results.csv               # Training metrics
    └── *.png                      # Plots & visualizations
```

---

## 🎬 Step-by-Step Execution

### Step 1: Auto-Label Images (Hand Detection)

This script automatically detects hands in your raw images and saves YOLO format labels.

#### Command:
```bash
python auto_label.py
```

#### What it does:
- ✅ Scans all images in `New data set/` directory
- ✅ Detects hands using MediaPipe Hand Landmarker
- ✅ Generates bounding boxes with **17.5% padding** (prevents finger cropping)
- ✅ Saves YOLO format `.txt` files in `labels/` directory
- ✅ Creates **empty `.txt` files for background images** (critical for negative samples)

#### Output Example:
```
✓ image_001.txt - 1 hand(s) detected
◯ background_sample.txt - Background (no hands detected)
✓ image_002.txt - 2 hand(s) detected
```

#### Customization:
Edit these variables in `auto_label.py` if needed:
```python
RAW_DATASET_PATH = "./data handling/New data set"  # Change input path
OUTPUT_LABELS_PATH = "./data handling/labels"      # Change output path
CONFIDENCE_THRESHOLD = 0.5                         # Adjust hand detection confidence
PADDING_PERCENT = 0.175                            # Adjust padding (0.15-0.20)
```

---

### Step 2: Train YOLO Model

This script trains a YOLOv8 model using your labeled dataset.

#### Command:
```bash
python train_yolo.py
```

#### What it does:
- ✅ Creates `data.yaml` configuration file automatically
- ✅ Loads YOLOv8n (nano) model with pretrained weights
- ✅ Trains for ~100 epochs with 640x640 image size
- ✅ Saves checkpoints and best model weights
- ✅ Generates training plots and metrics

#### Output Location:
```
output/hand_detection_model/
├── weights/best.pt          # Use this for inference!
├── results.csv              # Training metrics
├── results.png              # Training curves
├── confusion_matrix.png     # Confusion matrix
└── ...
```

#### Customization:
Edit these parameters in `train_yolo.py`:
```python
EPOCHS = 100                 # Number of training epochs
IMAGE_SIZE = 640             # Input resolution (640, 416, 832, etc.)
BATCH_SIZE = 16              # Batch size (reduce if out of memory)
DEVICE = 0                   # GPU ID (0 for first GPU, [0,1] for multi-GPU)
MODEL_VARIANT = "yolov8n"    # 'yolov8n' (nano) or 'yolov8s' (small)
```

---

## 📊 Understanding YOLO Label Format

YOLO format `.txt` files contain:
```
<class_id> <x_center> <y_center> <width> <height>
```

Example:
```
0 0.5 0.5 0.3 0.4
```

Where:
- `0` = Class ID (hand is class 0)
- `0.5` = X center (normalized 0-1)
- `0.5` = Y center (normalized 0-1)
- `0.3` = Width (normalized 0-1)
- `0.4` = Height (normalized 0-1)

**Empty `.txt` files** = Background/negative samples (crucial for training!)

---

## 🚀 Using Trained Model for Inference

After training, use your model for predictions:

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('./output/hand_detection_model/weights/best.pt')

# Predict on image
results = model.predict(source='image.jpg', conf=0.25)

# Predict on video
results = model.predict(source='video.mp4', conf=0.25)

# Predict on directory
results = model.predict(source='./data handling/New data set', conf=0.25)

# Process results
for result in results:
    for box in result.boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}")
        print(f"Bounding Box: {box.xyxy}")
```

---

## 🐛 Troubleshooting

### Issue: "hand_landmarker.task not found"
**Solution**: Download MediaPipe Hand Landmarker model and place in `data handling/` directory.

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size in `train_yolo.py`:
```python
BATCH_SIZE = 8  # or even smaller
```

### Issue: "No labels generated"
**Solution**: 
1. Check image format is supported (.jpg, .png, .bmp, .tiff)
2. Verify MediaPipe model is correctly loaded
3. Adjust `CONFIDENCE_THRESHOLD` (lower = more detections)

### Issue: "Data.yaml not found"
**Solution**: Ensure `IMAGES_DIR` and `LABELS_DIR` paths are correct in `train_yolo.py`.

---

## 📈 Monitoring Training

During training, monitor these metrics:
- **mAP50**: Mean Average Precision at 0.50 IoU
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **Loss**: Training and validation loss curves

Check `output/hand_detection_model/results.csv` for detailed metrics.

---

## 💡 Best Practices

1. **Dataset Balance**: Mix positive (hands) and negative (background) samples
2. **Image Diversity**: Ensure variety in hand poses, scales, lighting
3. **Label Quality**: Review auto-generated labels, especially edge cases
4. **Validation Split**: Consider organizing data into train/val/test splits before training
5. **Model Selection**:
   - `yolov8n` (nano): Fastest, best for edge devices (~3MB)
   - `yolov8s` (small): Good balance of speed/accuracy (~22MB)
   - `yolov8m` (medium): Better accuracy (~49MB)

---

## 📝 Configuration Summary

### Auto-Labeling (`auto_label.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `PADDING_PERCENT` | 0.175 | 17.5% padding around hands (15-20% range) |
| `CONFIDENCE_THRESHOLD` | 0.5 | Minimum hand detection confidence |
| `CLASS_ID` | 0 | Hand class ID in YOLO labels |

### Training (`train_yolo.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPOCHS` | 100 | Training epochs |
| `IMAGE_SIZE` | 640 | Input image resolution |
| `BATCH_SIZE` | 16 | Batch size per gradient step |
| `MODEL_VARIANT` | yolov8n | Model size (nano, small, medium) |
| `PATIENCE` | 10 | Early stopping patience |

---

## 🎓 Next Steps

1. ✅ Generate labels with `auto_label.py`
2. ✅ Review label quality in `labels/` directory
3. ✅ Adjust hyperparameters if needed
4. ✅ Train model with `train_yolo.py`
5. ✅ Evaluate results in `output/`
6. ✅ Fine-tune and iterate

---

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [YOLO Format Specification](https://docs.ultralytics.com/datasets/detect/)

---

**Good luck with your hand detection model! 🎉**
