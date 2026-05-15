# 🚀 YOLO Hand Detection Pipeline - Quick Reference

## 📂 Files Created

| File | Purpose |
|------|---------|
| `auto_label.py` | Automatic hand detection & YOLO label generation |
| `train_yolo.py` | YOLO model training script |
| `inference.py` | Model inference on images/videos/directories |
| `SETUP_GUIDE.md` | Comprehensive setup & usage guide |

---

## ⚡ Quick Start (3 Steps)

### Step 1: Generate Labels
```bash
python auto_label.py
```
✓ Detects hands in all images  
✓ Creates `/labels/` with `.txt` files  
✓ Generates empty files for background images  

### Step 2: Train Model
```bash
python train_yolo.py
```
✓ Trains YOLOv8n on your labeled dataset  
✓ Saves best weights to `/output/hand_detection_model/weights/best.pt`  

### Step 3: Run Inference
```python
from ultralytics import YOLO

model = YOLO('./output/hand_detection_model/weights/best.pt')
results = model.predict(source='image.jpg', conf=0.25)
```

---

## 🔧 Configuration Quick Reference

### `auto_label.py` - Key Variables
```python
RAW_DATASET_PATH = "./data handling/New data set"  # Input images
OUTPUT_LABELS_PATH = "./data handling/labels"      # Output labels
PADDING_PERCENT = 0.175                            # 17.5% padding (15-20%)
CONFIDENCE_THRESHOLD = 0.5                         # Hand detection confidence
```

### `train_yolo.py` - Key Variables
```python
EPOCHS = 100                    # Training epochs
IMAGE_SIZE = 640               # Input resolution
BATCH_SIZE = 16                # Batch size (↓ if GPU OOM)
MODEL_VARIANT = "yolov8n"      # nano or small
DEVICE = 0                     # GPU ID (0, or [0,1] for multi-GPU)
```

### `inference.py` - Key Variables
```python
MODEL_PATH = "./output/hand_detection_model/weights/best.pt"
INPUT_IMAGE_PATH = 'image.jpg' # or None
INPUT_VIDEO_PATH = 'video.mp4' # or None
INPUT_DIRECTORY = 'dir/path'   # or None
CONFIDENCE_THRESHOLD = 0.25
```

---

## 📦 Installation

```bash
pip install ultralytics mediapipe opencv-python numpy pyyaml torch
```

---

## 🎯 YOLO Label Format

Each `.txt` file contains detections in YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```

Example:
```
0 0.523 0.512 0.287 0.401
```

**Empty files** = Background/negative samples (automatically created)

---

## 📊 Expected Output Structure

```
data handling/
├── auto_label.py
├── train_yolo.py
├── inference.py
├── SETUP_GUIDE.md
├── labels/
│   ├── image_001.txt
│   ├── image_002.txt
│   └── ...

output/
└── hand_detection_model/
    ├── weights/
    │   ├── best.pt          ← Use this for inference
    │   └── last.pt
    ├── results.csv
    ├── results.png          ← Training plots
    └── confusion_matrix.png
```

---

## 💡 Common Tasks

### Check labeled data
```bash
# View first few labels
ls -la ./data handling/labels/ | head -20

# Count labeled images
ls ./data handling/labels/*.txt | wc -l
```

### Adjust training parameters
```python
# In train_yolo.py
BATCH_SIZE = 8           # If GPU out of memory
EPOCHS = 50              # For quick testing
MODEL_VARIANT = "yolov8s"  # Better accuracy
```

### Run inference on directory
```python
# In inference.py
INPUT_DIRECTORY = "./data handling/New data set"
# Then run: python inference.py
```

### Run inference on video
```python
# In inference.py
INPUT_VIDEO_PATH = "./video.mp4"
# Then run: python inference.py
```

---

## ⚠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `hand_landmarker.task` not found | Download from MediaPipe website & place in `data handling/` |
| CUDA out of memory | Reduce `BATCH_SIZE` in `train_yolo.py` |
| No labels generated | Check image format (.jpg, .png), adjust `CONFIDENCE_THRESHOLD` |
| Model not training well | Add more data, increase `EPOCHS`, check label quality |
| Poor inference results | Use `best.pt` (not `last.pt`), adjust `CONFIDENCE_THRESHOLD` |

---

## 🎓 Key Features

✅ **Automatic Labeling**
- Hand detection using MediaPipe
- 15-20% padding (prevents finger cropping)
- Empty labels for background images

✅ **Robust Training**
- Programmatic `data.yaml` generation
- Early stopping with patience
- Multi-GPU support
- Automatic checkpointing

✅ **Easy Inference**
- Single image, video, or batch processing
- Confidence filtering
- Detailed detection statistics

✅ **Production Ready**
- Clean, commented code
- Configurable paths & parameters
- Error handling & validation

---

## 📚 Useful Links

- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)
- [YOLO Format Spec](https://docs.ultralytics.com/datasets/detect/)

---

## 🎯 Next: Fine-tuning Tips

1. **More Data**: Collect more images for better generalization
2. **Data Augmentation**: YOLOv8 handles this automatically
3. **Hyperparameter Tuning**: Try different `BATCH_SIZE`, `lr`, `momentum`
4. **Validation**: Organize train/val/test splits for better evaluation
5. **Model Selection**: Try `yolov8s` or `yolov8m` for better accuracy

---

**Happy training! 🚀**
