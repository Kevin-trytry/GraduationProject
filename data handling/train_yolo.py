"""
YOLO Model Training Script  (GPU Edition)
==========================================
Trains a YOLOv8 model for hand gesture recognition (classes: 0-9 and B).

Pipeline:
  1. Splits labeled dataset into train / val / test (80 / 10 / 10)
  2. Generates data.yaml automatically
  3. Trains YOLOv8n on GPU (CUDA) — falls back to CPU if no GPU found
  4. Evaluates on test set and prints a detailed report
  5. Exports best.pt to ONNX for cross-platform deployment

Output:
  All results are saved to: <project_root>/model_output/
"""

import os
import shutil
import random
import yaml
from pathlib import Path

import torch
from ultralytics import YOLO


# ============================================================================
# CONFIGURATION
# ============================================================================

_BASE = os.path.dirname(os.path.abspath(__file__))   # .../data handling/

# Source paths (must match auto_label.py output)
IMAGES_ROOT  = os.path.join(_BASE, "New data set")   # Subfolders: 0/, 1/, ..., B/
LABELS_DIR   = os.path.join(_BASE, "labels")          # Flat folder of .txt files

# ⚠️  YOLO truncates Chinese characters in Windows paths.
#    Dataset split stays at a pure-ASCII path; model output goes into the project.
_PROJECT_ROOT = os.path.dirname(_BASE)   # = .../GraduationProject/
SPLIT_ROOT    = r"C:\YOLO_gesture\dataset_split"   # pure-ASCII temp dir for split data
OUTPUT_DIR    = os.path.join(_PROJECT_ROOT, "model_output")  # → project/model_output/

# Model
MODEL_VARIANT      = "yolov8n"
PRETRAINED_WEIGHTS = "yolov8n.pt"

# --------------------------------------------------------------------------
# Auto-detect GPU; fall back to CPU gracefully
# --------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE   = 0          # GPU index 0  (RTX 3060 Laptop)
    # RTX 3060 Laptop has 6 GB VRAM — batch 16 is comfortable for 640 imgsz
    BATCH_SIZE = 16
    WORKERS    = 4        # DataLoader workers (GPU pipeline is the bottleneck)
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    DEVICE     = "cpu"
    BATCH_SIZE = 8
    WORKERS    = 8
    print("⚠️  No GPU found — training on CPU (this will be slow).")

# Hyperparameters
EPOCHS     = 100
IMAGE_SIZE = 640
PATIENCE   = 15          # Early stopping — stop if no improvement for N epochs

# Augmentation (fine-tuned for hand gesture data)
AUGMENT_CFG = dict(
    hsv_h      = 0.015,  # Hue jitter
    hsv_s      = 0.5,    # Saturation jitter
    hsv_v      = 0.4,    # Brightness jitter
    flipud     = 0.0,    # No vertical flip (hands rarely appear upside down)
    fliplr     = 0.5,    # Horizontal flip
    mosaic     = 1.0,    # Mosaic augmentation (helps with small datasets)
    mixup      = 0.1,    # MixUp
)

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO  = 0.10  (remainder)

# Classes — must match CLASS_MAP in auto_label.py (index = class ID)
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B']

# Reproducibility
RANDOM_SEED = 42


# ============================================================================
# STEP 1: BUILD TRAIN / VAL / TEST SPLIT
# ============================================================================

def collect_pairs(images_root: str, labels_dir: str):
    """
    Walk images_root subfolders and pair each image with its label file.
    Returns a list of (image_path, label_path) tuples.
    Only includes pairs where the label file exists AND is non-empty.
    """
    supported = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    pairs = []
    labels_dir = Path(labels_dir)

    for root, _, files in os.walk(images_root):
        for fname in files:
            if Path(fname).suffix.lower() not in supported:
                continue
            img_path   = Path(root) / fname
            label_path = labels_dir / (img_path.stem + '.txt')
            if label_path.exists() and label_path.stat().st_size > 0:
                pairs.append((img_path, label_path))

    return pairs


def split_dataset(pairs, train_ratio=0.80, val_ratio=0.10, seed=42):
    """Shuffle and split pairs into train / val / test lists."""
    random.seed(seed)
    random.shuffle(pairs)
    n       = len(pairs)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train   = pairs[:n_train]
    val     = pairs[n_train:n_train + n_val]
    test    = pairs[n_train + n_val:]
    return train, val, test


def copy_split(pairs, split_name: str, split_root: str):
    """
    Copy image+label pairs into:
      {split_root}/{split_name}/images/
      {split_root}/{split_name}/labels/
    Returns the images directory path (used in data.yaml).
    """
    images_out = Path(split_root) / split_name / "images"
    labels_out = Path(split_root) / split_name / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for img_path, label_path in pairs:
        shutil.copy2(img_path,   images_out / img_path.name)
        shutil.copy2(label_path, labels_out / label_path.name)

    print(f"  {split_name:5s}: {len(pairs):4d} samples → {images_out}")
    return str(images_out.resolve())


def prepare_split(images_root, labels_dir, split_root,
                  train_ratio, val_ratio, seed):
    """Full dataset split pipeline. Returns (train_dir, val_dir, test_dir)."""
    print("\n📁 Collecting image-label pairs...")
    pairs = collect_pairs(images_root, labels_dir)
    print(f"   Found {len(pairs)} valid pairs.")

    if len(pairs) == 0:
        raise RuntimeError(
            "No valid image-label pairs found!\n"
            "Make sure auto_label.py has been run and labels/ is not empty."
        )

    train_pairs, val_pairs, test_pairs = split_dataset(
        pairs, train_ratio, val_ratio, seed
    )

    # Wipe old split and rebuild cleanly
    if Path(split_root).exists():
        shutil.rmtree(split_root)

    print("\n📂 Copying files into split directories...")
    train_dir = copy_split(train_pairs, "train", split_root)
    val_dir   = copy_split(val_pairs,   "val",   split_root)
    test_dir  = copy_split(test_pairs,  "test",  split_root)

    return train_dir, val_dir, test_dir


# ============================================================================
# STEP 2: GENERATE data.yaml
# ============================================================================

def create_data_yaml(yaml_path, train_dir, val_dir, test_dir, classes):
    """Write a YOLO-compatible data.yaml file."""
    Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)

    data = {
        'train': train_dir,
        'val':   val_dir,
        'test':  test_dir,
        'nc':    len(classes),
        'names': classes,
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"\n📝 data.yaml written → {yaml_path}")
    print(f"   Classes ({len(classes)}): {classes}")


# ============================================================================
# STEP 3: TRAIN (GPU)
# ============================================================================

def train_model(data_yaml, epochs, image_size, batch_size,
                device, workers, patience, output_dir):
    """Load pretrained YOLOv8n and fine-tune on our gesture dataset."""
    print(f"\n🚀 Loading {MODEL_VARIANT} pretrained weights...")
    model = YOLO(PRETRAINED_WEIGHTS)

    device_label = (
        f"GPU:{torch.cuda.get_device_name(device)}" if isinstance(device, int)
        else "CPU"
    )

    print(f"\n⚙️  Training config:")
    print(f"   Device   : {device_label}")
    print(f"   Workers  : {workers}")
    print(f"   Epochs   : {epochs}  (early-stop patience={patience})")
    print(f"   Img size : {image_size}x{image_size}")
    print(f"   Batch    : {batch_size}")
    print(f"   Data     : {data_yaml}")
    print("-" * 60)

    results = model.train(
        data        = data_yaml,
        epochs      = epochs,
        imgsz       = image_size,
        batch       = batch_size,
        device      = device,
        workers     = workers,
        patience    = patience,
        project     = output_dir,
        name        = "gesture_model",
        exist_ok    = True,        # overwrite previous run
        save        = True,
        save_period = 10,          # save checkpoint every 10 epochs
        verbose     = True,
        seed        = RANDOM_SEED,
        # Augmentation overrides
        **AUGMENT_CFG,
        # Mixed-precision (AMP) — significant speed-up on CUDA
        amp         = isinstance(device, int),
    )

    return model, results


# ============================================================================
# STEP 4: EVALUATE ON TEST SET
# ============================================================================

def evaluate_model(model, data_yaml, image_size, device, output_dir):
    """Run validation on the held-out test split and print a summary."""
    print("\n📊 Evaluating on TEST set...")

    metrics = model.val(
        data   = data_yaml,
        imgsz  = image_size,
        split  = "test",
        device = device,
        project= output_dir,
        name   = "gesture_model_test",
        exist_ok = True,
    )

    print("\n" + "=" * 60)
    print("📈 TEST SET RESULTS")
    print("=" * 60)
    print(f"   mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"   mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"   Precision    : {metrics.box.mp:.4f}")
    print(f"   Recall       : {metrics.box.mr:.4f}")
    print("=" * 60)

    return metrics


# ============================================================================
# STEP 5: EXPORT TO ONNX
# ============================================================================

def export_onnx(model, image_size, output_dir):
    """Export best.pt to ONNX for cross-platform inference."""
    print("\n📦 Exporting model to ONNX...")
    onnx_path = model.export(
        format  = "onnx",
        imgsz   = image_size,
        opset   = 12,          # Compatible with most runtimes
        simplify= True,
        dynamic = False,
    )
    print(f"   ONNX model saved → {onnx_path}")
    return onnx_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("🎯 Hand Gesture Recognition — YOLOv8 Training Pipeline")
    device_label = (
        f"GPU ({torch.cuda.get_device_name(0)})"
        if torch.cuda.is_available() else "CPU (12-core)"
    )
    print(f"   Classes: 0-9 and B  |  Device: {device_label}")
    print("=" * 60)

    # ── Step 1: Split dataset ─────────────────────────────────────────────
    train_dir, val_dir, test_dir = prepare_split(
        images_root = IMAGES_ROOT,
        labels_dir  = LABELS_DIR,
        split_root  = SPLIT_ROOT,
        train_ratio = TRAIN_RATIO,
        val_ratio   = VAL_RATIO,
        seed        = RANDOM_SEED,
    )

    # ── Step 2: data.yaml ─────────────────────────────────────────────────
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    create_data_yaml(yaml_path, train_dir, val_dir, test_dir, CLASSES)

    # ── Step 3: Train ─────────────────────────────────────────────────────
    model, results = train_model(
        data_yaml  = yaml_path,
        epochs     = EPOCHS,
        image_size = IMAGE_SIZE,
        batch_size = BATCH_SIZE,
        device     = DEVICE,
        workers    = WORKERS,
        patience   = PATIENCE,
        output_dir = OUTPUT_DIR,
    )

    # ── Step 4: Load best weights and evaluate ────────────────────────────
    best_pt = os.path.join(OUTPUT_DIR, "gesture_model", "weights", "best.pt")
    model   = YOLO(best_pt)            # reload the best checkpoint
    metrics = evaluate_model(model, yaml_path, IMAGE_SIZE, DEVICE, OUTPUT_DIR)

    # ── Step 5: Export to ONNX ────────────────────────────────────────────
    onnx_path = export_onnx(model, IMAGE_SIZE, OUTPUT_DIR)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✅ All done!")
    print(f"   best.pt  : {best_pt}")
    print(f"   ONNX     : {onnx_path}")
    print(f"   Results  : {OUTPUT_DIR}\\gesture_model\\results.csv")
    print(f"   mAP@0.5  : {metrics.box.map50:.4f}")
    print("=" * 60)
    print("\n▶  To run inference:")
    print(f"   from ultralytics import YOLO")
    print(f"   model = YOLO('{best_pt}')")
    print("   model.predict(source='your_image.jpg', conf=0.4)")


if __name__ == "__main__":
    main()
