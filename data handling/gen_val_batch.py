"""
gen_val_batch.py
================
Generate val_batch label + prediction images for specific classes (8, 9, B),
matching the style of YOLO's built-in val_batch*.jpg outputs.

Usage:
    python gen_val_batch.py
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
_BASE        = os.path.dirname(os.path.abspath(__file__))
_PROJECT     = os.path.dirname(_BASE)

BEST_PT      = os.path.join(_PROJECT, "model_output", "weights", "best.pt")
VAL_IMAGES   = r"C:\YOLO_gesture\dataset_split\val\images"
VAL_LABELS   = r"C:\YOLO_gesture\dataset_split\val\labels"
OUT_DIR      = os.path.join(_PROJECT, "model_output", "training_plots")

# Classes to visualize (index → name)
CLASSES      = ['0','1','2','3','4','5','6','7','8','9','B']
TARGET_IDS   = {8, 9, 10}   # class 8, 9, B (index 10)

# Grid layout
CELL_SIZE    = 256           # each image cell (px)
COLS         = 4             # images per row
MAX_IMAGES   = 16            # cap at 16 per batch image
CONF         = 0.4

# Colours per class (BGR) — bright, high-contrast palette
PALETTE = [
    (  0, 200, 255),  # 0  — yellow
    (  0, 165, 255),  # 1  — orange
    (  0,   0, 255),  # 2  — red
    (255,   0, 180),  # 3  — pink
    (255,   0, 255),  # 4  — magenta
    (255,   0,   0),  # 5  — blue
    (255, 140,   0),  # 6  — dodger blue
    (  0, 255,   0),  # 7  — green
    (  0, 255, 200),  # 8  — aqua
    ( 60,  20, 220),  # 9  — crimson
    (128,   0, 128),  # B  — purple
]
def put_label(img, text, x, y, color):
    """Draw text with a dark background for readability."""
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.8
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), (0,0,0), -1)
    cv2.putText(img, text, (x+2, y), font, scale, color, thickness)


def draw_boxes_labels(img, label_path, classes):
    """Draw ground-truth boxes on image."""
    h, w = img.shape[:2]
    if not Path(label_path).exists():
        return img
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            color = PALETTE[cls % len(PALETTE)]
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 3)
            label = classes[cls] if cls < len(classes) else str(cls)
            put_label(img, label, x1, max(y1-4, 20), color)
    return img


def draw_boxes_preds(img, result, classes):
    """Draw prediction boxes on image."""
    boxes = result.boxes
    if boxes is None:
        return img
    for box in boxes:
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = PALETTE[cls % len(PALETTE)]
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 3)
        label = f"{classes[cls] if cls < len(classes) else cls} {conf:.2f}"
        put_label(img, label, x1, max(y1-4, 20), color)
    return img


def make_grid(images, cols=4, cell_size=256):
    """Arrange a list of images into a grid."""
    rows = (len(images) + cols - 1) // cols
    grid = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        resized = cv2.resize(img, (cell_size, cell_size))
        grid[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size] = resized
    return grid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🔍 Loading model...")
    model = YOLO(BEST_PT)

    # Collect val images whose label contains a target class
    val_imgs = sorted(Path(VAL_IMAGES).glob("*.*"))
    selected = []
    for img_path in val_imgs:
        lbl_path = Path(VAL_LABELS) / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        with open(lbl_path) as f:
            ids = {int(l.split()[0]) for l in f if l.strip()}
        if ids & TARGET_IDS:
            selected.append(img_path)
        if len(selected) >= MAX_IMAGES:
            break

    print(f"   Found {len(selected)} val images for classes 8, 9, B")

    label_imgs = []
    pred_imgs  = []

    print("🖼  Running predictions...")
    for img_path in selected:
        lbl_path = Path(VAL_LABELS) / (img_path.stem + ".txt")

        # Ground-truth panel
        img_gt = cv2.imread(str(img_path))
        img_gt = draw_boxes_labels(img_gt.copy(), str(lbl_path), CLASSES)
        label_imgs.append(img_gt)

        # Prediction panel
        result   = model.predict(str(img_path), conf=CONF, verbose=False)[0]
        img_pred = cv2.imread(str(img_path))
        img_pred = draw_boxes_preds(img_pred.copy(), result, CLASSES)
        pred_imgs.append(img_pred)

    # Build grids
    grid_labels = make_grid(label_imgs, cols=COLS, cell_size=CELL_SIZE)
    grid_preds  = make_grid(pred_imgs,  cols=COLS, cell_size=CELL_SIZE)

    out_labels = os.path.join(OUT_DIR, "val_batch_89B_labels.jpg")
    out_preds  = os.path.join(OUT_DIR, "val_batch_89B_pred.jpg")

    cv2.imwrite(out_labels, grid_labels, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(out_preds,  grid_preds,  [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"\n✅ Saved:")
    print(f"   Labels : {out_labels}")
    print(f"   Preds  : {out_preds}")


if __name__ == "__main__":
    main()
