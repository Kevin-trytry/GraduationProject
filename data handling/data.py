import os
import json
import cv2
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ==============================================================
# Settings
# ==============================================================
dataset_dir  = './New data set'          # Dataset folder (subfolders = class labels: 0-9, B)
output_file  = 'bounding_boxes.labels'   # Edge Impulse output file

# Confidence threshold: raise it (e.g. 0.6) to reduce false positives,
# lower it (e.g. 0.3) if hands are being missed.
CONF_THRESH  = 0.40

# Padding ratio applied to the YOLO box after detection.
# Positive value = expand box outward (e.g. 0.05 adds 5% padding).
# Negative value = shrink box inward (e.g. -0.10 removes 10% from each side).
# 0.03 adds ~3% on each side so fingertips / wrist edges are not clipped.
PAD_RATIO    = 0.03

# FALLBACK: If YOLO detects NO hand in an image, use the FULL image as
# bounding box instead of skipping it (True = use full image, False = skip).
FALLBACK_FULL_IMAGE = True

# Save preview images with boxes drawn on them (True / False)
PREVIEW_MODE = True
preview_dir  = './preview'

# Valid class labels (subfolders that should be processed)
VALID_LABELS = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B'}
# ==============================================================

# 1. Load YOLOv8 hand detection model
# Using Bingsu/adetailer hand_yolov8n.pt (public, free, ~6 MB)
# keremberke/yolov8n-hand-detection is no longer publicly accessible.
print('[INFO] Downloading / loading YOLOv8 hand detection model...')
model_path = hf_hub_download(
    repo_id='Bingsu/adetailer',
    filename='hand_yolov8n.pt'
)
model = YOLO(model_path)
print('[INFO] Model ready.\n')

# 2. Edge Impulse bounding-box label structure
info_labels = {
    "version": 1,
    "type": "bounding-box-labels",
    "boundingBoxes": {}
}

if PREVIEW_MODE and not os.path.exists(preview_dir):
    os.makedirs(preview_dir)

# Stats
stats = {
    "total": 0,
    "labeled_yolo": 0,
    "labeled_fallback": 0,
    "skipped": 0,
    "boxes": 0
}

print('=' * 60)
print('  Hand Gesture Object Detection - Edge Impulse Labeling')
print('  Labels: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, B')
print('=' * 60)

# 3. Walk through each class subfolder
for label_name in sorted(os.listdir(dataset_dir)):
    class_dir = os.path.join(dataset_dir, label_name)
    if not os.path.isdir(class_dir):
        continue

    # Only process valid gesture folders
    if label_name not in VALID_LABELS:
        print(f'\n[SKIP] Folder "{label_name}" is not a valid gesture label, skipping.')
        continue

    img_files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f'\n[Class] {label_name}  ({len(img_files)} images)')

    if PREVIEW_MODE:
        os.makedirs(os.path.join(preview_dir, label_name), exist_ok=True)

    yolo_count      = 0
    fallback_count  = 0
    skipped_count   = 0

    for img_name in img_files:
        stats["total"] += 1
        img_path = os.path.join(class_dir, img_name)
        image    = cv2.imread(img_path)

        if image is None:
            print(f'  [WARN] Cannot read: {img_name} -- skipped')
            skipped_count    += 1
            stats["skipped"] += 1
            continue

        img_h, img_w = image.shape[:2]
        boxes_list   = []
        used_fallback = False

        # ── Run YOLO detection ──────────────────────────────────
        results = model(img_path, conf=CONF_THRESH, verbose=False)

        # Collect all detections sorted by confidence (highest first)
        detections = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                detections.append((conf, box))

        detections.sort(key=lambda d: d[0], reverse=True)

        # Keep ONLY the BEST (highest-confidence) detection.
        # If YOLO finds multiple hands in one image (e.g. false second hit),
        # we take only one box to avoid duplicate labels on the same image.
        if detections:
            _, best_box = detections[0]
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            box_w = x2 - x1
            box_h = y2 - y1

            # Apply padding (positive = expand, negative = shrink)
            pad_x = int(box_w * PAD_RATIO)
            pad_y = int(box_h * PAD_RATIO)
            x1 = max(0,     x1 - pad_x)
            y1 = max(0,     y1 - pad_y)
            x2 = min(img_w, x2 + pad_x)
            y2 = min(img_h, y2 + pad_y)

            final_w = x2 - x1
            final_h = y2 - y1

            if final_w > 10 and final_h > 10:
                boxes_list.append({
                    "label":  label_name,
                    "x":      x1,
                    "y":      y1,
                    "width":  final_w,
                    "height": final_h
                })

        # ── Fallback: full image as bounding box ────────────────
        if not boxes_list and FALLBACK_FULL_IMAGE:
            boxes_list = [{
                "label":  label_name,
                "x":      0,
                "y":      0,
                "width":  img_w,
                "height": img_h
            }]
            used_fallback = True

        # ── Store result ────────────────────────────────────────
        if boxes_list:
            # Edge Impulse uses the bare filename as key
            info_labels["boundingBoxes"][img_name] = boxes_list
            stats["boxes"] += len(boxes_list)

            if used_fallback:
                fallback_count       += 1
                stats["labeled_fallback"] += 1
            else:
                yolo_count           += 1
                stats["labeled_yolo"] += 1

            # ── Draw preview ────────────────────────────────────
            if PREVIEW_MODE:
                preview_img = image.copy()
                for b in boxes_list:
                    x1_p = b["x"]
                    y1_p = b["y"]
                    x2_p = x1_p + b["width"]
                    y2_p = y1_p + b["height"]
                    color = (0, 200, 0) if not used_fallback else (0, 165, 255)
                    cv2.rectangle(preview_img, (x1_p, y1_p), (x2_p, y2_p), color, 2)
                    cv2.putText(preview_img, b["label"],
                                (x1_p, max(20, y1_p - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.imwrite(os.path.join(preview_dir, label_name, img_name), preview_img)
        else:
            skipped_count    += 1
            stats["skipped"] += 1

    print(f'  [YOLO]     Detected  : {yolo_count}')
    print(f'  [FALLBACK] Full-img  : {fallback_count}')
    print(f'  [SKIP]     Error     : {skipped_count}')

# 4. Write Edge Impulse .labels file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(info_labels, f, indent=4, ensure_ascii=False)

# 5. Summary
total_labeled = stats["labeled_yolo"] + stats["labeled_fallback"]
print('\n' + '=' * 60)
print('  Summary')
print('=' * 60)
print(f'  Total images processed : {stats["total"]}')
print(f'  Labeled (YOLO boxes)   : {stats["labeled_yolo"]}')
print(f'  Labeled (full-image)   : {stats["labeled_fallback"]}')
print(f'  Skipped (read error)   : {stats["skipped"]}')
print(f'  Total bounding boxes   : {stats["boxes"]}')
print(f'  Output label file      : {output_file}')
if PREVIEW_MODE:
    print(f'  Preview folder         : {preview_dir}/')
print('=' * 60)
print('  [Done] Labeling complete!')
print('=' * 60)
print()
print('  --> Upload the file below to Edge Impulse:')
print(f'      {os.path.abspath(output_file)}')
print('  --> In Edge Impulse: Data acquisition > Upload data >')
print('      Select "Bounding box labels (.labels)" format')
print('=' * 60)