# -*- coding: utf-8 -*-
"""
export_full_int8.py
===================================================
將 YOLOv8n best.pt 導出為「全整數量化 (Full Integer Quantization)」TFLite，
讓 input/output tensor 也變成 int8，才能讓 Ethos-U55 NPU 全速加速。

針對 TensorFlow 在 Windows 上處理中文路徑會失敗的問題：
我們將工作區移至 C:\temp\yolo_export 進行轉換，完成後再將檔案複製回來。
"""
import os
import sys
import subprocess
import shutil
import numpy as np
import tensorflow as tf

# 強制 UTF-8 輸出
sys.stdout.reconfigure(encoding='utf-8')

# ─── 路徑設定 ─────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR     = os.path.join(BASE_DIR, "model_output", "weights")
BEST_PT         = os.path.join(WEIGHTS_DIR, "best.pt")
ORIGINAL_OUTPUT_DIR = os.path.join(WEIGHTS_DIR, "best_saved_model")
CALIB_NPY       = os.path.join(ORIGINAL_OUTPUT_DIR, "tmp_tflite_int8_calibration_images.npy")
HIMAX_VELA_INI  = os.path.join(ORIGINAL_OUTPUT_DIR, "himax_vela.ini")

# 建立純英文暫存目錄以避開 TensorFlow 中文路徑 bug
TEMP_WORKSPACE  = r"C:\temp\yolo_export"
os.makedirs(TEMP_WORKSPACE, exist_ok=True)

TEMP_PT         = os.path.join(TEMP_WORKSPACE, "best.pt")
OUTPUT_TFLITE   = os.path.join(TEMP_WORKSPACE, "best_full_int8.tflite")
TEMP_VELA_INI   = os.path.join(TEMP_WORKSPACE, "himax_vela.ini")

IMAGE_SIZE      = 192
NUM_CALIB_IMGS  = 100

print(f"[INFO] Project path: {BASE_DIR}")
print(f"[INFO] Temp workspace: {TEMP_WORKSPACE}")

# 複製必要檔案到暫存區
shutil.copy(BEST_PT, TEMP_PT)
if os.path.exists(HIMAX_VELA_INI):
    shutil.copy(HIMAX_VELA_INI, TEMP_VELA_INI)

# ─── Step 1: 用 Ultralytics 匯出 SavedModel（float32）────────────────────────
print("\n[Step 1] Exporting SavedModel from best.pt in ASCII path...")

os.chdir(TEMP_WORKSPACE)

from ultralytics import YOLO
model = YOLO(TEMP_PT)

# 先匯出 SavedModel（不量化，保留 float32 圖形）
export_result = model.export(
    format='saved_model',
    imgsz=IMAGE_SIZE,
    int8=False,  
)
print(f"[INFO] Export result: {export_result}")

saved_model_path = os.path.join(TEMP_WORKSPACE, "best_saved_model")

print(f"[INFO] SavedModel path: {saved_model_path}")
assert os.path.exists(os.path.join(saved_model_path, "saved_model.pb")), \
    f"SavedModel not found at {saved_model_path}"

# ─── Step 2: 載入校正資料 (使用真實圖片確保精確度) ──────────────────────────────────
print("\n[Step 2] Preparing REAL calibration data from dataset...")

import cv2
import glob

dataset_path = r"C:\YOLO_gesture\dataset_split\train\images"
image_files = glob.glob(os.path.join(dataset_path, "*.jpg"))

calib_data_list = []
if len(image_files) > 0:
    print(f"  Found {len(image_files)} images in {dataset_path}")
    import random
    random.shuffle(image_files)
    
    for img_path in image_files[:NUM_CALIB_IMGS]:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = img.astype(np.float32) / 255.0
            calib_data_list.append(img)

if len(calib_data_list) > 0:
    calib_data = np.array(calib_data_list)
    print(f"  Loaded real calibration data. Shape: {calib_data.shape}, dtype: {calib_data.dtype}")
else:
    print("  [WARN] Failed to load real images. Using random data (ACCURACY WILL DROP).")
    calib_data = np.random.rand(NUM_CALIB_IMGS, IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)

def representative_dataset():
    for i in range(len(calib_data)):
        yield [calib_data[i:i+1]]

# ─── Step 3: TFLite Full Integer Quantization ─────────────────────────────────
print("\n[Step 3] Running Full Integer Quantization...")
print("  This forces input/output to int8 — required for Ethos-U55 NPU!")

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

print("  Converting (may take 2-5 minutes)...")
tflite_model = converter.convert()

with open(OUTPUT_TFLITE, "wb") as f:
    f.write(tflite_model)

size_kb = os.path.getsize(OUTPUT_TFLITE) / 1024
print(f"  Saved: {OUTPUT_TFLITE} ({size_kb:.1f} KB)")

# ─── Step 4: 驗證 ─────────────────────────────────────────────────────────────
print("\n[Step 4] Verifying quantization...")
interp = tf.lite.Interpreter(model_path=OUTPUT_TFLITE)
interp.allocate_tensors()

inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
all_t = interp.get_tensor_details()
f32 = sum(1 for t in all_t if t['dtype'] == np.float32)
i8  = sum(1 for t in all_t if t['dtype'] == np.int8)

print(f"  Input  dtype: {inp['dtype']}  shape: {inp['shape']}")
print(f"  Output dtype: {out['dtype']}  shape: {out['shape']}")
print(f"  float32 tensors: {f32} / {len(all_t)}")
print(f"  int8    tensors: {i8} / {len(all_t)}")

if inp['dtype'] == np.int8 and out['dtype'] == np.int8:
    print("  [OK] Input/Output are int8 -- Ethos-U55 compatible!")
else:
    print("  [WARN] Still float32 input/output. NPU may not accelerate.")

# ─── Step 5: Vela 編譯 ────────────────────────────────────────────────────────
print("\n[Step 5] Vela compilation for Himax WiseEye2 (Ethos-U55-64)...")

if not os.path.exists(TEMP_VELA_INI):
    print(f"  [ERROR] himax_vela.ini not found: {TEMP_VELA_INI}")
    sys.exit(1)

vela_cmd = [
    sys.executable, "-m", "ethosu.vela",
    "--accelerator-config", "ethos-u55-64",
    "--config", TEMP_VELA_INI,
    "--system-config", "My_Sys_Cfg",
    "--memory-mode", "My_Mem_Mode_Parent",
    "--output-dir", TEMP_WORKSPACE,
    OUTPUT_TFLITE
]

print(f"  Running: {' '.join(vela_cmd)}")
result = subprocess.run(vela_cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')

print("--- Vela stdout ---")
print(result.stdout[-3000:] if result.stdout else "(empty)")
if result.stderr:
    print("--- Vela stderr (last 500 chars) ---")
    print(result.stderr[-500:])

vela_out = os.path.join(TEMP_WORKSPACE, "best_full_int8_vela.tflite")
if os.path.exists(vela_out):
    # 複製回原始目錄
    final_output = os.path.join(ORIGINAL_OUTPUT_DIR, "best_full_int8_vela.tflite")
    shutil.copy(vela_out, final_output)
    
    # 也把 best_full_int8.tflite 複製回來
    shutil.copy(OUTPUT_TFLITE, os.path.join(ORIGINAL_OUTPUT_DIR, "best_full_int8.tflite"))
    
    vela_size = os.path.getsize(final_output) / 1024
    print(f"\n[DONE] Vela output copied to: {final_output} ({vela_size:.1f} KB)")
    print(f"\n  Upload this file to your Himax WiseEye2:")
    print(f"  --> {final_output}")
else:
    print(f"\n[WARN] Expected output not found: {vela_out}")

if result.returncode != 0:
    print(f"\n[ERROR] Vela exited with code {result.returncode}")
