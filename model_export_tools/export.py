import os
from ultralytics import YOLO

# 使用絕對路徑，避免 Windows 相對路徑問題（尤其是含中文的路徑）
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model_output', 'weights', 'best.pt')
data_path  = os.path.join(base_dir, 'model_output', 'data.yaml')

# 切換工作目錄到 base_dir（確保 TF SavedModel 寫入路徑正確）
os.chdir(base_dir)

model = YOLO(model_path)

# 使用 data.yaml 提供校正圖片（quantization calibration）
model.export(
    format='tflite',
    int8=True,
    data=data_path,
    imgsz=192   # 建議 192x192（不超過 240，適合 Himax WiseEye）
)
