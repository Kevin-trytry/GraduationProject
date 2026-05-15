from ultralytics import YOLO


model = YOLO('model_output/weights/best.pt')
model.predict(source=1, conf=0.4, show=True, stream=True)  # 改成你的編號
