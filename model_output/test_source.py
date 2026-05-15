import cv2
# 試 0~5 號，看哪個能打開
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ source={i} 有效")
        cap.release()
    else:
        print(f"❌ source={i} 無效")