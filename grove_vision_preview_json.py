import cv2
import serial
import threading
import time
import argparse
import numpy as np
import queue
import json
import base64
import math

# 全域變數，用於執行緒間通訊
frame_queue = queue.Queue(maxsize=2)
current_detections = []
is_running = True

GESTURE_CLASSES = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'B'
}

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def serial_reader(ser):
    global current_detections, is_running
    buffer = bytearray()
    
    while is_running:
        try:
            if ser.in_waiting:
                chunk = ser.read(ser.in_waiting)
                buffer.extend(chunk)
                
                while b'\n' in buffer:
                    line_end = buffer.find(b'\n')
                    line = buffer[:line_end].decode('utf-8', errors='ignore').strip()
                    buffer = buffer[line_end+1:]
                    
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            data = json.loads(line)
                            if data.get('name') == 'INVOKE' and 'data' in data:
                                payload = data['data']
                                
                                # 解析辨識框
                                if 'boxes' in payload:
                                    current_detections = payload['boxes']
                                    
                                # 解析影像
                                if 'image' in payload and payload['image']:
                                    img_data = base64.b64decode(payload['image'])
                                    np_arr = np.frombuffer(img_data, np.uint8)
                                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                                    
                                    if img is not None:
                                        if frame_queue.full():
                                            try:
                                                frame_queue.get_nowait()
                                            except queue.Empty:
                                                pass
                                        frame_queue.put(img)
                        except json.JSONDecodeError:
                            pass
            else:
                time.sleep(0.001)
        except Exception as e:
            if is_running:
                print(f'\n[串列埠錯誤] {e}')
                is_running = False
            break

def draw_detections(img, detections, orig_w, orig_h, scale):
    for det in detections:
        if len(det) >= 6:
            x_json, y_json, w_json, h_json, score, cls_id = det[:6]
            cls_id = int(cls_id)
            
            # Himax 韌體內部錯誤地將座標多乘上了 192，且使用無號數 uint16
            # 當模型輸出雜訊（微小負數）時，會發生 Underflow 變成 65535 左右的極大值
            if w_json > 32767: w_json -= 65536
            if h_json > 32767: h_json -= 65536
            if x_json > 32767: x_json -= 65536
            if y_json > 32767: y_json -= 65536
            
            # 除以 192 還原正確的像素座標
            x = x_json / 192.0
            y = y_json / 192.0
            w = w_json / 192.0
            h = h_json / 192.0
            
            # 信心度還原 (Raw Logit -> Sigmoid)
            raw_logit = score / 100.0
            real_score = sigmoid(raw_logit) * 100.0
            
            # 如果座標亂飛，直接忽略不畫
            if w <= 0 or h <= 0 or x < -100 or y < -100:
                continue

            # 縮放至顯示視窗大小
            x1 = int(max(0, x * scale))
            y1 = int(max(0, y * scale))
            x2 = int(min(orig_w * scale, (x + w) * scale))
            y2 = int(min(orig_h * scale, (y + h) * scale))
            
            label = f'{GESTURE_CLASSES.get(cls_id, str(cls_id))} ({real_score:.1f}%)'
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(20, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def main():
    global is_running
    parser = argparse.ArgumentParser(description="Grove Vision AI V2 Preview")
    parser.add_argument('--port', type=str, default='COM3', help='Serial port')
    parser.add_argument('--baud', type=int, default=921600, help='Baud rate')
    parser.add_argument('--scale', type=float, default=2.0, help='Display scale')
    args = parser.parse_args()

    print("\nGrove Vision AI V2 JSON Preview (High Performance)")
    print(f"Port: {args.port} @ {args.baud}")
    print("Press Q or ESC to exit\n")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
        print("[連線成功]")
    except Exception as e:
        print(f"[連線失敗] {e}")
        return

    reader_thread = threading.Thread(target=serial_reader, args=(ser,), daemon=True)
    reader_thread.start()

    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for camera...", (150, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    fps_counter = 0
    fps_start = time.time()
    fps = 0.0

    while is_running:
        try:
            frame = frame_queue.get(timeout=0.05)
            fps_counter += 1
            
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = time.time()
                
            orig_h, orig_w = frame.shape[:2]
            display = cv2.resize(frame, (int(orig_w * args.scale), int(orig_h * args.scale)),
                                 interpolation=cv2.INTER_NEAREST)
            draw_detections(display, current_detections, orig_w, orig_h, args.scale)
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Grove Vision AI V2', display)
            
        except queue.Empty:
            if fps == 0:
                cv2.imshow('Grove Vision AI V2', placeholder)
        
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            is_running = False
            break
            
    ser.close()
    cv2.destroyAllWindows()
    print("[結束]")

if __name__ == "__main__":
    main()
