"""
grove_vision_preview.py
=========================================
PC 端接收 Grove Vision AI Module V2 的影像串流與推論結果，
並在 OpenCV 視窗中顯示即時畫面與 bounding box。

用法:
    pip install pyserial opencv-python
    python grove_vision_preview.py --port COM3

通訊協定 (tflm_yolov8_od 韌體):
  - 開發板透過 UART (921600 baud) 同時送出:
    1. SPI 傳輸 JPEG 影像 (透過板子上的 SPI master -> USB bridge)
    2. UART 傳輸 JSON 推論結果 (bounding boxes)
  - JSON 格式: {"type": 1, "name": "INVOKE", "code": 0, "data": {"count": N, "boxes": [[x,y,w,h,score,class], ...]}}
  
  影像部分透過 SPI 協定傳輸，需用 hx_drv_spi_mst_protocol_write_sp
  在標準接法下，影像是透過特殊的二進位協定傳輸，
  本腳本接收 UART 上的 JPEG 串流 (DATA_TYPE_JPG)

注意:
  - Windows: 請使用 Microsoft Edge 打開 Himax_AI_web_toolkit/index.html 是最簡單的方法
  - 本腳本是備用的 Python 方案
"""

import serial
import struct
import cv2
import numpy as np
import json
import argparse
import time
import threading
import queue
from io import BytesIO

# ===================== 設定 =====================
BAUD_RATE = 921600
TIMEOUT = 2.0

# Himax SPI 協定的 magic bytes
HIMAX_PKG_HEADER = b'\x2B\x00'  # DATA_TYPE_JPG header start

# 手勢類別名稱 (對應我們的 11 個類別)
GESTURE_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "B"]

# bounding box 顏色 (BGR)
BBOX_COLOR = (0, 255, 100)      # 綠色
TEXT_COLOR = (255, 255, 255)    # 白色
BG_COLOR   = (0, 180, 70)      # 深綠背景

# ===================== JPEG 串流解析 =====================
def find_jpeg_in_stream(data: bytes):
    """在原始串流中找到 JPEG 圖片 (SOI=FFD8, EOI=FFD9)"""
    results = []
    start = 0
    while True:
        soi = data.find(b'\xFF\xD8', start)
        if soi == -1:
            break
        eoi = data.find(b'\xFF\xD9', soi)
        if eoi == -1:
            break
        results.append(data[soi:eoi+2])
        start = eoi + 2
    return results

# ===================== JSON 結果解析 =====================
def parse_json_results(line: str):
    """解析開發板送出的 JSON 推論結果"""
    try:
        line = line.strip()
        if not line or '{' not in line:
            return None
        # 找到 JSON 開始位置
        start = line.find('{')
        data = json.loads(line[start:])
        if data.get('name') == 'INVOKE' and 'data' in data:
            boxes_raw = data['data'].get('boxes', [])
            detections = []
            for box in boxes_raw:
                if len(box) >= 6:
                    x, y, w, h, score, class_id = box[:6]
                    detections.append({
                        'x': int(x), 'y': int(y),
                        'w': int(w), 'h': int(h),
                        'score': float(score) / 100.0,  # 開發板送的是 0-100
                        'class_id': int(class_id),
                        'class_name': GESTURE_CLASSES[int(class_id)] if int(class_id) < len(GESTURE_CLASSES) else '?'
                    })
            return detections
    except Exception:
        pass
    return None

# ===================== 繪圖函式 =====================
def draw_detections(frame, detections, img_orig_w, img_orig_h):
    """在 frame 上繪製 bounding box 和類別標籤"""
    h, w = frame.shape[:2]
    scale_x = w / max(img_orig_w, 1)
    scale_y = h / max(img_orig_h, 1)
    
    for det in detections:
        x = int(det['x'] * scale_x)
        y = int(det['y'] * scale_y)
        bw = int(det['w'] * scale_x)
        bh = int(det['h'] * scale_y)
        
        # 畫框
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), BBOX_COLOR, 2)
        
        # 標籤背景
        label = f"{det['class_name']} {det['score']:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 4, y), BG_COLOR, -1)
        
        # 標籤文字
        cv2.putText(frame, label, (x + 2, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

def draw_overlay(frame, fps, n_detections):
    """在左上角畫 FPS 和偵測數量"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (220, 60), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
    cv2.putText(frame, f"Detections: {n_detections}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 255), 2)

# ===================== 串列埠讀取執行緒 =====================
def serial_reader(ser, jpeg_queue, detection_queue):
    """從 UART 持續讀取資料，分離 JPEG 圖片和 JSON 結果"""
    buffer = bytearray()
    line_buffer = ""
    
    print("[串列埠] 開始讀取...")
    
    while True:
        try:
            raw = ser.read(4096)
            if not raw:
                continue
            
            buffer.extend(raw)
            
            # 嘗試解析 JPEG
            jpegs = find_jpeg_in_stream(bytes(buffer))
            if jpegs:
                for jpg in jpegs:
                    jpeg_queue.put(jpg)
                # 清除已處理的部分
                last_eoi = bytes(buffer).rfind(b'\xFF\xD9')
                if last_eoi != -1:
                    buffer = buffer[last_eoi + 2:]
            
            # 限制 buffer 大小
            if len(buffer) > 500000:
                buffer = buffer[-100000:]
            
            # 嘗試解析 JSON 行 (文字部分)
            try:
                text = raw.decode('utf-8', errors='ignore')
                line_buffer += text
                lines = line_buffer.split('\n')
                for line in lines[:-1]:
                    dets = parse_json_results(line)
                    if dets is not None:
                        detection_queue.put(dets)
                line_buffer = lines[-1]
            except Exception:
                pass
                
        except serial.SerialException as e:
            print(f"[串列埠錯誤] {e}")
            break

# ===================== 主程式 =====================
def main():
    parser = argparse.ArgumentParser(description='Grove Vision AI V2 預覽工具')
    parser.add_argument('--port', type=str, default='COM3',
                        help='開發板的 COM Port，例如 COM3 或 /dev/ttyACM0')
    parser.add_argument('--baud', type=int, default=921600,
                        help='Baud rate (預設 921600)')
    parser.add_argument('--scale', type=float, default=2.0,
                        help='顯示視窗放大倍率 (預設 2.0)')
    args = parser.parse_args()
    
    print(f"""
╔═══════════════════════════════════════╗
║  Grove Vision AI V2 手勢辨識預覽      ║
╠═══════════════════════════════════════╣
║  Port: {args.port:<31}║
║  Baud: {args.baud:<31}║
║  按 Q 或 ESC 退出                     ║
╚═══════════════════════════════════════╝
""")
    
    # 嘗試連線
    try:
        ser = serial.Serial(args.port, args.baud, timeout=TIMEOUT)
        print(f"[連線成功] {args.port} @ {args.baud} baud")
    except serial.SerialException as e:
        print(f"[連線失敗] {e}")
        print("\n請確認:")
        print("  1. 開發板已插上 USB")
        print("  2. COM Port 號碼正確 (裝置管理員查看)")
        print("  3. Tera Term 或其他串列軟體已關閉")
        return
    
    jpeg_queue = queue.Queue(maxsize=5)
    detection_queue = queue.Queue(maxsize=20)
    
    # 啟動讀取執行緒
    reader_thread = threading.Thread(
        target=serial_reader,
        args=(ser, jpeg_queue, detection_queue),
        daemon=True
    )
    reader_thread.start()
    
    # 顯示視窗
    cv2.namedWindow('Grove Vision AI V2 - 手勢辨識', cv2.WINDOW_NORMAL)
    
    current_detections = []
    fps_counter = 0
    fps_start = time.time()
    fps = 0.0
    
    # 建立預設空白畫面
    placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for camera...", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    
    print("[顯示] 視窗已開啟，等待影像中...")
    
    while True:
        # 更新偵測結果
        while not detection_queue.empty():
            current_detections = detection_queue.get_nowait()
        
        # 取得最新 JPEG
        frame = None
        while not jpeg_queue.empty():
            jpg_bytes = jpeg_queue.get_nowait()
            try:
                nparr = np.frombuffer(jpg_bytes, np.uint8)
                decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if decoded is not None:
                    frame = decoded
                    fps_counter += 1
            except Exception:
                pass
        
        # FPS 計算
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps = fps_counter / elapsed
            fps_counter = 0
            fps_start = time.time()
        
        # 顯示
        if frame is not None:
            orig_h, orig_w = frame.shape[:2]
            display = cv2.resize(frame, (int(orig_w * args.scale), int(orig_h * args.scale)),
                                 interpolation=cv2.INTER_NEAREST)
            draw_detections(display, current_detections, orig_w, orig_h)
            draw_overlay(display, fps, len(current_detections))
            cv2.imshow('Grove Vision AI V2 - 手勢辨識', display)
        else:
            cv2.imshow('Grove Vision AI V2 - 手勢辨識', placeholder)
        
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # Q 或 ESC
            break
    
    ser.close()
    cv2.destroyAllWindows()
    print("[結束] 已關閉連線")

if __name__ == '__main__':
    main()
