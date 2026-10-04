"""
HimaxWiseEye2 手勢辨識即時預覽 + Frame 抓取版本（單軌 UART 模式）

傳輸架構：
  韌體以單一 UART 通道，將 JPEG 影像（Base64）與 JSON 推論結果
  封裝在同一個 INVOKE 訊息內送出 → PC 端一次解析完整資料。

韌體座標格式：
  boxes: [x, y, w, h, score, cls_id]
  - x, y  為 bounding box 左上角像素座標（基於 camera raw 解析度）
  - w, h  為寬高（基於 camera raw 解析度）
  - score 為信心度 0~100
  - cls_id 為手勢類別 0~10
  重要：韌體直接輸出像素座標，不需要位元移位轉換。

新增功能：
  1. 偵測到手勢時，將 frame 存成圖片（含 bounding box、手勢標籤、信心度)
  2. 同步將每個 frame 的 log 資訊寫入 CSV / TXT 檔，方便對照圖片

用法範例:
  python grove_vision_capture.py --port COM3
  python grove_vision_capture.py --port COM3 --cam-width 640 --cam-height 480
  python grove_vision_capture.py --port COM3 --save-dir captures --threshold 30
  python grove_vision_capture.py --port COM3 --save-all       # 有無手勢都存圖
  python grove_vision_capture.py --port COM3 --debug

重要參數：
  --cam-width / --cam-height: 相機 raw 解析度（短點韋體 APP_DP_RES_RGB640x480 對應 640x480）
  若 JPEG 圖片為 320x240（韋體 2x 降調樣）且座標基於 640x480，
  則需要指定 --cam-width 640 --cam-height 480 來正確縮放座標。
"""

import sys

# Windows 主控台預設 cp950，一旦把輸出導向檔案（> fw_log.txt），
# 程式裡的 ✓ 、× 等字元會觸發 UnicodeEncodeError 讓整支程式當掉。
# 強制 UTF-8 輸出，讓 log 能完整寫到檔案。
if sys.platform == 'win32':
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

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
import os
import csv
from collections import deque, Counter
from datetime import datetime

# ─── 全域狀態 ─────────────────────────────────────────────────────────────────
frame_queue        = queue.Queue(maxsize=2)
current_detections = []
detections_lock    = threading.Lock()
is_running         = True
debug_mode         = False

# ─── 手勢標籤（對應 Grove Vision AI V2 官方手勢模型）────────────────────────
GESTURE_CLASSES = {
    0:  '0',
    1:  '1',
    2:  '2',
    3:  '3',
    4:  '4',
    5:  '5',
    6:  '6',
    7:  '7',
    8:  '8',
    9:  '9',
    10: 'B',
}

# 每個 class 對應不同顏色 (BGR)
CLASS_COLORS = [
    (0, 255, 0),   (0, 200, 255), (255, 100, 0), (0, 100, 255),
    (200, 0, 255), (0, 255, 200), (255, 200, 0), (100, 255, 0),
    (255, 0, 100), (128, 255, 128), (255, 128, 0),
]

def get_color(cls_id: int):
    return CLASS_COLORS[cls_id % len(CLASS_COLORS)]

# ─── 座標轉換 ─────────────────────────────────────────────────────────────────
def to_signed_16(v):
    v = int(v)
    return v - 65536 if v > 32767 else v

def normalize_score(score_val) -> float:
    s = float(score_val)
    if isinstance(score_val, float) and s <= 1.0:
        return s * 100.0
    elif s <= 100.0:
        return s
    elif s <= 255.0:
        return (s / 255.0) * 100.0
    else:
        s = to_signed_16(s)
        return max(0.0, min(100.0, s))

# ─── JSON 解析 ────────────────────────────────────────────────────────────────
class GestureVoter:
    """多幀投票 + 框平滑，用來壓掉單幀跳動。

    問題背景（2026-09-09 實測）：
      同一個 Shaka 手勢連續 4 秒，預測在 cls8 / cls9 / cls6 之間亂跳；
      整段 80 秒的 session 裡有 41 個「只出現 1 幀」的類別跳動。

    做法：
      看最近 window 幀的「本幀最高分類別」（沒偵測到就投 None），
      只有當某個類別拿到 >= min_votes 票時才輸出它。
      模型猶豫不決時寧可不輸出，也不要顯示錯的標籤。

      hold：贏家一旦確立，後續幾幀即使票數不足也維持住，避免手勢
      中途因為一兩幀雜訊就整個消失、閃爍。

      box_alpha：框位置做指數平滑，0 = 完全不動，1 = 完全跟隨最新。
    """

    def __init__(self, window=5, min_votes=3, box_alpha=0.5, hold=3):
        self.votes      = deque(maxlen=window)
        self.min_votes  = min_votes
        self.box_alpha  = box_alpha
        self.hold_max   = hold
        self.hold       = 0
        self.stable_cls = None
        self.stable_box = None
        self.stable_sc  = 0.0
        self.n_raw      = 0   # 原始有偵測的幀數
        self.n_stable   = 0   # 投票後有輸出的幀數

    def update(self, dets):
        top = max(dets, key=lambda d: d[4]) if dets else None
        if top is not None:
            self.n_raw += 1
        self.votes.append(None if top is None else int(top[5]))

        winner, count = Counter(self.votes).most_common(1)[0]

        if winner is not None and count >= self.min_votes:
            if winner != self.stable_cls:
                self.stable_cls = winner
                self.stable_box = None      # 換手勢就重設平滑，不要拖影
            self.hold = self.hold_max
        elif self.hold > 0:
            self.hold -= 1                  # 維持住，避免閃爍
        else:
            self.stable_cls = None
            self.stable_box = None

        if self.stable_cls is None:
            return []

        # 用本幀中屬於 stable_cls 的框來更新位置
        same = [d for d in dets if int(d[5]) == self.stable_cls]
        if same:
            b  = max(same, key=lambda d: d[4])
            nb = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
            if self.stable_box is None:
                self.stable_box = nb
            else:
                a = self.box_alpha
                self.stable_box = [(1 - a) * o + a * n
                                   for o, n in zip(self.stable_box, nb)]
            self.stable_sc = float(b[4])

        if self.stable_box is None:
            return []

        self.n_stable += 1
        x, y, w, h = self.stable_box
        return [[x, y, w, h, self.stable_sc, self.stable_cls]]


def parse_detections(payload: dict) -> list:
    if 'boxes' in payload and payload['boxes']:
        result = []
        for item in payload['boxes']:
            if len(item) >= 6:
                x, y, w, h, score, cls_id = item[:6]
                result.append([x, y, w, h, normalize_score(score), int(cls_id)])
            elif len(item) == 5:
                x, y, w, h, score = item
                result.append([x, y, w, h, normalize_score(score), 0])
        return result

    if 'results' in payload and payload['results']:
        result = []
        for item in payload['results']:
            if not isinstance(item, dict):
                continue
            x   = item.get('x', item.get('cx', 0))
            y   = item.get('y', item.get('cy', 0))
            w   = item.get('w', item.get('width', 0))
            h   = item.get('h', item.get('height', 0))
            s   = item.get('score', item.get('confidence', 0))
            cls = int(item.get('target', item.get('class', item.get('cls', 0))))
            result.append([x, y, w, h, normalize_score(s), cls])
        return result

    return []

# ─── 串列埠讀取執行緒 ─────────────────────────────────────────────────────────
def serial_reader(ser, raw_debug=False):
    global current_detections, is_running

    buffer     = bytearray()
    json_count = 0
    bbox_count = 0
    invoke_count = 0

    print('[串列埠] 讀取執行緒啟動，等待韌體資料...')

    while is_running:
        try:
            if ser.in_waiting:
                chunk = ser.read(ser.in_waiting)

                # ── raw debug 模式：直接印出原始位元組 ────────────────────────
                if raw_debug:
                    try:
                        txt = chunk.decode('utf-8', errors='replace')
                        print(f'[RAW] {repr(txt[:300])}')
                    except Exception:
                        pass

                buffer.extend(chunk)

                while b'\n' in buffer:
                    idx  = buffer.find(b'\n')
                    line = buffer[:idx].decode('utf-8', errors='ignore').strip()
                    buffer = buffer[idx + 1:]

                    # 過濾空行與非 JSON 行
                    if not line:
                        continue
                    if not (line.startswith('{') and line.endswith('}')):
                        if debug_mode and line:
                            # 不截斷：韌體的 [CLS-DBG] / [filter] / OUTPUT LAYOUT
                            # 這幾行都超過 80 字元，截掉就看不到後面的類別分數
                            print(f'[FW] {line}')
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        if json_count < 10 or debug_mode:
                            print(f'[JSON解析失敗] {line[:120]}')
                        continue

                    json_count += 1
                    msg_name = data.get('name', '')

                    # ── 非 INVOKE 訊息：印出前幾筆供診斷 ────────────────────
                    if msg_name != 'INVOKE':
                        if json_count <= 20 or debug_mode:
                            print(f'[韌體訊息] name={msg_name!r} (#{json_count})')
                        continue

                    if 'data' not in data:
                        if debug_mode:
                            print(f'[INVOKE 無 data 欄位] #{json_count}')
                        continue

                    payload = data['data']
                    dets    = parse_detections(payload)
                    invoke_count += 1

                    if debug_mode:
                        boxes_raw = payload.get('boxes', payload.get('results', []))
                        print(f'[INVOKE #{invoke_count}] 框數={len(dets)} | '
                              f'原始boxes={str(boxes_raw)[:180]}')
                    elif invoke_count <= 3:
                        # 前幾幀總是印出，方便確認收到資料
                        print(f'[INVOKE #{invoke_count}] 框數={len(dets)} ← 收到推論結果 ✓')

                    with detections_lock:
                        current_detections = dets
                    if dets:
                        bbox_count += 1

                    if invoke_count % 100 == 0:
                        print(f'[統計] INVOKE: {invoke_count} | 有框: {bbox_count} | '
                              f'當前框數: {len(dets)}')

                    # ── 解析影像（單軌模式：INVOKE 必定包含 JPEG）────────────
                    img_b64 = payload.get('image', '')
                    if img_b64:
                        try:
                            img_bytes = base64.b64decode(img_b64)
                            np_arr    = np.frombuffer(img_bytes, np.uint8)
                            img       = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            if img is not None:
                                if frame_queue.full():
                                    try:
                                        frame_queue.get_nowait()
                                    except queue.Empty:
                                        pass
                                frame_queue.put((img, dets))
                            else:
                                if debug_mode:
                                    print(f'[INVOKE #{invoke_count}] JPEG 解碼後為 None，跳過此幀')
                        except Exception as e:
                            if debug_mode:
                                print(f'[影像解碼失敗] {e}')
                    else:
                        # 單軌模式下不應出現無 JPEG 的 INVOKE
                        # 仍建立空白幀送入 queue，避免畫面永久黑屏
                        if invoke_count <= 5 or debug_mode:
                            print(f'[警告] INVOKE #{invoke_count} 無影像資料'
                                  f'（請確認韌體 FRAME_CHECK_DEBUG=0 且 trans_type=0）')
                        img = np.zeros((240, 320, 3), dtype=np.uint8)
                        if frame_queue.full():
                            try:
                                frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        frame_queue.put((img, dets))
            else:
                time.sleep(0.001)

        except Exception as e:
            if is_running:
                print(f'\n[串列埠錯誤] {e}')
                import traceback
                traceback.print_exc()
                is_running = False
            break

    print('[串列埠] 讀取執行緒結束')

# ─── 繪製 Bounding Box ────────────────────────────────────────────────────────
def draw_detections(img, detections, orig_w, orig_h, scale,
                    use_center_coords: bool, threshold: float,
                    box_scale_x: float = 1.0, box_scale_y: float = 1.0,
                    max_box_area: float = 0.95):
    """
    韌體（單軌 UART 模式）直接輸出像素座標，座標格式為：
      [x, y, w, h, score, cls_id]
    其中 x, y 為 bounding box 左上角（或中心點，視 use_center_coords）
    w, h 為寬高，單位為 raw camera 解析度的像素。
    不需要 >>8 位元移位。
    """
    drawn    = 0
    filtered = 0

    for idx, det in enumerate(detections):
        if len(det) < 6:
            continue

        raw_x, raw_y, raw_w, raw_h, score, cls_id = det[:6]
        cls_id = int(cls_id)

        raw_x = int(raw_x)
        raw_y = int(raw_y)
        raw_w = int(raw_w)
        raw_h = int(raw_h)

        if score < threshold:
            filtered += 1
            if debug_mode:
                print(f'[過濾] idx={idx} 信心度 {score:.0f}% < 門檻 {threshold:.0f}%')
            continue

        if raw_w <= 0 or raw_h <= 0:
            filtered += 1
            if debug_mode:
                print(f'[過濾] idx={idx} w/h 無效: w={raw_w}, h={raw_h}')
            continue

        # ── BBox 大小過濾 ────────────────────────────────────────────────
        # 韌體端已有 YOLOV8_MAX_BOX_AREA_RATIO（目前 0.95）。這裡的門檻必須
        # 與韌體一致，否則會出現「韌體有輸出、CSV 有記錄、畫面卻沒有框」。
        # 2026-09-10：本值原為 0.80，比韌體嚴格，導致 cls4 有 72% 的偵測
        # （最高 93%）被靜默丟棄。已改為可設定並預設對齊韌體。
        FRAME_W = orig_w / box_scale_x  # raw 座標系下的畫面寬
        FRAME_H = orig_h / box_scale_y  # raw 座標系下的畫面高
        frame_area = FRAME_W * FRAME_H
        box_area   = raw_w * raw_h
        if frame_area > 0 and max_box_area < 1.0 and \
           (box_area / frame_area) > max_box_area:
            filtered += 1
            # 永遠出聲，不要靜默丟棄 —— 這正是先前難以察覺的原因
            gesture_name = GESTURE_CLASSES.get(cls_id, f'cls#{cls_id}')
            print(f'[未畫框] {gesture_name} {score:.0f}%  '
                  f'框佔畫面 {box_area/frame_area:.0%} > 上限 {max_box_area:.0%}'
                  f'（--max-box-area 1.0 可停用此過濾）')
            continue

        if use_center_coords:
            left = raw_x - raw_w // 2
            top  = raw_y - raw_h // 2
        else:
            left = raw_x
            top  = raw_y

        # box_scale_x/y 用於當 raw 座標是基於不同解析度時的縮放
        # orig_w/orig_h 是 JPEG 圖片的實際像素尺寸
        # 若 raw 座標是基於 camera raw 尺寸（640x480），而 JPEG 是縮圖（320x240），
        # 則 box_scale_x = 640/320 = 2，需除以它再乘以 scale
        adj_left = left  / box_scale_x
        adj_top  = top   / box_scale_y
        adj_w    = raw_w / box_scale_x
        adj_h    = raw_h / box_scale_y

        x1 = int(max(0, adj_left * scale))
        y1 = int(max(0, adj_top  * scale))
        x2 = int(min(img.shape[1] - 1, (adj_left + adj_w) * scale))
        y2 = int(min(img.shape[0] - 1, (adj_top  + adj_h) * scale))

        if x1 >= x2 or y1 >= y2:
            filtered += 1
            if debug_mode:
                print(f'[過濾] idx={idx} bbox 無效: ({x1},{y1})-({x2},{y2})')
            continue

        color        = get_color(cls_id)
        gesture_name = GESTURE_CLASSES.get(cls_id, f'cls#{cls_id}')
        label        = f'{gesture_name}  {score:.0f}%'

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y1 = max(0, y1 - th - 8)
        cv2.rectangle(img, (x1, label_y1), (x1 + tw + 6, y1),
                      tuple(c // 2 for c in color), -1)
        cv2.putText(img, label, (x1 + 3, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        drawn += 1

    h_img, w_img = img.shape[:2]
    stats = f'Detections: {drawn}/{len(detections)}'
    (sw, sh), _ = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(img, stats, (w_img - sw - 10, h_img - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)

    return drawn

# ─── Frame 儲存器 ─────────────────────────────────────────────────────────────
class FrameSaver:
    """
    負責將 frame 與 log 寫入磁碟。
    每次執行會在 save_dir 底下建立獨立的 session 子資料夾：
      captures/
        session_20260719_203921/
          frame_xxx.jpg
          detections_log.csv
          session_log.txt
    """

    def __init__(self, save_dir: str, save_clean: bool = False):
        # ── 每次執行建立獨立 session 子資料夾 ─────────────────────────────
        session_name    = 'session_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir   = os.path.join(save_dir, session_name)
        self.frame_idx  = 0
        os.makedirs(self.save_dir, exist_ok=True)

        # ── raw/：不含任何疊字與框的乾淨原圖 ───────────────────────────────
        # 用途：拿板子的實拍畫面當測試集，直接餵給 best.pt 評估，
        #       不能有畫框和 FPS 文字干擾推論。
        self.save_clean = save_clean
        self.raw_dir = os.path.join(self.save_dir, 'raw')
        if save_clean:
            os.makedirs(self.raw_dir, exist_ok=True)

        # ── CSV：一行一個 detection box ─────────────────────────────────
        csv_path = os.path.join(self.save_dir, 'detections_log.csv')
        self._csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            'frame_idx', 'filename', 'timestamp',
            'det_idx', 'gesture', 'cls_id', 'confidence_pct',
            'raw_x', 'raw_y', 'raw_w', 'raw_h'
        ])

        # ── TXT：session 文字 log ─────────────────────────────────────────────
        txt_path = os.path.join(self.save_dir, 'session_log.txt')
        self._log_file = open(txt_path, 'w', encoding='utf-8')
        self._write_log(f'=== Session started at {datetime.now().isoformat()} ===')
        self._write_log(f'Save directory: {os.path.abspath(self.save_dir)}')

        print(f'[Capture] 儲存資料夾: {os.path.abspath(self.save_dir)}')
        print(f'[Capture] CSV log   : {csv_path}')
        print(f'[Capture] TXT log   : {txt_path}')

    def _write_log(self, msg: str):
        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        line = f'[{ts}] {msg}'
        print(line)
        self._log_file.write(line + '\n')
        self._log_file.flush()

    def save(self, display_img, raw_dets, fps: float,
             threshold: float, orig_w: int, orig_h: int,
             scale: float, use_center: bool,
             box_scale_x: float, box_scale_y: float,
             max_box_area: float = 0.95) -> bool:
        """
        儲存一幀。
        display_img : 已放大（scale 過）的 BGR 影像（尚未畫框）
        raw_dets    : 原始偵測清單 [[x,y,w,h,score,cls], ...]
        回傳 True 表示此幀有被儲存
        """
        self.frame_idx += 1
        ts_str  = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # 毫秒
        fname   = f'frame_{ts_str}_f{self.frame_idx:05d}.jpg'
        fpath   = os.path.join(self.save_dir, fname)

        # ── 乾淨原圖（無框、無疊字），供訓練端當測試集用 ───────────────────
        if self.save_clean:
            cv2.imwrite(os.path.join(self.raw_dir, fname), display_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

        # ── 複製一份用於標注，避免影響畫面顯示 ─────────────────────────────
        annotated = display_img.copy()

        # ── 畫 bounding box（同主迴圈邏輯）──────────────────────────────────
        drawn = draw_detections(
            annotated, raw_dets,
            orig_w, orig_h, scale,
            use_center_coords=use_center,
            threshold=threshold,
            box_scale_x=box_scale_x,
            box_scale_y=box_scale_y,
            max_box_area=max_box_area
        )

        # ── 在圖片左上角加時間戳記與 FPS ────────────────────────────────────
        cv2.putText(annotated, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 2)
        ts_display = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(annotated, ts_display, (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # ── 寫入 JPEG ────────────────────────────────────────────────────────
        cv2.imwrite(fpath, annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # ── 寫入 CSV（每個 detection box 一行）──────────────────────────────
        timestamp_iso = datetime.now().isoformat(timespec='milliseconds')
        if raw_dets:
            for d_idx, det in enumerate(raw_dets):
                if len(det) < 6:
                    continue
                rx, ry, rw, rh, score, cls_id = det[:6]
                cls_id = int(cls_id)
                gesture = GESTURE_CLASSES.get(cls_id, f'cls#{cls_id}')
                self._csv_writer.writerow([
                    self.frame_idx, fname, timestamp_iso,
                    d_idx, gesture, cls_id, f'{score:.1f}',
                    rx, ry, rw, rh
                ])
        else:
            # 有存圖但沒有偵測框（save-all 模式）
            self._csv_writer.writerow([
                self.frame_idx, fname, timestamp_iso,
                -1, 'none', -1, '0.0', 0, 0, 0, 0
            ])
        self._csv_file.flush()

        # ── TXT log ──────────────────────────────────────────────────────────
        det_summary = ', '.join(
            f'{GESTURE_CLASSES.get(int(d[5]), "?")}({d[4]:.0f}%)'
            for d in raw_dets if len(d) >= 6
        ) or 'none'
        self._write_log(
            f'SAVED #{self.frame_idx:05d}  file={fname}  '
            f'fps={fps:.1f}  dets=[{det_summary}]'
        )
        return True

    def close(self):
        self._write_log('=== Session ended ===')
        self._csv_file.close()
        self._log_file.close()


# ─── 主程式 ───────────────────────────────────────────────────────────────────
def main():
    global is_running, debug_mode

    parser = argparse.ArgumentParser(description='HimaxWiseEye2 手勢即時預覽 + Frame 抓取')
    parser.add_argument('--port',       type=str,   default='COM3')
    parser.add_argument('--baud',       type=int,   default=921600)
    parser.add_argument('--scale',      type=float, default=2.0)
    parser.add_argument('--threshold',  type=float, default=20.0,
                        help='信心度門檻 %% (預設 20，與韌體端 score_threshold=0.20 一致。'
                             '設得比韌體高會造成「log 有記錄、畫面沒有框」)')
    parser.add_argument('--crop-guide', type=int, default=240,
                        help='在預覽上畫出韌體 center-crop 的取用範圍（像素，'
                             '需與韌體的 YOLOV8_CROP_SIZE 一致）。0 = 不畫。'
                             '韌體用 letterbox 或 stretch 時請設 0')
    parser.add_argument('--max-box-area', type=float, default=0.95,
                        help='畫框的面積上限比例 (預設 0.95，與韌體端 '
                             'YOLOV8_MAX_BOX_AREA_RATIO 一致；設 1.0 完全停用)')
    parser.add_argument('--model-size', type=int,   default=96)
    parser.add_argument('--cam-width',  type=int,   default=0)
    parser.add_argument('--cam-height', type=int,   default=0)
    parser.add_argument('--cx-cy',      dest='use_center', action='store_true',  default=False)
    parser.add_argument('--no-cx-cy',   dest='use_center', action='store_false')
    parser.add_argument('--debug',      action='store_true')
    parser.add_argument('--raw-debug',  action='store_true',
                        help='印出所有原始 UART 位元組，用於診斷韌體輸出格式')

    # ── 新增抓圖相關參數 ───────────────────────────────────────────────────────
    parser.add_argument('--save-dir',   type=str,   default='captures',
                        help='圖片和 log 的儲存資料夾 (預設: captures/)')
    parser.add_argument('--save-all',   action='store_true',
                        help='所有 frame 都存（預設只在偵測到手勢時存）')
    parser.add_argument('--save-every', type=int,   default=1,
                        help='每 N 幀存一次，降低磁碟寫入量 (預設 1，即每幀都存)')
    parser.add_argument('--save-clean', action='store_true',
                        help='額外在 raw/ 存一份不含框與疊字的乾淨原圖，'
                             '供訓練端當實機測試集（要餵 best.pt 評估時用這個）')

    # ── 多幀投票（壓掉單幀跳動）────────────────────────────────────────────────
    parser.add_argument('--no-vote', dest='vote', action='store_false', default=True,
                        help='關閉多幀投票，顯示每幀的原始結果（診斷用）')
    parser.add_argument('--vote-window', type=int,   default=5,
                        help='投票視窗幀數 (預設 5，約 1 秒 @5.3FPS)')
    parser.add_argument('--vote-min',    type=int,   default=3,
                        help='視窗內至少幾幀一致才輸出 (預設 3)')
    parser.add_argument('--vote-hold',   type=int,   default=3,
                        help='贏家確立後維持幾幀，避免閃爍 (預設 3)')
    parser.add_argument('--box-smooth',  type=float, default=0.5,
                        help='框位置指數平滑係數，0=不動 1=完全跟隨最新 (預設 0.5)')

    args = parser.parse_args()
    debug_mode = args.debug

    voter = GestureVoter(window=args.vote_window, min_votes=args.vote_min,
                         box_alpha=args.box_smooth, hold=args.vote_hold) \
            if args.vote else None

    print('\n══════════════════════════════════════════')
    print(' HimaxWiseEye2 手勢辨識預覽 + Frame 抓取')
    print(' 傳輸模式：單軌 UART（JPEG + JSON 合一）')
    print('══════════════════════════════════════════')
    print(f'  串列埠    : {args.port} @ {args.baud} baud')
    print(f'  顯示倍率  : {args.scale}x')
    print(f'  信心度門檻: {args.threshold:.0f}%  (韌體端 20%)')
    if args.max_box_area >= 1.0:
        print(f'  畫框面積上限: 停用（全部畫出）')
    else:
        print(f'  畫框面積上限: {args.max_box_area:.0%}  (韌體端 95%)')
    print(f'  模型輸入  : {args.model_size}x{args.model_size}')
    if args.crop_guide > 0:
        print(f'  裁切指示框: {args.crop_guide}x{args.crop_guide} 置中'
              f'（手要整隻在黃框內，並盡量撐到內圈虛線）')
    if args.vote:
        print(f'  多幀投票  : 開啟（最近 {args.vote_window} 幀取 {args.vote_min} 票，'
              f'維持 {args.vote_hold} 幀，框平滑 {args.box_smooth}）')
    else:
        print(f'  多幀投票  : 關閉（--no-vote，顯示每幀原始結果）')
    coord_mode = '中心點 (cx,cy)' if args.use_center else '左上角 (x,y)'
    print(f'  座標格式  : {coord_mode}')
    print(f'  調試模式  : {"開啟" if debug_mode else "關閉"}')
    print(f'  原始資料  : {"開啟 (--raw-debug)" if args.raw_debug else "關閉"}')
    print(f'  存圖資料夾: {args.save_dir}')
    print(f'  存圖模式  : {"全部 frame" if args.save_all else "只存有手勢的 frame"}')
    print(f'  存圖間隔  : 每 {args.save_every} 幀')
    print('  按 Q / ESC 退出 | 按 S 手動存一幀')
    print('══════════════════════════════════════════\n')

    # ── 建立串列埠連線 ─────────────────────────────────────────────────────────
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
        print('[連線成功] 等待影像資料...\n')
    except Exception as e:
        print(f'[連線失敗] {e}')
        return

    # ── 等待開發板完整開機 ─────────────────────────────────────────────────────
    # 韌體需要約 1~2 秒初始化感測器與模型，過早連線會收不到資料
    print('[等待] 等待開發板開機 (2.5 秒)...')
    time.sleep(2.5)

    # 清空串列埠緩衝區（丟棄開機雜訊）
    try:
        ser.reset_input_buffer()
    except Exception:
        pass
    print('[就緒] 開始接收資料\n')

    # 傳送 0xFF (255) 給韌體，強制切換到 UART 傳輸模式 (trans_type = 0)
    try:
        ser.write(b'\xff')
        print('[連線] 已發送切換 UART 模式指令 (0xFF)')
    except Exception as e:
        print(f'[連線警告] 發送切換指令失敗: {e}')



    # ── 啟動串列埠讀取執行緒 ───────────────────────────────────────────────────
    reader_thread = threading.Thread(
        target=serial_reader,
        args=(ser, getattr(args, 'raw_debug', False)),
        daemon=True
    )
    reader_thread.start()

    # ── 初始化 FrameSaver ──────────────────────────────────────────────────────
    saver = FrameSaver(args.save_dir, save_clean=args.save_clean)

    # ── 等待畫面 placeholder ───────────────────────────────────────────────────
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, 'Waiting for HimaxWE2...', (100, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    cv2.putText(placeholder, f'Port: {args.port}', (220, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

    # ── 主迴圈 ─────────────────────────────────────────────────────────────────
    fps_counter   = 0
    fps_start     = time.time()
    fps           = 0.0
    frame_count   = 0
    saved_count   = 0
    force_save    = False   # 按 S 手動觸發
    last_display  = None    # 上一幀影像，在等待時繼續顯示
    no_data_warn  = 0       # 無資料警告計數
    luma_mean     = 0.0     # 本幀畫面平均亮度
    luma_std      = 0.0     # 本幀畫面對比
    luma_hist     = []      # 整場的亮度，結束時印平均供 A/B 對照

    # ── 首幀等待提示 ───────────────────────────────────────────────────────────
    print('[提示] 正在等待第一幀推論結果，若超過 5 秒未出現請按 Ctrl+C 後改用:')
    print('       python grove_vision_capture.py --port COM3 --raw-debug')
    print('       以查看韌體實際輸出的原始資料')

    while is_running:
        try:
            # frame_queue 帶著 (img, dets) tuple
            frame_data = frame_queue.get(timeout=0.05)
            frame, frame_dets = frame_data

            fps_counter += 1
            frame_count += 1
            no_data_warn = 0  # 收到資料，重置警告計數

            # 計算 FPS
            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps         = fps_counter / elapsed
                fps_counter = 0
                fps_start   = time.time()
                with detections_lock:
                    n_det = len(current_detections)
                saver._write_log(f'FPS={fps:.1f}  偵測框數={n_det}  已存圖={saved_count}')

            # 第一幀診斷
            orig_h, orig_w = frame.shape[:2]
            if frame_count == 1:
                cam_w = args.cam_width  if args.cam_width  > 0 else orig_w
                cam_h = args.cam_height if args.cam_height > 0 else orig_h
                print(f'\n[診斷] ✓ 收到第 1 幀！')
                print(f'[診斷] JPEG 尺寸      : {orig_w}×{orig_h}px')
                print(f'[診斷] 相機 raw 解析度 : {cam_w}×{cam_h}px')
                print(f'[診斷] 模型輸入        : {args.model_size}×{args.model_size}px')

            coord_cam_w = args.cam_width  if args.cam_width  > 0 else orig_w
            coord_cam_h = args.cam_height if args.cam_height > 0 else orig_h
            box_scale_x = coord_cam_w / orig_w
            box_scale_y = coord_cam_h / orig_h

            # 縮放顯示
            disp_w  = int(orig_w * args.scale)
            disp_h  = int(orig_h * args.scale)
            display = cv2.resize(frame, (disp_w, disp_h),
                                 interpolation=cv2.INTER_NEAREST)

            # 取最新偵測框
            with detections_lock:
                dets_snapshot = list(current_detections)

            # ── 多幀投票：壓掉單幀跳動 ────────────────────────────────────────
            if voter is not None:
                dets_snapshot = voter.update(dets_snapshot)

            # ── 判斷是否需要存圖 ──────────────────────────────────────────────
            has_detection = len(dets_snapshot) > 0
            should_save   = (
                force_save or
                (frame_count % args.save_every == 0 and
                 (args.save_all or has_detection))
            )

            if should_save:
                saver.save(
                    display.copy(),
                    dets_snapshot,
                    fps,
                    args.threshold,
                    orig_w, orig_h,
                    args.scale,
                    args.use_center,
                    box_scale_x, box_scale_y,
                    max_box_area=args.max_box_area
                )
                saved_count += 1
                force_save = False

            # ── 畫面亮度 / 對比（A/B 對照時用來確認光線一致）─────────────────
            # 09-11 的教訓：兩場的平均信心度差 8 個百分點，後來量出來是亮度
            # 差了 23%（95.7 -> 73.3）。不顯示出來就只能靠感覺，A/B 會失效。
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            luma_mean = float(gray.mean())
            luma_std = float(gray.std())
            luma_hist.append(luma_mean)

            # ── 韌體 center-crop 的取用範圍指示框 ─────────────────────────────
            # 韌體 YOLOV8_INPUT_MODE=2 時只看畫面中央的 CROP_SIZE 正方形，
            # 框外的東西模型完全看不到。比手勢時「整隻手要在這個框裡面，
            # 而且盡量填滿它」——訓練圖裡手佔畫面寬度 60~97%。
            if args.crop_guide > 0:
                cs = min(args.crop_guide, orig_w, orig_h)
                gx = int((orig_w - cs) / 2 * args.scale)
                gy = int((orig_h - cs) / 2 * args.scale)
                gw = int(cs * args.scale)
                cv2.rectangle(display, (gx, gy), (gx + gw, gy + gw),
                              (0, 255, 255), 1)
                # 內側 75% 的參考線：手撐到這裡大約就是訓練時的大小
                m = int(gw * 0.125)
                cv2.rectangle(display, (gx + m, gy + m),
                              (gx + gw - m, gy + gw - m), (0, 160, 160), 1)
                cv2.putText(display, f'crop {cs}', (gx + 4, gy + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # ── 主視窗：畫框 + FPS ────────────────────────────────────────────
            draw_detections(display, dets_snapshot,
                            orig_w, orig_h, args.scale,
                            use_center_coords=args.use_center,
                            threshold=args.threshold,
                            box_scale_x=box_scale_x,
                            box_scale_y=box_scale_y,
                            max_box_area=args.max_box_area)

            cv2.putText(display, f'FPS: {fps:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 80, 255), 2)
            cv2.putText(display,
                        f'LUMA {luma_mean:.0f}  CONTRAST {luma_std:.0f}',
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 220, 220), 1)

            mode_txt = 'CX,CY mode' if args.use_center else 'XY mode'
            cv2.putText(display, mode_txt, (disp_w - 130, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

            # ── 存圖狀態指示（右下角）────────────────────────────────────────
            save_txt = f'Saved: {saved_count}'
            (stw, sth), _ = cv2.getTextSize(save_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(display, save_txt, (disp_w - stw - 10, disp_h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150), 1)

            last_display = display  # 記住上一幀，queue 空時繼續顯示
            cv2.imshow('HimaxWiseEye2 Gesture [單軌 UART 模式]', display)

        except queue.Empty:
            no_data_warn += 1
            if frame_count == 0 and no_data_warn % 100 == 0:
                # 首幀等待時顯示 placeholder
                cv2.imshow('HimaxWiseEye2 Gesture [單軌 UART 模式]', placeholder)
            elif last_display is not None:
                # 有過資料但暫時沒有新幀：繼續顯示上一幀（避免畫面消失）
                cv2.imshow('HimaxWiseEye2 Gesture [單軌 UART 模式]', last_display)

            # 超過 5 秒沒有收到任何幀：提示診斷
            if frame_count == 0 and no_data_warn == 1000:  # ≈ 50ms × 1000 = 50s
                print('\n[警告] 超過 50 秒未收到推論資料！')
                print('[診斷] 建議：重新 RESET 開發板，或改用 --raw-debug 模式查看原始輸出')

        # ── 鍵盤處理 ─────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            is_running = False
            break
        elif key in (ord('s'), ord('S')):
            force_save = True
            print('[手動] 手動觸發存圖')

    print(f'\n[結束] 共處理 {frame_count} 幀，已儲存 {saved_count} 張圖片')
    if luma_hist:
        import statistics as _st
        _m = _st.mean(luma_hist)
        print(f'[光線] 本場平均亮度 = {_m:.1f}'
              f'（最低 {min(luma_hist):.0f} / 最高 {max(luma_hist):.0f}）')
        print('       A/B 對照時三場的平均亮度應相差 5 以內，'
              '差太多就不能直接比信心度')
        try:
            saver._write_log(f'[光線] 平均亮度={_m:.1f} '
                             f'min={min(luma_hist):.0f} max={max(luma_hist):.0f}')
        except Exception:
            pass
    if voter is not None:
        print(f'[投票] 原始有偵測 {voter.n_raw} 幀 → 投票後輸出 {voter.n_stable} 幀'
              f'（濾掉 {voter.n_raw - voter.n_stable} 幀不穩定結果）')
    print(f'[結束] 圖片和 log 位於: {os.path.abspath(args.save_dir)}')
    saver.close()
    ser.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
