# -*- coding: utf-8 -*-
"""
elevator_bus.py — 把狀態機的事件送出去，給組員的模擬器接

三個管道，可以同時開，互不影響：

  1. stdout        每個事件印一行 `[ELEV] {json}`
                   → Python 模擬器用 subprocess 接管道最簡單
  2. JSONL 檔案    captures/session_*/elevator_events.jsonl
                   → 事後重播、對帳用
  3. HTTP 伺服器   --elev-http 8765（純標準函式庫，不需要額外套件）
                   → HTML 模擬器用 fetch 輪詢最簡單，已開 CORS

HTTP 介面
─────────
  GET /state
      { "state": "holding", "digits": ["1"], "pos": 2,
        "holding": "2", "hold_elapsed": 1.4, "hold_need": 3.0,
        "hold_ratio": 0.47, "pending_floor": null, "ts": 12.34 }

  GET /events?since=0
      { "next": 7, "events": [ {...}, ... ] }
      拿 next 當下一次的 since，就不會漏也不會重複。

  GET /health
      { "ok": true }
"""

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# 不想讓使用者看到的高頻事件（HTTP 的 /state 已經涵蓋）
NOISY = {'progress', 'confirm_progress', 'gap_progress'}


class EventBus:
    def __init__(self, jsonl_path=None, http_port=None, echo=True,
                 echo_noisy=False, ring=500):
        self.echo = echo
        self.echo_noisy = echo_noisy
        self._lock = threading.Lock()
        self._events = deque(maxlen=ring)
        self._base = 0               # 已經被擠出 ring 的事件數
        self._snapshot = {'state': 'idle', 'digits': [], 'pos': 1}
        self._fh = None
        self._httpd = None

        if jsonl_path:
            try:
                self._fh = open(jsonl_path, 'a', encoding='utf-8')
            except Exception as e:
                print(f'[ELEV] 無法開啟事件檔 {jsonl_path}: {e}')

        if http_port:
            self._start_http(int(http_port))

    # ── 對外 ────────────────────────────────────────────────────────────────
    def publish(self, events, snapshot=None):
        if snapshot is not None:
            with self._lock:
                self._snapshot = dict(snapshot)
        if not events:
            return
        with self._lock:
            for e in events:
                if len(self._events) == self._events.maxlen:
                    self._base += 1
                self._events.append(e)
        for e in events:
            noisy = e.get('event') in NOISY
            if self._fh and not noisy:
                try:
                    self._fh.write(json.dumps(e, ensure_ascii=False) + '\n')
                    self._fh.flush()
                except Exception:
                    pass
            if self.echo and (self.echo_noisy or not noisy):
                print('[ELEV] ' + json.dumps(e, ensure_ascii=False), flush=True)

    def close(self):
        if self._fh:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None
        if self._httpd:
            try:
                self._httpd.shutdown()
            except Exception:
                pass
            self._httpd = None

    # ── HTTP ────────────────────────────────────────────────────────────────
    def _start_http(self, port):
        bus = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass                       # 不要洗畫面

            def _send(self, obj, code=200):
                body = json.dumps(obj, ensure_ascii=False).encode('utf-8')
                self.send_response(code)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Headers', '*')
                self.end_headers()

            def do_GET(self):
                u = urlparse(self.path)
                if u.path == '/health':
                    return self._send({'ok': True})
                if u.path == '/state':
                    with bus._lock:
                        return self._send(dict(bus._snapshot))
                if u.path == '/events':
                    q = parse_qs(u.query)
                    try:
                        since = int(q.get('since', ['0'])[0])
                    except ValueError:
                        since = 0
                    with bus._lock:
                        base = bus._base
                        items = list(bus._events)
                    start = max(0, since - base)
                    out = [e for e in items[start:] if e.get('event') not in NOISY]
                    return self._send({'next': base + len(items), 'events': out})
                self._send({'error': 'not found',
                            'endpoints': ['/state', '/events?since=N', '/health']},
                           404)

        try:
            self._httpd = ThreadingHTTPServer(('127.0.0.1', port), H)
        except OSError as e:
            print(f'[ELEV] HTTP 埠 {port} 開不起來：{e}')
            self._httpd = None
            return
        t = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        t.start()
        print(f'[ELEV] HTTP 介面已啟動 → http://127.0.0.1:{port}/state  '
              f'/events?since=0')


# ── OpenCV 畫面上的狀態顯示 ──────────────────────────────────────────────────
def draw_hud(display, snap, cv2):
    """在預覽視窗左下角畫出目前的輸入狀態。"""
    h, w = display.shape[:2]
    F = cv2.FONT_HERSHEY_SIMPLEX
    x, y = 10, h - 86

    cv2.rectangle(display, (x - 6, y - 22), (x + 320, h - 10), (25, 25, 25), -1)

    digits = snap.get('digits') or []
    slot = [digits[0] if len(digits) > 0 else '_',
            digits[1] if len(digits) > 1 else '_']
    cv2.putText(display, f'FLOOR  [{slot[0]}][{slot[1]}]', (x, y), F, 0.7,
                (0, 255, 200), 2)

    st = snap.get('state', 'idle')
    msg, col = {
        'idle':    ('比出第一位數字', (200, 200, 200)),
        'holding': ('維持不動...',     (0, 220, 255)),
        'gapwait': ('手放下，準備下一位', (255, 200, 0)),
        'confirm': ('確認中，舉手可取消', (120, 255, 120)),
    }.get(st, (st, (200, 200, 200)))

    if st == 'holding':
        r = snap.get('hold_ratio', 0.0)
        g = snap.get('holding') or '?'
        msg = f'{g}  維持中 {snap.get("hold_elapsed",0):.1f}/{snap.get("hold_need",3):.0f}s'
        bw = 300
        cv2.rectangle(display, (x, y + 28), (x + bw, y + 40), (70, 70, 70), -1)
        cv2.rectangle(display, (x, y + 28), (x + int(bw * r), y + 40), col, -1)
    elif st == 'gapwait':
        if snap.get('hand_present'):
            msg = '把手移開才能輸入下一位'
            col = (80, 160, 255)
        else:
            msg = f'手已移開 {snap.get("gap_elapsed",0):.1f}/{snap.get("gap_need",0.8):.1f}s'
        bw = 300
        r = snap.get('gap_ratio', 0.0)
        cv2.rectangle(display, (x, y + 28), (x + bw, y + 40), (70, 70, 70), -1)
        cv2.rectangle(display, (x, y + 28), (x + int(bw * r), y + 40), col, -1)
    elif st == 'confirm':
        msg = (f'即將前往 {snap.get("pending_floor")} '
               f'({snap.get("confirm_remain", 0):.1f}s) 舉手取消')

    cv2.putText(display, msg, (x, y + 22), F, 0.5, col, 1)
