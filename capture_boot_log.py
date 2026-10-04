# -*- coding: utf-8 -*-
"""
capture_boot_log.py
===================
純粹的 UART 傾印工具，用來抓「開機 log」。

為什麼需要這支：
  grove_vision_capture.py 啟動後會先等 2.5 秒才開始讀，板子開機當下噴出來的
  TA[]、Ethos-U55 device initialised、schema version、initial done、
  === YOLOv8 OUTPUT LAYOUT === 這些訊息全都會錯過。

用法：
  1. 先執行本程式（它會立刻開始監聽）
  2. 看到「請按板子上的 RST」之後，按一下開發板的 RST 鈕（不要按 BOOT）
  3. 等 15~20 秒讓它跑幾十幀，再按 Ctrl+C 結束

  python capture_boot_log.py --port COM3
  python capture_boot_log.py --port COM3 --out boot_log.txt --seconds 25
"""
import argparse
import sys
import time

if sys.platform == 'win32':
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

import serial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM3')
    ap.add_argument('--baud', type=int, default=921600)
    ap.add_argument('--out', default='boot_log.txt')
    ap.add_argument('--seconds', type=float, default=0,
                    help='跑滿幾秒後自動結束；0 = 一直跑到 Ctrl+C')
    ap.add_argument('--no-jpeg', action='store_true', default=True,
                    help='過濾掉 base64 JPEG 那種超長行，讓 log 好讀（預設開啟）')
    ap.add_argument('--keep-jpeg', dest='no_jpeg', action='store_false')
    a = ap.parse_args()

    ser = serial.Serial(a.port, a.baud, timeout=0.1)
    # 送出切換 UART 模式指令（與 grove_vision_capture.py 相同）
    try:
        ser.write(b'\xff')
    except Exception:
        pass

    f = open(a.out, 'w', encoding='utf-8', errors='replace')
    print('=' * 60)
    print(f'  監聽中：{a.port} @ {a.baud}')
    print(f'  輸出檔：{a.out}')
    print('=' * 60)
    print()
    print('  >>> 現在請按一下開發板上的 RST 鈕（不要按 BOOT）<<<')
    print()
    print('  按下之後會看到開機訊息。跑 15~20 秒後按 Ctrl+C 結束。')
    print('-' * 60)

    buf = bytearray()
    t0 = time.time()
    n_lines = 0
    try:
        while True:
            if a.seconds and (time.time() - t0) > a.seconds:
                break
            chunk = ser.read(4096)
            if not chunk:
                continue
            buf.extend(chunk)
            while b'\n' in buf:
                i = buf.find(b'\n')
                line = bytes(buf[:i]).decode('utf-8', errors='replace').rstrip('\r')
                buf = buf[i + 1:]
                # 過濾 JSON / base64 影像那種超長行，但保留 [DUMP]
                if a.no_jpeg and len(line) > 200 and '[DUMP]' not in line:
                    line = line[:120] + f'   ...(省略 {len(line)-120} 字元)'
                f.write(line + '\n')
                f.flush()
                n_lines += 1
                print(line)
    except KeyboardInterrupt:
        print('\n[結束] 使用者中斷')
    finally:
        f.close()
        ser.close()
        print(f'\n[完成] 共 {n_lines} 行，已寫入 {a.out}')


if __name__ == '__main__':
    main()
