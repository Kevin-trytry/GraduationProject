# -*- coding: utf-8 -*-
"""
elevator_replay.py — 拿既有的 detections_log.csv 回放狀態機

不需要板子。用途是在調參數之前，先看看「如果當時就有這套邏輯，
會鎖定出哪些數字、會不會誤觸發」。

用法：
    python elevator_replay.py captures/session_20260930_190950/detections_log.csv
    python elevator_replay.py <csv> --dwell 2.5 --agree 0.6 --verbose

注意：CSV 裡只有「有偵測的畫面」。沒有偵測的畫面不會存檔，
所以回放時會依時間間隔推算，把空檔補成「無偵測」的幀。
"""
import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime

from elevator_input import ElevatorInput, ElevatorConfig, B_CLASS

NAME2CLS = {str(i): i for i in range(10)}
NAME2CLS['B'] = B_CLASS


def load_frames(path):
    """回傳 [(相對秒數, [det, ...]), ...]，det = [x,y,w,h,conf,cls]。"""
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    if not rows:
        sys.exit('CSV 是空的')
    by_frame = defaultdict(list)
    order = []
    for r in rows:
        f = r['filename']
        if f not in by_frame:
            order.append((f, datetime.fromisoformat(r['timestamp'])))
        cls = r.get('cls_id')
        cls = int(cls) if cls not in (None, '') else NAME2CLS.get(r['gesture'], -1)
        by_frame[f].append([float(r.get('raw_x', 0)), float(r.get('raw_y', 0)),
                            float(r.get('raw_w', 0)), float(r.get('raw_h', 0)),
                            float(r['confidence_pct']), cls])
    order.sort(key=lambda x: x[1])
    t0 = order[0][1]
    return [((ts - t0).total_seconds(), by_frame[f]) for f, ts in order]


def fill_gaps(frames):
    """空檔補成無偵測的幀，讓狀態機看得到「手離開了」。"""
    if len(frames) < 2:
        return frames
    gaps = [frames[i + 1][0] - frames[i][0] for i in range(len(frames) - 1)]
    step = statistics.median(gaps)
    out = []
    for i, (t, dets) in enumerate(frames):
        out.append((t, dets))
        if i + 1 < len(frames):
            nxt = frames[i + 1][0]
            k = t + step
            while k < nxt - step * 0.5:
                out.append((k, []))
                k += step
    return out, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--dwell', type=float, default=3.0)
    ap.add_argument('--trim-head', type=float, default=0.5)
    ap.add_argument('--trim-tail', type=float, default=0.5)
    ap.add_argument('--min-frames', type=int, default=5)
    ap.add_argument('--agree', type=float, default=0.70)
    ap.add_argument('--conf-min', type=float, default=30.0)
    ap.add_argument('--conf-floor', type=float, default=20.0)
    ap.add_argument('--gap', type=float, default=0.8)
    ap.add_argument('--confirm', type=float, default=3.0)
    ap.add_argument('--verbose', action='store_true', help='連 progress 一起印')
    ap.add_argument('--jsonl', type=str, default=None, help='把事件寫成 JSONL')
    a = ap.parse_args()

    cfg = ElevatorConfig(dwell=a.dwell, trim_head=a.trim_head, trim_tail=a.trim_tail,
                         min_frames=a.min_frames, agree_ratio=a.agree,
                         conf_min=a.conf_min, conf_floor=a.conf_floor,
                         gap=a.gap, confirm=a.confirm)

    raw = load_frames(a.csv)
    frames, step = fill_gaps(raw)
    print(f'檔案    : {a.csv}')
    print(f'原始幀  : {len(raw)} 張有偵測，補空檔後 {len(frames)} 幀，'
          f'間隔中位數 {step:.3f}s（{1/step:.1f} fps）')
    print(f'參數    : 停留 {a.dwell}s（掐頭 {a.trim_head} 去尾 {a.trim_tail}）'
          f'　一致度 ≥{a.agree:.0%}　信心中位數 ≥{a.conf_min:.0f}%　'
          f'最少 {a.min_frames} 幀')
    print('─' * 78)

    sm = ElevatorInput(cfg)
    events = []
    for t, dets in frames:
        for e in sm.update(t, dets):
            events.append(e)
            if e['event'].endswith('progress') and not a.verbose:
                continue
            extra = {k: v for k, v in e.items()
                     if k not in ('t', 'event', 'state')}
            print(f"{e['t']:7.2f}s  {e['event']:<16} {extra}")

    print('─' * 78)
    locked = [e for e in events if e['event'] == 'digit_locked']
    rej = [e for e in events if e['event'] == 'digit_rejected']
    disp = [e for e in events if e['event'] == 'floor_dispatched']
    print(f'鎖定 {len(locked)} 個數字：{[e["value"] for e in locked]}')
    print(f'拒絕 {len(rej)} 次：', end='')
    if rej:
        from collections import Counter
        print(dict(Counter(e['reason'] for e in rej)))
    else:
        print('無')
    print(f'送出樓層：{[e["floor"] for e in disp] or "無"}')

    if a.jsonl:
        with open(a.jsonl, 'w', encoding='utf-8') as f:
            for e in events:
                f.write(json.dumps(e, ensure_ascii=False) + '\n')
        print(f'事件已寫入 {a.jsonl}')


if __name__ == '__main__':
    main()
