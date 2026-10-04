#!/usr/bin/env python3
"""
尺度不變性檢測 —— 直接量「手在畫面裡佔多大」對辨識結果的影響。

這是目前整個專案最關鍵的一個指標。板子上手的大小會變，
如果模型只在某個特定大小才準，實機就會像 09-10/09-11 那樣忽好忽壞。

做法：
  對每張驗證圖，用標註框把手「重新取景」成佔畫面 20%~90% 的各種大小，
  再送進模型，看預測會不會跟著變。理想情況是每個尺寸都預測對同一類。

用法：
    python check_scale_invariance.py best.pt <images 資料夾> <labels 資料夾>
    python check_scale_invariance.py best.pt C:\\data\\val\\images C:\\data\\val\\labels --imgsz 192

輸出：
    每個「手佔畫面比例」下的準確率，以及逐類別的表現。
"""
import argparse
import glob
import os
import sys
from collections import defaultdict

try:
    import numpy as np
    import cv2
    from ultralytics import YOLO
except ImportError as e:
    sys.exit(f"缺少套件: {e}\n  pip install ultralytics opencv-python numpy")

NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B']
PAD_VALUE = 114          # 與 ultralytics letterbox 相同


def load_label(path):
    """回傳 (cls, cx, cy, w, h) 正規化座標；只取第一個框，沒有就回 None。"""
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                p = line.split()
                if len(p) >= 5:
                    return int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
    except Exception:
        pass
    return None


def reframe(img, box_norm, target_frac):
    """
    把影像重新取景，讓標註框的長邊佔畫面 target_frac。
    回傳一張正方形影像（不足處補 114 灰）。
    """
    H, W = img.shape[:2]
    _, cx, cy, bw, bh = box_norm
    cx_px, cy_px = cx * W, cy * H
    box_side = max(bw * W, bh * H)
    if box_side < 4:
        return None

    canvas = int(round(box_side / target_frac))
    if canvas < 16:
        return None

    x0 = int(round(cx_px - canvas / 2))
    y0 = int(round(cy_px - canvas / 2))

    out = np.full((canvas, canvas, 3), PAD_VALUE, dtype=img.dtype)

    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + canvas), min(H, y0 + canvas)
    if sx1 <= sx0 or sy1 <= sy0:
        return None

    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('images')
    ap.add_argument('labels')
    ap.add_argument('--imgsz', type=int, default=192)
    ap.add_argument('--conf', type=float, default=0.01,
                    help='刻意設很低，我們要看的是 top-1 類別而不是有沒有過門檻')
    ap.add_argument('--limit', type=int, default=200,
                    help='最多取幾張圖（預設 200，夠用且快）')
    ap.add_argument('--fracs', type=str, default='0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')
    args = ap.parse_args()

    fracs = [float(x) for x in args.fracs.split(',')]
    model = YOLO(args.model)

    imgs = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        imgs += glob.glob(os.path.join(args.images, '**', ext), recursive=True)
    imgs.sort()
    if not imgs:
        sys.exit(f'在 {args.images} 找不到影像')

    # 平均取樣，避免只取到同一類
    if len(imgs) > args.limit:
        step = len(imgs) / args.limit
        imgs = [imgs[int(i * step)] for i in range(args.limit)]

    hit = defaultdict(int)          # (frac) -> 正確數
    tot = defaultdict(int)
    hit_c = defaultdict(int)        # (frac, cls) -> 正確數
    tot_c = defaultdict(int)
    used = 0

    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(args.labels, stem + '.txt')
        if not os.path.exists(lp):
            continue
        lab = load_label(lp)
        if lab is None:
            continue                # 純背景圖，跳過
        gt = lab[0]

        img = cv2.imread(ip)
        if img is None:
            continue
        used += 1

        for fr in fracs:
            crop = reframe(img, lab, fr)
            if crop is None:
                continue
            r = model.predict(crop, imgsz=args.imgsz, conf=args.conf,
                              verbose=False)[0]
            tot[fr] += 1
            tot_c[(fr, gt)] += 1
            if len(r.boxes) > 0:
                k = int(r.boxes.conf.argmax())
                pred = int(r.boxes.cls[k])
                if pred == gt:
                    hit[fr] += 1
                    hit_c[(fr, gt)] += 1

    if used == 0:
        sys.exit('沒有任何圖片配對到標籤，請確認 images / labels 路徑')

    print(f'\n用了 {used} 張圖，每張重新取景成 {len(fracs)} 種大小\n')
    print('手佔畫面   準確率')
    print('-' * 46)
    accs = []
    for fr in fracs:
        if tot[fr] == 0:
            continue
        a = hit[fr] / tot[fr]
        accs.append(a)
        bar = '█' * int(a * 30)
        print(f'  {fr*100:3.0f}%    {a*100:5.1f}%  {bar}')
    print('-' * 46)

    if accs:
        lo, hi = min(accs), max(accs)
        print(f'\n最好 {hi*100:.1f}%   最差 {lo*100:.1f}%   落差 {(hi-lo)*100:.1f} 個百分點')
        if hi - lo > 0.30:
            print('  ❌ 落差過大 —— 模型高度依賴手的大小，實機會隨距離忽好忽壞。')
            print('     請確認 scale / translate augmentation 真的有生效。')
        elif hi - lo > 0.15:
            print('  ⚠️ 仍有明顯落差，可再加大 scale。')
        else:
            print('  ✅ 各種大小表現一致，尺度不變性良好。')

    print('\n逐類別（列=類別，欄=手佔畫面比例）')
    head = '      ' + ''.join(f'{int(f*100):>6}%' for f in fracs)
    print(head)
    for c in range(len(NAMES)):
        if not any(tot_c[(f, c)] for f in fracs):
            continue
        row = f'  {NAMES[c]:>2}  '
        for f in fracs:
            t = tot_c[(f, c)]
            row += f'{(hit_c[(f,c)]/t*100):>6.0f}' if t else '     -'
        print(row)

    print('\n判讀：某一列如果只有中間幾欄高、兩端掉下來，')
    print('      就是那個手勢只在特定距離才認得出來 —— 實機上會時好時壞。')


if __name__ == '__main__':
    main()
