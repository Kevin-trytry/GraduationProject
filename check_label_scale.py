#!/usr/bin/env python3
"""
量訓練集標註框的尺度分布 —— 判斷模型有沒有見過足夠的尺度變化。

用法：
    python check_label_scale.py <labels 資料夾>
    python check_label_scale.py C:\\YOLO_gesture\\dataset_split\\train\\labels

輸出兩件事：
  1. 每個類別的典型框寬 / 框高 / 面積      -> 看哪些類別「天生比較大」
  2. 類別內的 P90/P10 面積比               -> 看有沒有 scale augmentation
     1.x 倍  = 幾乎沒有尺度變化（模型會對距離極度敏感）
     3~10 倍 = 健康
"""
import sys, os, glob
from collections import defaultdict

try:
    import numpy as np
except ImportError:
    sys.exit("需要 numpy： pip install numpy")

NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'B']


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    root = sys.argv[1]
    files = glob.glob(os.path.join(root, '**', '*.txt'), recursive=True)
    if not files:
        sys.exit(f'在 {root} 底下找不到任何 .txt')

    boxes = defaultdict(list)          # cls -> [(w, h), ...]
    n_lines = 0
    for f in files:
        try:
            with open(f, encoding='utf-8') as fh:
                for line in fh:
                    p = line.split()
                    if len(p) >= 5:
                        boxes[int(p[0])].append((float(p[3]), float(p[4])))
                        n_lines += 1
        except Exception as e:
            print(f'  [略過] {f}: {e}')

    print(f'\n{len(files)} 個 label 檔，{n_lines} 個框\n')

    print('cls    n    框寬%   框高%   面積%   w/h    面積 P10   P50   P90   P90/P10')
    print('-' * 78)
    all_area = []
    for c in sorted(boxes):
        a = np.array(boxes[c])
        area = a[:, 0] * a[:, 1]
        all_area += list(area)
        p10, p50, p90 = np.percentile(area, [10, 50, 90])
        ratio = p90 / p10 if p10 > 0 else float('inf')
        flag = '  <<< 尺度變化過小' if ratio < 2.0 else ''
        name = NAMES[c] if c < len(NAMES) else str(c)
        print(f' {name:>2}  {len(a):4d}   {np.median(a[:,0])*100:5.1f}  '
              f'{np.median(a[:,1])*100:5.1f}  {np.median(area)*100:5.1f}  '
              f'{np.median(a[:,0]/np.maximum(a[:,1],1e-6)):5.2f}    '
              f'{p10*100:5.1f} {p50*100:5.1f} {p90*100:5.1f}   {ratio:5.2f}x{flag}')

    allA = np.array(all_area)
    print('-' * 78)
    print('全體面積分布： ' + '  '.join(
        f'P{q}={np.percentile(allA, q)*100:.0f}%' for q in (5, 25, 50, 75, 95)))

    ratios = []
    for c in boxes:
        a = np.array(boxes[c]); area = a[:, 0] * a[:, 1]
        if len(area) >= 5:
            p10, p90 = np.percentile(area, [10, 90])
            if p10 > 0:
                ratios.append(p90 / p10)
    if ratios:
        med = float(np.median(ratios))
        print(f'\n類別內尺度變化中位數 = {med:.2f}x')
        if med < 2.0:
            print('  ❌ 太小。模型只看過固定大小的手，換個距離就會失準。')
            print('     建議訓練時加： scale=0.7  translate=0.2  mosaic=1.0')
        elif med < 3.0:
            print('  ⚠️ 偏小，建議加大 scale augmentation。')
        else:
            print('  ✅ 尺度變化足夠。')

    print('\n提醒：上面的「框寬%」是相對於整張訓練圖。')
    print('部署時手在畫面中的相對大小若明顯小於這個數字，模型等於沒看過。')


if __name__ == '__main__':
    main()
