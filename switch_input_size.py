# -*- coding: utf-8 -*-
"""
switch_input_size.py
====================
一個指令切換整條鏈路的模型輸入尺寸（96 <-> 192），避免手動改三個地方時漏掉。

會同時改動：
  1. cvapp_yolov8n_ob.cpp 的 YOLOV8_OB_INPUT_TENSOR_WIDTH / HEIGHT
  2. 把對應的 vela 模型複製到 model_zoo 並改成正確檔名
  3. build_and_flash.py 的 MODEL_FILE
  4. deploy.py 的 MODEL_IMGSZ 與 VELA_MODEL_SRC

用法：
  python switch_input_size.py 192      # 切到 192（Layout A 舊模型）
  python switch_input_size.py 96       # 切回 96（Layout B 新模型）
  python switch_input_size.py --status # 只顯示目前狀態，不改任何東西
"""
import argparse
import os
import re
import shutil
import sys

if sys.platform == 'win32':
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

ROOT = os.path.dirname(os.path.abspath(__file__))
CVAPP = os.path.join(ROOT, 'Seeed_Grove_Vision_AI_Module_V2', 'EPII_CM55M_APP_S',
                     'app', 'scenario_app', 'tflm_yolov8_od', 'cvapp_yolov8n_ob.cpp')
MODEL_ZOO = os.path.join(ROOT, 'Seeed_Grove_Vision_AI_Module_V2', 'model_zoo', 'tflm_yolov8_od')
WEIGHTS = os.path.join(ROOT, 'model_output', 'weights', 'best_saved_model')
BUILD_FLASH = os.path.join(ROOT, 'build_and_flash.py')
DEPLOY = os.path.join(ROOT, 'deploy.py')
FLASH_ADDR = '0xB7B000'

# 每個尺寸對應的來源模型
SOURCES = {
    96:  ('best_int8_vela.tflite',
          'Layout B / DFL raw：2026-09-08 訓練的 96px 模型'),
    192: ('best_full_int8_vela_RGB.tflite',
          'Layout A / 已解碼：192px 舊模型（實測可偵測，作為對照組）'),
}


def read(p):
    with open(p, 'rb') as f:
        return f.read().decode('utf-8')


def write(p, t):
    with open(p, 'wb') as f:
        f.write(t.encode('utf-8'))


def current_size():
    t = read(CVAPP)
    m = re.search(r'#if 1\r?\n#define YOLOV8_OB_INPUT_TENSOR_WIDTH\s+(\d+)', t)
    return int(m.group(1)) if m else None


def show_status():
    size = current_size()
    print(f'  韌體 input size : {size}')
    exp = {96: 189, 192: 756}.get(size)
    if exp:
        print(f'  對應 anchor 數  : {exp}  (開機 log 的 num_anchors= 應為此值)')
    m = re.search(r'gesture_yolov8n_(\d+)_vela', read(BUILD_FLASH))
    print(f'  燒錄的模型檔名  : gesture_yolov8n_{m.group(1) if m else "?"}_vela_{FLASH_ADDR}.tflite')
    m = re.search(r'MODEL_IMGSZ\s*=\s*(\d+)', read(DEPLOY))
    print(f'  deploy.py IMGSZ : {m.group(1) if m else "?"}')
    dst = os.path.join(MODEL_ZOO, f'gesture_yolov8n_{size}_vela_{FLASH_ADDR}.tflite')
    print(f'  model_zoo 檔案  : {"存在" if os.path.exists(dst) else "**不存在**"}  '
          f'({os.path.basename(dst)})')


def switch(size):
    src_name, desc = SOURCES[size]
    src = os.path.join(WEIGHTS, src_name)
    if not os.path.exists(src):
        print(f'[ERROR] 找不到來源模型：{src}')
        return 1

    # 1) 韌體 define
    t = read(CVAPP)
    new_t, n = re.subn(
        r'(#if 1\r?\n#define YOLOV8_OB_INPUT_TENSOR_WIDTH\s+)\d+(\r?\n#define YOLOV8_OB_INPUT_TENSOR_HEIGHT\s+)\d+',
        lambda m: f'{m.group(1)}{size}{m.group(2)}{size}', t)
    if n != 1:
        print(f'[ERROR] cvapp define 取代失敗（找到 {n} 處，預期 1 處）')
        return 1
    write(CVAPP, new_t)
    print(f'  [OK] cvapp_yolov8n_ob.cpp -> {size}x{size}')

    # 2) 複製模型
    dst_name = f'gesture_yolov8n_{size}_vela_{FLASH_ADDR}.tflite'
    dst = os.path.join(MODEL_ZOO, dst_name)
    shutil.copy2(src, dst)
    print(f'  [OK] {src_name}  ->  model_zoo/{dst_name}  '
          f'({os.path.getsize(dst):,} bytes)')

    # 3) build_and_flash.py
    t = read(BUILD_FLASH)
    new_t, n = re.subn(r'gesture_yolov8n_\d+_vela_0x[0-9A-Fa-f]+\.tflite', dst_name, t)
    write(BUILD_FLASH, new_t)
    print(f'  [OK] build_and_flash.py 模型路徑已更新 ({n} 處)')

    # 4) deploy.py
    t = read(DEPLOY)
    t, n1 = re.subn(r'MODEL_IMGSZ\s*=\s*\d+', f'MODEL_IMGSZ = {size}', t)
    t, n2 = re.subn(r'"best[^"]*\.tflite"', f'"{src_name}"', t)
    write(DEPLOY, t)
    print(f'  [OK] deploy.py 已更新 (IMGSZ {n1} 處, 來源 {n2} 處)')

    print()
    print(f'  模型說明：{desc}')
    print()
    print('  接下來（必須全編，define 改過了）：')
    print('    cd Seeed_Grove_Vision_AI_Module_V2\\EPII_CM55M_APP_S')
    print('    make clean && make')
    print('    cd ..\\..')
    print('    python build_and_flash.py --port COM3')
    print()
    print(f'  燒完確認開機 log 的 num_anchors = {189 if size == 96 else 756}')
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('size', nargs='?', type=int, choices=[96, 192])
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--model', default=None,
                    help='指定 model_output/weights/best_saved_model/ 底下的檔名，'
                         '覆蓋預設來源（Johnson 交新模型時用）')
    a = ap.parse_args()

    if a.model:
        if a.size is None:
            print('  [ERROR] --model 必須跟尺寸一起用，例如：'
                  'python switch_input_size.py 192 --model best_full_int8_vela.tflite')
            return 1
        SOURCES[a.size] = (a.model, f'手動指定：{a.model}')

    print('=' * 62)
    print('  目前狀態')
    print('=' * 62)
    show_status()
    print()

    if a.status or a.size is None:
        if a.size is None and not a.status:
            print('  用法： python switch_input_size.py 96 | 192')
        return 0

    if a.size == current_size():
        print(f'  [提示] 已經是 {a.size}，仍會重新同步模型與腳本。')
    print('=' * 62)
    print(f'  切換到 {a.size}x{a.size}')
    print('=' * 62)
    rc = switch(a.size)
    print('=' * 62)
    print('  切換後狀態')
    print('=' * 62)
    show_status()
    return rc


if __name__ == '__main__':
    sys.exit(main())
