# -*- coding: utf-8 -*-
"""
deploy.py — HimaxWiseEye2 手勢辨識模型部署輔助腳本

功能：
  1. 將 Vela 編譯後的模型複製到 model_zoo 目錄（正確命名）
  2. 顯示完整的 make 編譯指令
  3. 顯示完整的 xmodem 燒錄指令
  4. 顯示 PC 端預覽指令

用法：
  python deploy.py
  python deploy.py --port COM5
  python deploy.py --skip-copy         # 跳過模型複製（已複製過）
"""

import os
import sys

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import shutil
import argparse

# ─── 路徑設定 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SDK_ROOT = os.path.join(PROJECT_ROOT, "Seeed_Grove_Vision_AI_Module_V2")
EPII_ROOT = os.path.join(SDK_ROOT, "EPII_CM55M_APP_S")
MODEL_ZOO_DIR = os.path.join(SDK_ROOT, "model_zoo", "tflm_yolov8_od")
WE2_IMAGE_GEN = os.path.join(SDK_ROOT, "we2_image_gen_local")
XMODEM_DIR = os.path.join(SDK_ROOT, "xmodem")

# 模型來源路徑（Vela 編譯輸出）
VELA_MODEL_SRC = os.path.join(
    PROJECT_ROOT, "model_output", "weights", "best_saved_model",
    "best_int8_vela.tflite"
)

# Flash 地址（對應 common_config.h 中的 YOLOV8_OBJECT_DETECTION_FLASH_ADDR）
FLASH_ADDR = "0xB7B000"
FLASH_OFFSET = "0x00000"

# 模型輸入尺寸 —— 必須與 cvapp_yolov8n_ob.cpp 的 YOLOV8_OB_INPUT_TENSOR_WIDTH
# 以及訓練/匯出端的 IMAGE_SIZE 一致（2026-09-08 起為 96）
MODEL_IMGSZ = 192

# 目標模型檔名（含輸入尺寸與 flash 地址，遵循 Himax 命名慣例）
MODEL_FILENAME = f"gesture_yolov8n_{MODEL_IMGSZ}_vela_{FLASH_ADDR}.tflite"
MODEL_DST = os.path.join(MODEL_ZOO_DIR, MODEL_FILENAME)

# 韌體 output
ELF_FILE = os.path.join(
    EPII_ROOT, "obj_epii_evb_icv30_bdv10", "gnu_epii_evb_WLCSP65",
    "EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf"
)
OUTPUT_IMG = os.path.join(
    WE2_IMAGE_GEN, "output_case1_sec_wlcsp", "output.img"
)


def print_banner(title):
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def step_1_copy_model(skip=False):
    """將 Vela 模型複製到 model_zoo"""
    print_banner("Step 1: 複製模型到 model_zoo")

    if skip:
        print(f"  [跳過] --skip-copy 已啟用")
        if os.path.exists(MODEL_DST):
            size_kb = os.path.getsize(MODEL_DST) / 1024
            print(f"  [OK] 模型已存在: {MODEL_DST} ({size_kb:.0f} KB)")
        else:
            print(f"  [WARN] 模型不存在: {MODEL_DST}")
            print(f"         請手動複製或移除 --skip-copy")
        return

    if not os.path.exists(VELA_MODEL_SRC):
        print(f"  [ERROR] 找不到 Vela 模型: {VELA_MODEL_SRC}")
        print(f"  請先執行 model_export_tools/export_full_int8.py 進行模型轉換")
        sys.exit(1)

    src_size = os.path.getsize(VELA_MODEL_SRC) / 1024
    print(f"  來源: {VELA_MODEL_SRC} ({src_size:.0f} KB)")
    print(f"  目標: {MODEL_DST}")

    os.makedirs(MODEL_ZOO_DIR, exist_ok=True)
    shutil.copy2(VELA_MODEL_SRC, MODEL_DST)
    print(f"  [OK] 模型複製成功！")


def step_2_show_build_commands():
    """顯示編譯韌體指令"""
    print_banner("Step 2: 編譯韌體")

    print(f"  工作目錄: {EPII_ROOT}")
    print(f"  確認 makefile 中 APP_TYPE = tflm_yolov8_od")
    print()
    print(f"  請在終端機執行以下指令：")
    print(f"  ┌──────────────────────────────────────────")
    print(f"  │ cd {EPII_ROOT}")
    print(f"  │ make clean")
    print(f"  │ make")
    print(f"  └──────────────────────────────────────────")
    print()
    print(f"  預期輸出 ELF: {ELF_FILE}")

    if os.path.exists(ELF_FILE):
        print(f"  [INFO] ELF 檔案已存在（可能是舊的，建議重新編譯）")


def step_3_show_image_gen_commands():
    """顯示 image gen 指令"""
    print_banner("Step 3: 生成韌體映像")

    input_dir = os.path.join(WE2_IMAGE_GEN, "input_case1_secboot")
    print(f"  請在終端機執行以下指令：")
    print(f"  ┌──────────────────────────────────────────")
    print(f"  │ cd {WE2_IMAGE_GEN}")
    print(f"  │ copy {ELF_FILE} {input_dir}\\")
    print(f"  │ we2_local_image_gen.exe project_case1_blp_wlcsp.json")
    print(f"  └──────────────────────────────────────────")
    print()
    print(f"  預期輸出: {OUTPUT_IMG}")


def step_4_show_flash_commands(port):
    """顯示燒錄指令"""
    print_banner("Step 4: 燒錄韌體 + 模型")

    xmodem_script = os.path.join(XMODEM_DIR, "xmodem_send.py")
    model_arg = f'"{MODEL_DST} {FLASH_ADDR} {FLASH_OFFSET}"'

    print(f"  [!!] 燒錄前請確認：")
    print(f"     1. 開發板已連接 USB")
    print(f"     2. 沒有其他程式佔用 {port}（關閉 TeraTerm 等）")
    print()
    print(f"  請在終端機執行以下指令：")
    print(f"  ┌──────────────────────────────────────────")
    print(f"  │ python {xmodem_script} \\")
    print(f"  │   --port={port} \\")
    print(f"  │   --baudrate=921600 \\")
    print(f"  │   --protocol=xmodem \\")
    print(f"  │   --file={OUTPUT_IMG} \\")
    print(f"  │   --model={model_arg}")
    print(f"  └──────────────────────────────────────────")
    print()
    print(f"  燒錄完成後，按開發板上的 RESET 按鈕重啟。")


def step_5_show_preview_commands(port):
    """顯示 PC 端預覽指令"""
    print_banner("Step 5: PC 端即時預覽")

    preview_script = os.path.join(PROJECT_ROOT, "grove_vision_preview_json.py")
    print(f"  請在終端機執行以下指令：")
    print(f"  ┌──────────────────────────────────────────")
    print(f"  │ python {preview_script} \\")
    print(f"  │   --port {port} --debug")
    print(f"  └──────────────────────────────────────────")
    print()
    print(f"  操作說明：")
    print(f"    • 按 Q 或 ESC 退出")
    print(f"    • --debug 可印出原始 JSON 資料")
    print(f"    • --threshold 50 可只顯示信心度 ≥ 50% 的框")


def main():
    parser = argparse.ArgumentParser(description='HimaxWiseEye2 手勢辨識模型部署')
    parser.add_argument('--port', type=str, default='COM3',
                        help='開發板串列埠 (預設: COM3)')
    parser.add_argument('--skip-copy', action='store_true',
                        help='跳過模型複製步驟')
    args = parser.parse_args()

    print_banner("HimaxWiseEye2 手勢辨識模型部署工具")
    print(f"  專案根目錄: {PROJECT_ROOT}")
    print(f"  SDK 根目錄: {SDK_ROOT}")
    print(f"  COM 埠:     {args.port}")
    print(f"  Flash 地址: {FLASH_ADDR}")

    step_1_copy_model(skip=args.skip_copy)
    step_2_show_build_commands()
    step_3_show_image_gen_commands()
    step_4_show_flash_commands(args.port)
    step_5_show_preview_commands(args.port)

    print_banner("部署流程完成")
    print(f"  請依序執行 Step 2 ~ Step 5 的指令。")
    print(f"  如有問題，請用 --debug 模式執行預覽程式查看原始資料。\n")


if __name__ == '__main__':
    main()
