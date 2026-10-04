#!/usr/bin/env python3
"""
build_and_flash.py
==================
HimaxWiseEye2 (Grove Vision AI V2) 自動化編譯 + 燒錄腳本

流程：
  Step 3: make clean + make（編譯韌體）
  Step 4: 複製 ELF → image gen（生成 output.img）
  Step 5: xmodem 燒錄（需手動按 BOOT + RST）

使用方式：
  python build_and_flash.py               # 完整流程（clean + make + flash）
  python build_and_flash.py --no-clean    # 跳過 make clean（增量編譯）
  python build_and_flash.py --build-only  # 只編譯，不燒錄
  python build_and_flash.py --flash-only  # 跳過編譯，直接燒錄
  python build_and_flash.py --port COM5   # 指定 COM port（預設 COM3）
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

# ─── 路徑設定（若專案移動，只需改這裡）────────────────────────────────────────
PROJECT_ROOT   = r"C:\GraduationProject"
SDK_ROOT       = os.path.join(PROJECT_ROOT, "Seeed_Grove_Vision_AI_Module_V2")

# Step 3: 編譯
BUILD_DIR      = os.path.join(SDK_ROOT, "EPII_CM55M_APP_S")
ELF_PATH       = os.path.join(
    BUILD_DIR,
    r"obj_epii_evb_icv30_bdv10\gnu_epii_evb_WLCSP65\EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf"
)

# Step 4: Image gen
IMAGE_GEN_DIR  = os.path.join(SDK_ROOT, "we2_image_gen_local")
ELF_DEST_DIR   = os.path.join(IMAGE_GEN_DIR, "input_case1_secboot")
IMAGE_GEN_EXE  = os.path.join(IMAGE_GEN_DIR, "we2_local_image_gen.exe")
IMAGE_GEN_JSON = "project_case1_blp_wlcsp.json"
OUTPUT_IMG     = os.path.join(
    IMAGE_GEN_DIR, r"output_case1_sec_wlcsp\output.img"
)

# Step 5: 燒錄
XMODEM_SCRIPT  = os.path.join(SDK_ROOT, "xmodem", "xmodem_send.py")
MODEL_FILE     = os.path.join(
    SDK_ROOT,
    r"model_zoo\tflm_yolov8_od\gesture_yolov8n_192_vela_0xB7B000.tflite"
)
MODEL_ADDR     = "0xB7B000"
MODEL_OFFSET   = "0x00000"
DEFAULT_PORT   = "COM3"
BAUDRATE       = "921600"

# ─── 顏色輸出（Windows 終端機支援 ANSI）─────────────────────────────────────
def c(text, color):
    codes = {"red": 31, "green": 32, "yellow": 33, "cyan": 36, "bold": 1}
    return f"\033[{codes.get(color, 0)}m{text}\033[0m"

def header(title):
    print()
    print(c("═" * 60, "cyan"))
    print(c(f"  {title}", "cyan"))
    print(c("═" * 60, "cyan"))

def ok(msg):    print(c(f"  ✅ {msg}", "green"))
def warn(msg):  print(c(f"  ⚠️  {msg}", "yellow"))
def err(msg):   print(c(f"  ❌ {msg}", "red"))
def info(msg):  print(f"  {msg}")

def run(cmd, cwd=None, check=True):
    """執行指令，即時輸出，回傳 returncode"""
    info(f"執行: {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
    result = subprocess.run(
        cmd, cwd=cwd, shell=isinstance(cmd, str),
        stdout=None, stderr=None   # 直接繼承父程序 stdout/stderr（即時顯示）
    )
    if check and result.returncode != 0:
        err(f"指令失敗，退出碼 {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode

# ─── Step 3: 編譯 ─────────────────────────────────────────────────────────────
def step_build(clean: bool):
    header("Step 3：編譯韌體（make）")

    if not os.path.isdir(BUILD_DIR):
        err(f"BUILD_DIR 不存在: {BUILD_DIR}")
        sys.exit(1)

    if clean:
        info("執行 make clean...")
        run("make clean", cwd=BUILD_DIR)
        ok("make clean 完成")

    info("執行 make（這會花幾分鐘）...")
    start = time.time()
    rc = run("make", cwd=BUILD_DIR, check=False)
    elapsed = time.time() - start

    if rc != 0:
        err(f"make 失敗（退出碼 {rc}）")
        sys.exit(rc)

    if not os.path.isfile(ELF_PATH):
        err(f"ELF 檔案未生成: {ELF_PATH}")
        sys.exit(1)

    ok(f"編譯成功！耗時 {elapsed:.1f} 秒")
    ok(f"ELF: {ELF_PATH}")

# ─── Step 4: 生成映像 ─────────────────────────────────────────────────────────
def step_image_gen():
    header("Step 4：生成韌體映像（image gen）")

    # 複製 ELF
    os.makedirs(ELF_DEST_DIR, exist_ok=True)
    dest = os.path.join(ELF_DEST_DIR, os.path.basename(ELF_PATH))
    info(f"複製 ELF → {dest}")
    shutil.copy2(ELF_PATH, dest)
    ok("ELF 複製完成")

    # 執行 image gen
    if not os.path.isfile(IMAGE_GEN_EXE):
        err(f"image gen 執行檔不存在: {IMAGE_GEN_EXE}")
        sys.exit(1)

    run([IMAGE_GEN_EXE, IMAGE_GEN_JSON], cwd=IMAGE_GEN_DIR)

    if not os.path.isfile(OUTPUT_IMG):
        err(f"output.img 未生成: {OUTPUT_IMG}")
        sys.exit(1)

    ok(f"映像生成成功: {OUTPUT_IMG}")

# ─── Step 5: 燒錄 ─────────────────────────────────────────────────────────────
def step_flash(port: str):
    header("Step 5：燒錄韌體 + 模型（xmodem）")

    if not os.path.isfile(OUTPUT_IMG):
        err(f"output.img 不存在，請先執行 image gen: {OUTPUT_IMG}")
        sys.exit(1)

    if not os.path.isfile(MODEL_FILE):
        err(f"模型檔案不存在: {MODEL_FILE}")
        sys.exit(1)

    # ── 重要提示：需要人工按按鈕 ──────────────────────────────────────────────
    print()
    print(c("  ╔══════════════════════════════════════════════════╗", "yellow"))
    print(c("  ║   ⚠️  請準備好按開發板按鈕！                      ║", "yellow"))
    print(c("  ║                                                  ║", "yellow"))
    print(c("  ║  xmodem 腳本啟動後，終端機會顯示：               ║", "yellow"))
    print(c('  ║  "Please press reset button!!"                   ║', "yellow"))
    print(c("  ║                                                  ║", "yellow"))
    print(c("  ║  看到後，依序執行：                               ║", "yellow"))
    print(c("  ║    1. 按住 BOOT 按鈕不放                          ║", "yellow"))
    print(c("  ║    2. 按一下 RST 按鈕（放開）                     ║", "yellow"))
    print(c("  ║    3. 放開 BOOT 按鈕                              ║", "yellow"))
    print(c("  ║                                                  ║", "yellow"))
    print(c("  ║  燒錄完成後會自動重啟開發板。                     ║", "yellow"))
    print(c("  ╚══════════════════════════════════════════════════╝", "yellow"))
    print()

    input(c("  按下 Enter 繼續啟動燒錄腳本...", "bold"))
    print()

    model_arg = f"{MODEL_FILE} {MODEL_ADDR} {MODEL_OFFSET}"

    cmd = [
        sys.executable, XMODEM_SCRIPT,
        "--port",      port,
        "--baudrate",  BAUDRATE,
        "--protocol",  "xmodem",
        "--file",      OUTPUT_IMG,
        "--model",     model_arg,
    ]

    info(f"COM Port: {port}")
    info(f"韌體映像: {OUTPUT_IMG}")
    info(f"模型: {model_arg}")
    print()

    rc = run(cmd, check=False)

    if rc == 0:
        print()
        ok("燒錄完成！按開發板 RST 按鈕重新啟動（不需按 BOOT）。")
    else:
        err(f"燒錄失敗（退出碼 {rc}）")
        warn("常見原因：")
        warn("  1. COM port 被其他程式佔用（關閉 TeraTerm 等）")
        warn("  2. 沒有在正確時機按 BOOT + RST")
        warn("  3. USB 線只有充電功能（換數據線）")
        sys.exit(rc)

# ─── 主程式 ───────────────────────────────────────────────────────────────────
def main():
    # 啟用 Windows ANSI 顏色支援
    os.system("")

    parser = argparse.ArgumentParser(
        description="HimaxWiseEye2 自動化編譯 + 燒錄腳本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  python build_and_flash.py               # 完整流程（clean + make + flash）
  python build_and_flash.py --no-clean    # 增量編譯（跳過 clean）
  python build_and_flash.py --build-only  # 只編譯，不燒錄
  python build_and_flash.py --flash-only  # 直接燒錄（跳過編譯）
  python build_and_flash.py --port COM5   # 指定 COM port
        """
    )
    parser.add_argument("--no-clean",    action="store_true", help="跳過 make clean（增量編譯）")
    parser.add_argument("--build-only",  action="store_true", help="只編譯，不燒錄")
    parser.add_argument("--flash-only",  action="store_true", help="跳過編譯，直接燒錄")
    parser.add_argument("--port",        default=DEFAULT_PORT, help=f"COM port（預設 {DEFAULT_PORT}）")
    args = parser.parse_args()

    print(c("\n  HimaxWiseEye2 自動化部署腳本", "bold"))
    print(c(f"  Port: {args.port}", "cyan"))
    if args.flash_only:
        print(c("  模式: 僅燒錄（跳過編譯）", "cyan"))
    elif args.build_only:
        print(c("  模式: 僅編譯（不燒錄）", "cyan"))
    elif args.no_clean:
        print(c("  模式: 增量編譯 + 燒錄", "cyan"))
    else:
        print(c("  模式: 完整流程（clean + make + flash）", "cyan"))

    total_start = time.time()

    if not args.flash_only:
        step_build(clean=not args.no_clean)
        step_image_gen()

    if not args.build_only:
        step_flash(port=args.port)

    total_elapsed = time.time() - total_start
    print()
    print(c("═" * 60, "green"))
    ok(f"全部完成！總耗時 {total_elapsed:.1f} 秒")
    print(c("═" * 60, "green"))
    print()

if __name__ == "__main__":
    main()
