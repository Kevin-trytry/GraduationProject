"""
比較新舊模型在 capture 圖片上的偵測結果
"""
import numpy as np, os, glob, cv2

try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter

MODELS = {
    "best_full_int8 (最新,非Vela)": r"C:\GraduationProject\model_output\weights\best_saved_model\best_full_int8.tflite",
    "best_full_int8_vela (舊Vela)": r"C:\GraduationProject\model_output\weights\best_saved_model\best_full_int8_vela.tflite",
}

captures_dir = r"C:\GraduationProject\captures"
jpg_files = sorted(glob.glob(os.path.join(captures_dir, "**", "*.jpg"), recursive=True))[-5:]

if not jpg_files:
    # 用純色測試圖代替
    jpg_files = []
    print("[警告] 找不到 capture 圖片，用純色圖測試")

gesture_names = ['0','1','2','3','4','5','6','7','8','9','B']

for model_label, model_path in MODELS.items():
    if not os.path.exists(model_path):
        print(f"[跳過] 找不到: {model_path}")
        continue
    
    print(f"\n{'='*60}")
    print(f"  {model_label}")
    print(f"  {os.path.basename(model_path)}  ({os.path.getsize(model_path):,} bytes)")
    print(f"{'='*60}")
    
    try:
        interp = Interpreter(model_path=model_path)
        interp.allocate_tensors()
        inp_d = interp.get_input_details()[0]
        out_d = interp.get_output_details()[0]
        zp = out_d['quantization_parameters']['zero_points'][0]
        sc = out_d['quantization_parameters']['scales'][0]
        
        print(f"  output shape={out_d['shape']}  zp={zp}  scale={sc:.4f}")
        
        passed = 0
        for f in jpg_files:
            img = cv2.imread(f)
            if img is None: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_r = cv2.resize(img_rgb, (192, 192))
            img_i8 = (img_r.astype(np.int32) - 128).astype(np.int8)[np.newaxis]
            interp.set_tensor(inp_d['index'], img_i8)
            interp.invoke()
            raw = interp.get_tensor(out_d['index'])
            deq = (raw[0].astype(np.float32) - zp) * sc
            cls = deq[4:, :]
            max_s = cls.max()
            if max_s >= 0.25:
                bc = cls.argmax() // cls.shape[1]
                ba = cls.argmax() % cls.shape[1]
                print(f"  DETECT! {os.path.basename(f)}: score={max_s:.3f}  class={gesture_names[bc]}({bc})  anchor={ba}")
                passed += 1
        
        if not jpg_files:
            # 純亮圖測試
            for br in [50, 120, 180]:
                img_i8 = np.ones((1, 192, 192, 3), dtype=np.int8) * (br - 128)
                interp.set_tensor(inp_d['index'], img_i8)
                interp.invoke()
                raw = interp.get_tensor(out_d['index'])
                deq = (raw[0].astype(np.float32) - zp) * sc
                cls = deq[4:, :]
                print(f"  亮度{br}: max_score={cls.max():.4f}  {'PASS' if cls.max()>=0.25 else 'fail'}")
        else:
            print(f"  → {passed}/{len(jpg_files)} 張圖偵測到手勢 (threshold=0.25)")

    except Exception as e:
        import traceback
        print(f"  [ERROR] {e}")

print("\n完成。")
