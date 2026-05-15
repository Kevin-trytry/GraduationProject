import os

# 填入你 Vela 編譯出來的 tflite 路徑
tflite_path = "model_output/weights/best_saved_model/best_full_int8_vela.tflite"
out_path = "model.cc" # 輸出的 C++ 檔案

with open(tflite_path, "rb") as f:
    data = f.read()

with open(out_path, "w") as f:
    f.write("#include <stdint.h>\n\n")
    f.write("alignas(16) const unsigned char g_model[] = {\n")
    for i, byte in enumerate(data):
        f.write(f"0x{byte:02x}, ")
        if (i + 1) % 12 == 0:
            f.write("\n")
    f.write("\n};\n")
    f.write(f"const unsigned int g_model_len = {len(data)};\n")

print(f"轉換成功！已生成 {out_path}")
