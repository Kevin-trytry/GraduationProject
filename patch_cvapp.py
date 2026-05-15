"""
Patch cvapp_yolov8n_ob.cpp for gesture recognition:
1. Replace COCO class names with our 11 gesture classes
2. Enable debug logging to print class names
"""
import re

filepath = (
    r"Seeed_Grove_Vision_AI_Module_V2\EPII_CM55M_APP_S\app\scenario_app"
    r"\tflm_yolov8_od\cvapp_yolov8n_ob.cpp"
)

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Replace COCO class array with our gesture classes
old_coco = (
    'std::string coco_classes[] = {"person","bicycle","car","motorcycle","airplane",'
    '"bus","train","truck","boat","traffic light","fire hydrant","stop sign",'
    '"parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant",'
    '"bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",'
    '"frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove",'
    '"skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork",'
    '"knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot",'
    '"hot dog","pizza","donut","cake","chair","couch","potted plant","bed",'
    '"dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",'
    '"microwave","oven","toaster","sink","refrigerator","book","clock","vase",'
    '"scissors","teddy bear","hair drier","toothbrush"};'
)

new_gesture = (
    '// Hand gesture class names (0-10: digit 0~9 + B)\n'
    'std::string coco_classes[] = {"0","1","2","3","4","5","6","7","8","9","B"};'
)

if old_coco in content:
    content = content.replace(old_coco, new_gesture)
    print("✅ Replaced COCO classes with gesture classes")
else:
    print("⚠️  Could not find COCO classes string - might already be replaced")

# 2. Enable debug logging (set YOLOV8N_OB_DBG_APP_LOG to 1)
content = content.replace(
    "#define YOLOV8N_OB_DBG_APP_LOG 0",
    "#define YOLOV8N_OB_DBG_APP_LOG 1"
)
print("✅ Enabled debug logging")

# 3. Remove the coco_ids array (not needed for gesture classes)
old_ids = re.compile(
    r'int coco_ids\[\] = \{.*?\};',
    re.DOTALL
)
content = old_ids.sub(
    '// coco_ids removed - using sequential class indices for gesture recognition',
    content
)
print("✅ Removed coco_ids array")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print(f"\n✅ Patched: {filepath}")
