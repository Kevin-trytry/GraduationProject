#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

// 定義推論設定
#define NUM_CLASSES 11
#define NUM_ANCHORS 756
#define SCORE_THRESHOLD 0.25f // 信心度門檻
#define NMS_THRESHOLD 0.45f   // NMS 門檻 (IoU)
#define IMAGE_SIZE 192.0f     // 假設你的模型輸入大小是 192x192

// Bounding Box 結構體
struct Detection {
    float x; // 框的中心點 x
    float y; // 框的中心點 y
    float w; // 框的寬度
    float h; // 框的高度
    float score; // 最高信心度
    int class_id; // 預測的類別 ID
};

// 輔助函數：計算 Intersection over Union (IoU)
float calculate_iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x - a.w / 2.0f, b.x - b.w / 2.0f);
    float y1 = std::max(a.y - a.h / 2.0f, b.y - b.h / 2.0f);
    float x2 = std::min(a.x + a.w / 2.0f, b.x + b.w / 2.0f);
    float y2 = std::min(a.y + a.h / 2.0f, b.y + b.h / 2.0f);

    float intersection_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float area_a = a.w * a.h;
    float area_b = b.w * b.h;
    float union_area = area_a + area_b - intersection_area;

    if (union_area <= 0.0f) return 0.0f;
    return intersection_area / union_area;
}

// Non-Maximum Suppression (NMS)
std::vector<Detection> nms(std::vector<Detection>& detections, float nms_thresh) {
    std::vector<Detection> result;
    
    // 依照信心度由高到低排序
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            // 如果類別相同且 IoU 大於門檻，則抑制該候選框
            if (detections[i].class_id == detections[j].class_id) {
                if (calculate_iou(detections[i], detections[j]) > nms_thresh) {
                    suppressed[j] = true;
                }
            }
        }
    }
    return result;
}

/**
 * @brief 處理 YOLOv8 int8 輸出矩陣
 * 
 * @param output_tensor 指向模型輸出的 int8 陣列指標
 * @param scale 從 TFLite Tensor 讀取到的 quantization scale
 * @param zero_point 從 TFLite Tensor 讀取到的 quantization zero_point
 * @return std::vector<Detection> 最終過濾後的 Bounding Boxes
 */
std::vector<Detection> process_yolov8_output(int8_t* output_tensor, float scale, int32_t zero_point) {
    std::vector<Detection> raw_detections;

    // YOLOv8 輸出維度為 [1, 15, 756]
    // Memory layout: [batch=1, channel=15, anchor=756]
    // 因此 index 的計算方式是: c * 756 + a
    
    for (int a = 0; a < NUM_ANCHORS; a++) {
        // 1. 反量化 (Dequantize) 座標
        float cx = (output_tensor[0 * NUM_ANCHORS + a] - zero_point) * scale;
        float cy = (output_tensor[1 * NUM_ANCHORS + a] - zero_point) * scale;
        float w  = (output_tensor[2 * NUM_ANCHORS + a] - zero_point) * scale;
        float h  = (output_tensor[3 * NUM_ANCHORS + a] - zero_point) * scale;

        // 2. 尋找 11 個類別中的最高分
        float max_score = -1.0f;
        int max_class_id = -1;

        for (int c = 0; c < NUM_CLASSES; c++) {
            float score = (output_tensor[(4 + c) * NUM_ANCHORS + a] - zero_point) * scale;
            if (score > max_score) {
                max_score = score;
                max_class_id = c;
            }
        }

        // 3. 過濾掉低於信心度門檻的框
        if (max_score > SCORE_THRESHOLD) {
            Detection det;
            det.x = cx;
            det.y = cy;
            det.w = w;
            det.h = h;
            det.score = max_score;
            det.class_id = max_class_id;
            raw_detections.push_back(det);
        }
    }

    // 4. 執行 NMS 去除重疊框
    return nms(raw_detections, NMS_THRESHOLD);
}

// (模擬) main 函數示範如何在 TFLM 中呼叫
int main() {
    // 假設這是你從 TFLM (TensorFlow Lite for Microcontrollers) 拿到的 output tensor
    // TfLiteTensor* output = interpreter->output(0);
    // int8_t* output_data = output->data.int8;
    // float scale = output->params.scale;
    // int32_t zero_point = output->params.zero_point;
    
    // 這裡給予虛擬數值作示範
    int8_t dummy_output_data[15 * 756] = {0}; 
    float scale = 0.05f;
    int32_t zero_point = -128;

    std::cout << "開始解析 YOLOv8 [1, 15, 756] 張量..." << std::endl;
    std::vector<Detection> results = process_yolov8_output(dummy_output_data, scale, zero_point);
    
    std::cout << "最終偵測到 " << results.size() << " 個物件。" << std::endl;
    for (const auto& det : results) {
        std::cout << "Class: " << det.class_id 
                  << " | Score: " << det.score 
                  << " | Box: (" << det.x << ", " << det.y << ", " << det.w << ", " << det.h << ")" << std::endl;
    }

    return 0;
}
