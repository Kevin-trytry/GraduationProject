> ## ⚠️ 這份文件的結論已經過時
>
> 2026-09-29 的 `docs/根因報告_權重訓練尺寸不符_20260929.md` 查出：
> 從 2026-06 到 2026-09，匯出腳本一直讀到一顆 **imgsz=640 訓練的舊權重**，
> 而韌體以 192 運作。**這段期間所有的實機數據與推論都建立在錯誤的模型上。**
>
> 保留本文是為了記錄除錯過程，**不要拿它的結論當依據**。
> 現行有效的文件請看 `docs/` 根目錄。

> **特別註記**：本文主張「切開匯出的位置決定量化後精度差 14 倍」。實機對照為 Layout A 43% vs B 45%，**沒有差異**，該主張不成立。
>
---

# Layout A / Layout B — 匯出端要做的事

> 對象：Johnson（訓練 / 匯出端）　|　撰寫：Kevin（韌體端）　|　2026-09-09
> 一句話：**同一顆 `best.pt`，在 Detect head 的哪個位置切開匯出，決定量化後的精度差 14 倍。**

---

## 一、兩者的差別

YOLOv8 的偵測頭內部：

```
backbone → Detect head
              │
              ├─ box 分支：64 通道（4 邊 × reg_max 16 bins 的 DFL 分布）
              └─ cls 分支：11 通道（各類別 raw logit）
                       │
        ┌──────────────┴──────────────┐   ← ★ Layout B 在這裡切（現況）
        │                             │
   DFL softmax + 積分            sigmoid()
   → ltrb × stride
   → cx,cy,w,h 像素座標
        │                             │
        └──────────────┬──────────────┘   ← ★ Layout A 在這裡切（目標）
                       ↓
              預設會 cat 成單一輸出
```

| | **Layout A**（目標） | **Layout B**（現況） |
|---|---|---|
| bbox tensor | `[1, 4, 756]`　cx,cy,w,h **像素值** | `[1, 64, 756]`　DFL **raw logits** |
| cls tensor | `[1, 756, 11]` 或 `[1, 11, 756]`　**已 sigmoid**（0~1） | `[1, 11, 756]`　**raw logits** |
| 量化 scale | **各自獨立** | **兩者共用同一組** |

---

## 二、為什麼 Layout B 的兩個輸出會共用 scale

Layout B 是從**同一個張量**用 `split` 拆出來的。`Split` 在 TFLite 裡是純資料搬移，不做重新量化 —— 拆出來的兩半**繼承同一組 scale / zero_point**。

Layout A 的兩個輸出各自走過不同運算（一邊 DFL 解碼成像素、一邊 sigmoid 成 0~1），量化器會依各自的實際分布給獨立的 scale。

**問題核心：Layout B 用一個 scale 去服務兩種量級完全不同的張量。**

---

## 三、實測差距（我們自己的模型）

| | Layout A（`best_full_int8_vela_RGB.tflite`） | Layout B（`best_int8_vela.tflite` 09-09） |
|---|---|---|
| bbox scale / zp | `1.281372 / −120` → 值域 [−10.3, 316.5] | `0.215574 / 39` → 值域 [−36.0, 19.0] |
| cls scale / zp | `0.00390625 / −128` → 值域 [0, 0.996] | 同上（共用） |
| **信心度解析度** | **0.39 個百分點／階** | **約 5.4 個百分點／階** |

**差約 14 倍。** Layout B 下模型想表達「73%」，實際只能落在 68% 或 73.5% 這類格子上。

另外 Layout B 對 calibration 品質敏感得多 —— 一個 scale 要同時貼合兩種分布，校正資料一偏就兩邊一起遭殃。（這與你已經修好的 `representative_dataset` 偏斜是**兩個獨立的問題**，都要處理。）

---

## 四、★ 要改的地方：`export_full_int8.py`

在呼叫 `model.export(...)` **之前**，把 `Detect._inference` 換掉。

### 建議版本（不做 permute）

```python
import torch
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import make_anchors


def split_output_inference(self, x):
    """回傳 (已解碼的 bbox, 已 sigmoid 的 cls) 兩個輸出，而非 cat 成一個。"""
    shape = x[0].shape                       # BCHW
    x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

    if self.dynamic or self.shape != shape:
        self.anchors, self.strides = (
            t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5)
        )
        self.shape = shape

    box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

    #  dbox            -> [b, 4,  anchors]  已是像素座標
    #  cls.sigmoid()   -> [b, nc, anchors]  已是 0~1
    return dbox, cls.sigmoid()


Detect._inference = split_output_inference          # ← 一定要在 export 之前
```

> **為什麼不做 permute**：`yolov8_quantization_analysis.md` §A 的版本有 `.permute(0, 2, 1)` 把 cls 轉成 `[b, anchors, nc]`。**不需要。**
>
> 韌體會自動判斷 class tensor 是 anchor-major 還是 class-major，兩種都吃。而 permute 在轉檔後很可能變成一個 CPU 端的 `Transpose` op，把 NPU 的計算圖切成兩段（原廠 COCO 模型就是這樣，所以檔名才叫 `delete_transpose`）。少一個 transpose 比較乾淨、也比較快。

### 版本相容性提醒

`decode_bboxes()` 的簽章在不同 ultralytics 版本略有差異（有些版本多一個 `xywh=True` 參數）。如果報 `TypeError`，先印出來確認：

```python
import inspect
print(inspect.signature(Detect.decode_bboxes))
```

---

## 五、怎麼確認改對了

### 5.1 匯出後檢查 tflite 的輸出張量

```python
import tensorflow as tf
i = tf.lite.Interpreter(model_path='best_full_int8.tflite')   # Vela 編譯「前」那顆
i.allocate_tensors()
for d in i.get_output_details():
    q = d['quantization_parameters']
    print(d['name'], d['shape'], 'scale=', q['scales'], 'zp=', q['zero_points'])
```

**正確的結果應該長這樣：**

```
... [1, 4, 756]   scale=[1.28...]      zp=[-120]     ← bbox，像素值，值域約 [-10, 316]
... [1, 11, 756]  scale=[0.00390625]   zp=[-128]     ← cls，已 sigmoid，值域 [0, 1]
```

判斷重點：

| 檢查項 | 正確 | 錯誤（還是 Layout B） |
|---|---|---|
| 輸出張量數 | 2 個 | 2 個（一樣，看形狀） |
| bbox 通道數 | **4** | 64 |
| **兩者的 scale** | **不同** | **完全相同** |
| cls 值域 | [0, 1] 附近 | 幾十的量級 |

> 若只有 **1 個** 輸出、形狀是 `[1, 15, 756]`，代表 patch 沒生效（那是 ultralytics 預設的 `cat` 行為）。

### 5.2 檢查 Vela 報告的 operator 配置

```
Info: The following operators are placed on the CPU: ...
Total NPU operators: xx
Total CPU operators: xx
```

**理想是 CPU operators = 0**（整張圖一顆 `ethos-u`）。若出現 `Transpose` 留在 CPU，就是 permute 造成的 —— 用第四節的無 permute 版本可避免。

### 5.3 燒錄後看韌體開機 log

我會確認這幾行：

```
mode      : A / decoded (4ch, export 已解碼)
bbox  dims: [1, 4, 756]   scale(x1000)=1281 zp=-120
class dims: [1, 11, 756]  scale(x1000)=3    zp=-128
num_classes=11  num_anchors=756  cls_layout=class-major  sigmoid=NO (模型已算)
bbox 座標    : pixel (直接使用)
```

`mode : A` 就代表成功了。

---

## 六、韌體端你不用管

**兩種 Layout 韌體都支援，執行期自動偵測，換模型不需要改任何 code、也不需要重新編譯韌體。**

自動偵測的規則（寫出來只是讓你知道什麼情況會誤判）：

```
bbox tensor  = 兩個輸出中 dims[1] 為 4 或 64 的那個
dfl_mode     = (dims[1] == 64)
class-major  = (cls->dims[2] == num_anchors)
需要 sigmoid = ((127 - zp) * scale > 1.5)      ← 已 sigmoid 的值域必在 0~1
bbox 是否歸一化 = ((127 - zp) * scale <= 2.0)   ← 我們的是像素值，不會誤判
```

只有在 `num_classes` 剛好等於 4 或 64 時才會誤判。目前 11 類，安全。

**輸出順序（哪個在 output(0)）也不用管** —— 韌體看形狀判斷，不看順序。

---

## 七、驗收標準

| # | 項目 | 標準 |
|---|---|---|
| 1 | tflite 輸出 | bbox `[1, 4, 756]` + cls `[1, 11, 756]`，**兩者 scale 不同** |
| 2 | cls 值域 | 落在 [0, 1] |
| 3 | Vela CPU operators | 0（或至少不含 Transpose） |
| 4 | 韌體開機 log | `mode : A / decoded`、`sigmoid=NO` |
| 5 | 實機信心度 | 應優於目前 Layout B 的水準（平均 46%、中位 42%） |

---

## 八、與其他待辦的關係

| 項目 | 狀態 | 說明 |
|---|---|---|
| calibration 只取 cls0 100 張 | 你已修（改成每類 20 張） | **獨立問題**，兩個都要做 |
| 改產 Layout A | 本文件 | 讓修好的 calibration 發揮最大效果 |
| validation 改 group split | 待辦 | 不影響產出的模型，只影響評估數字是否可信 |

三者互相獨立、可以分開進行。優先順序建議：**calibration（已完成）→ Layout A → group split**。

---

## 附：相關檔案

| 路徑 | 內容 |
|---|---|
| `docs/pipeline_3_firmware.md` §2.2 | 韌體端的兩種 Layout 規格與介面契約 |
| `yolov8_quantization_analysis.md` §A | 最早提出 split-output 的分析（其 permute 版本可省略） |
| `claude/發現紀錄_量化校正偏斜_20260908.md` | calibration 偏斜的發現與修正 |
| `claude/交接_韌體端已排除_模型待驗證_20260908.md` | 96 vs 192 對照實驗與完整證據鏈 |
