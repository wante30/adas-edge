# B-05C — 交通灯检测与颜色识别瓶颈定位

日期：2026-10-24  
分支：`research/rk3566-adas-models`  
范围：B-05C-1 诊断 + B-05C-2 有限安全修复；不训练、不换模型、不改 A 线 Qt/FPGA。

## 0. 结论摘要

在 B-03D 固定 50 张 BDD 图、154 个交通灯 GT 上：

- **Specialist 在 conf=0.25 的 Recall 更高**：31.8% vs BDD 12.3%；
- **BDD 的 AP50 略高**：1.68% vs Specialist 1.25%，主要因为 Specialist 在 0.001 阈值有大量低分 FP；
- **两模型融合没有增加 TP**（0.25 下都是 49），但 FP 从 85 增到 92；
- **远距离 8–16 px 灯**：BDD Recall 1.2%，Specialist 23.8%，融合 20.2%；
- **有 GT 框时 HSV 颜色识别并不差**：红 25/33、绿 64/81、黄 3/3 正确；但红→绿 3 例、绿→红 2 例，不能忽略；
- **最大的“有框无颜色”原因**：检测框缺失/过小 + Specialist 1 Hz 旧框套新帧 + 0.05/0.25 阈值冲突，不是单一 HSV 算法；
- 本轮**没有完成视频首次检测时间测量**，也没有把 B-05B Tracker 接入正式 HUD。

所有数字均来自同一批 50 张图、同一 GT；没有把合成测试当真实精度。

## 1. 板端实际部署审计

### 1.1 模型和 SHA

实际服务加载：

```text
主模型：/home/cat/adas-edge-rect/rect_models/bdd_yolov8n_512x288_int8_kl.rknn
SHA256：cf56189116c35d7af35b0669221b7e6a2b9f920eab14db45aaf87d318889af5e

Specialist：/home/cat/adas-edge-rect/tl_check/tl_specialist_512x192_hybrid2.rknn
SHA256：becfaada5da49c4dd6c143e815f8f957fb7b7ed759040d4e18c685a9c434e6f8
```

服务参数：

```text
--main-hz 10
--specialist-hz 1
--main-profile bdd10
```

BDD 约 10 Hz，Specialist 仍是 **1 Hz**。

### 1.2 BDD class_id=8 在哪里被丢弃

`rknn_detector.py` 的 `RknnConsoleDetector.detect()`：

```python
if not self.specialist:
    if class_name == "traffic light":
        continue
```

因此 BDD 的 `class_id=8 traffic light` 在 AI Bridge 结果层被完全移除，不进入 HUD。  
这是 B-05A 为消除蓝牌误报做的临时策略，不是当前 HUD 的“仅显示过滤”。

### 1.3 Specialist 输出

- 输入：前视图上方 60% ROI，原图 1280×432 → 512×192；
- 置信度门槛：Specialist 0.05；
- HUD 门槛：`MIN_SIGNAL_CONFIDENCE=0.25`；
- 两者存在冲突：0.05–0.25 的 Specialist 框在 HUD 被丢掉；
- 1 Hz 更新，`FrontPipeline.snapshot()` 当前保留 Specialist 1200 ms；
- 旧框使用当前帧做 HSV 裁剪，因此框和帧可能不同步。

### 1.4 HSV 和 Tracker

实际 HUD 颜色函数：

```text
competition_ai.traffic_light.classify_traffic_light
```

调用点：

```text
tools/competition_console/fpga_perception.py::build_hud_perception
```

B-05B 新增的：

```text
competition_ai/traffic_light_tracking.py
```

**没有接入** `ai_bridge` 或 `fpga_perception`。  
所以当前 HUD 仍是单状态、逐帧 HSV，没有多灯 Tracker、没有跨帧颜色确认。

## 2. 公平检测对照

### 2.1 数据与协议

- 50 张 B-03D 固定图：`board_holdout50`；
- GT：`b03d_gt_hf_20261021`，154 个 traffic light；
- 原图坐标 1280×720；
- BDD KL：512×288，conf=0.001，NMS IoU=0.45；
- Specialist：512×192，前视 60% ROI，conf=0.001，NMS IoU=0.70；
- Specialist 框还原到原图时保持 y 不变（ROI 从 y=0 开始），不再二次缩放；
- 匹配 IoU=0.5。

先跑低置信度保存原始预测，再离线按阈值评估；不是用 HUD 过滤后的框统计。

### 2.2 总体结果

| 来源 | AP50 | conf .25 TP/FP/FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| YOLO-BDD KL | 1.68% | 19 / 31 / 135 | 0.380 | 0.123 | 0.186 |
| Specialist | 1.25% | 49 / 85 / 105 | 0.366 | **0.318** | **0.340** |
| 融合去重 | 1.07% | 49 / 92 / 105 | 0.348 | 0.318 | 0.332 |

**回答 Q1：Specialist 的 Recall 更高**（31.8% vs 12.3%）。  
但 AP50 反而 BDD 略高，因为 Specialist 在 0.001–0.05 有大量低分误检；不能用 AP 单独判定谁更好。

### 2.3 尺寸分组 Recall（conf=0.25）

| GT 宽度 | GT 数 | BDD | Specialist | 融合 |
|---|---:|---:|---:|---:|
| <8 px | 17 | 5.9% | 11.8% | 17.6% |
| 8–16 px | 84 | 1.2% | **23.8%** | 20.2% |
| 16–32 px | 47 | 27.7% | **48.9%** | 51.1% |
| >32 px | 6 | 66.7% | 66.7% | 83.3% |

远距离 8–16 px 是核心瓶颈：BDD 基本失效，Specialist 也只有约四分之一。

### 2.4 融合

融合规则：两模型框按置信度排序，IoU≥0.5 去重。

- conf 0.25：TP 49 → 49，没有新增；
- FP 85 → 92，增加 7；
- 低阈值全量：TP 79 → 80，仅多 1 个 TP，但 FP 从 88,995 增到 66,582（仍然巨大）。

**回答 Q3：在当前数据上，融合没有真正提高 Recall，反而增加误检。**  
不建议默认开启双模型融合。

## 3. 颜色 GT 与颜色混淆

### 3.1 GT 恢复

`b03d_gt_hf_20261021` 的 50 个 JSON 包含：

```text
trafficLightColor: green / red / yellow / none
```

共 154 个 traffic light：

```text
green: 81
red: 33
yellow: 3
none: 37
```

9000 张 ROI 的 `dataset.yaml` 仍只有 `traffic light` 一类，**没有颜色标签**；不能把 ROI 图当颜色 GT。

### 3.2 GT 框裁剪后的 HSV

| GT 颜色 | 正确 | 误判为 RED | 误判为 GREEN | 误判为 YELLOW | UNKNOWN |
|---|---:|---:|---:|---:|---:|
| RED (33) | 25 | — | 3 | 1 | 4 |
| GREEN (81) | 64 | 2 | — | 4 | 11 |
| YELLOW (3) | 3 | 0 | 0 | — | 0 |

红/绿主要问题不是整体失效，而是：

- 11 个绿灯和 4 个红灯返回 UNKNOWN；
- 红→绿 3 个，绿→红 2 个，属于安全相关误判。

### 3.3 预测框裁剪后的颜色

只有检测框与 GT IoU≥0.5 才统计颜色；`NO_DET` 单独列出。

| 来源 | GT 颜色 | 匹配数 | 正确颜色 | 红↔绿错误 | UNKNOWN |
|---|---|---:|---:|---:|---:|
| BDD | GREEN | 22 | 17 | 3 红 | 1 |
| BDD | RED | 10 | 9 | 1 绿 | 0 |
| BDD | YELLOW | 2 | 2 | 0 | 0 |
| Specialist | GREEN | 38 | 30 | 0 红 | 5 |
| Specialist | RED | 19 | 16 | 2 绿 | 1 |
| Specialist | YELLOW | 3 | 3 | 0 | 0 |

可见：

- 检测框正确时，HSV 颜色并非完全不可用；
- **BDD 的 NO_DET 更多**：红灯 33 个中 23 个无框，绿灯 81 个中 59 个无框；
- Specialist 的 NO_DET 少一些：红灯 14/33，绿灯 43/81；
- Specialist 有 **2 个红灯被判绿**，不能直接用颜色触发停车/通行。

## 4. 为什么有框却识别不到颜色

按优先级：

### 4.1 框太小/框质量不足

- 8–16 px GT 占 84/154；
- BDD 该尺寸 Recall 1.2%，Specialist 23.8%；
- 小 ROI 中发光像素可能只有几个，低于 HSV 的有效像素/连通区域门槛；
- 框包含过多背景时，颜色比例被稀释。

### 4.2 旧框套新帧

Specialist 1 Hz，snapshot 保留 1200 ms。  
`fpga_perception` 用**当前帧**裁剪，但 bbox 可能来自最多 1.2 秒前的 Specialist 结果。  
车辆/灯在画面中移动后，ROI 已经不对应原灯，颜色自然 UNKNOWN 或错色。

### 4.3 1 Hz 更新 + 400/1200 ms 窗口

- 1 Hz 只在每秒约 1 次更新；
- 原来 400 ms 窗口有约 60% 时间没有结果；
- B-05B 已把板端窗口改为 1200 ms，但这只是“保留旧结果”，不是提高检测频率；
- 1200 ms 旧框还会扩大“旧框套新帧”的错配。

### 4.4 0.05/0.25 门槛冲突

Specialist 以 0.05 输出，HUD 以 0.25 过滤，导致：

```text
有 Specialist 低分框
→ HUD 收不到
→ 表现为“没有框/没有颜色”
```

### 4.5 BDD class_id=8 被完全移除

BDD 的交通灯候选没有进入 HUD，不能作为 Specialist 的补充候选。  
离线评测时 BDD 的 8–16 px Recall 只有 1.2%，所以恢复 BDD 也不解决远距离，但能提供部分粗定位。

### 4.6 HSV 本身

- 红/绿阈值在 GT 框上仍有 15 个 UNKNOWN；
- 红→绿 3 个、绿→红 2 个；
- 不是“HSV 完全失效”，但也不满足安全关键颜色判定；
- 需要多帧确认和 UNKNOWN 回退，不能单帧硬判。

## 5. 视频首次检测时间

**未完成可信测量。**

本轮没有在统一视频时间轴上记录：

```text
首次有框时间
首次稳定框时间
首次有效颜色时间
```

原因：板端当前为 LOCAL_REPLAY/PCIe 混合状态，服务与 Publisher 窗口不稳定；没有用不一致的截图冒充时间统计。  
因此不能回答“谁在真实视频中更早检测到交通灯”，只能说静态 50 帧 Recall Specialist 更高。

## 6. 安全风险

必须记录：

- Specialist 有 2 个 RED→GREEN；
- BDD 有 3 个 GREEN→RED；
- 红绿灯 TTL 过期不等于灯色仍有效；
- 1 Hz 旧框 + 当前帧 ROI 会制造错误颜色；
- 当前 HUD 的 YELLOW 被映射为 OFF（协议不支持）；
- 不能用重复发送旧 GREEN 伪造新鲜度。

B-05B 的 `TrafficLightTracker` 未接入，不能声称已经解决旧 GREEN/延迟问题。

## 7. 有限修复

本轮没有新增训练、没有改 A 线 Qt、没有改 FPGA/PCIe、没有更换模型。

已存在/已确认的板端变更：

```text
pipeline.py: specialist 400 ms -> 1200 ms（B-05B）
rknn_detector.py: main traffic light continue（B-05A 临时 HUD 过滤）
```

本轮新增的离线诊断脚本：

```text
tools/b05c_diagnose.py
```

它只读取固定预测和 GT，不改变运行服务。

## 8. 回答最终问题

1. **谁 Recall 更高**：Specialist，conf 0.25 下 31.8% vs BDD 12.3%。
2. **谁更早检测到真实视频中的灯**：没有可信时间轴，不能回答；未测。
3. **融合是否提高 Recall**：没有；0.25 下 TP 49→49，FP 85→92。低阈值只多 1 TP，但误检巨大。
4. **为什么远处有框却识别不到颜色**：小 ROI 发光像素不足、框含背景、旧框套新帧、1 Hz/1200 ms 延迟、0.05/0.25 门槛冲突；不是单一 HSV 公式。
5. **HSV 表现**：GT 框上红 25/33、绿 64/81、黄 3/3；预测框匹配后 Specialist 红 16/19、绿 30/38、黄 3/3；仍有红→绿/绿→红和 UNKNOWN。
6. **慢主要来自**：1 Hz 调度 + 旧框/时间同步 + 检测框质量；颜色算法本身是毫秒级 CPU，不是主要延迟。
7. **下一阶段**：优先高分辨率/更高质量 Specialist ROI + 帧同步/新鲜度，其次才是轻量颜色 CNN；本轮数据不支持先训练 CNN。
8. **证据 vs 推测**：
   - 实测：50 图检测/尺寸/颜色混淆、模型 SHA、1 Hz、class_id=8 过滤、TTL、Tracker 未接入；
   - 推测/未测：真实视频首次检测时间、实时颜色稳定性、板端性能收益。

## 9. 复现

板端推理脚本（50 张，BDD+Specialist，低置信度）：

```text
b05c_board_infer.py
```

离线评测：

```bash
python tools/b05c_diagnose.py \
  outputs/b05c_predictions.json \
  --images D:/Datasets/adas_edge_training/b03c_rknn_validation_20261021/board_holdout50 \
  --gt-json D:/Datasets/adas_edge_training/b03d_gt_hf_20261021 \
  --out outputs/b05c_summary.json
```

测试：

```text
tests/test_b05b_traffic_light.py
```

## 10. 交付

```text
tools/b05c_diagnose.py
research/rk3566_models/reports/B05C_traffic_light_diagnosis.md
```

未提交模型、BDD 原图或大型视频。
