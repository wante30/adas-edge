# B-05D — 交通灯小目标检测与实时颜色识别优化

日期：2026-10-24  
分支：`research/rk3566-adas-models`  
参考：`B05C_traffic_light_diagnosis.md`

## 0. 结论

本轮完成了三件事：

1. 修复 B-05C 的 AP50 逐图排序错误，改为所有预测全局按置信度排序；
2. 在同一 50 张图、154 个 GT 上测试了 3 个 Specialist ROI 方案；
3. 修复 `TrafficLightTracker` 的颜色新鲜度、确认次数和红绿冲突状态机，并补测试。

结果：

- 正确 AP50：BDD **10.87%**，Specialist **23.19%**，融合 **22.55%**；
- 中心 960px ROI 把 8–16px Recall 从 **23.8%→31.0%**，总 Recall **31.8%→35.1%**；
- 但中心 960px 完全丢掉画面两侧 20 个 GT（侧边 Recall 35%→0%），**不能直接替换正式配置**；
- 颜色问题仍未完全解决：中心 ROI 有 2 个 RED→GREEN，基线也有 2 个；
- Tracker 已在研究代码中修正“旧 GREEN 仍有效”的问题，但**未接入 A 线**；
- 2Hz 没有取得可信的主视频/Qt 性能对照，暂保留 1Hz。

## 1. P0：AP50 修复

B-05C 的 `ap50()` 按每张图内部排序，违反 COCO 的全局排序口径。已改为：

1. 收集所有图片的所有预测；
2. 全局按 confidence 降序；
3. 每条预测只在对应图片内匹配未使用 GT；
4. 计算 101-point AP50。

新增测试：

```text
tests/test_b05c_ap.py
```

覆盖：

- IoU；
- 全局排序（高置信度假框在前时 AP 不应按图内排序）；
- conf 阈值匹配不跨图。

### 修正后的基线

| 来源 | AP50 | conf .25 TP/FP/FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| YOLO-BDD KL | **10.87%** | 19/31/135 | 0.380 | 0.123 | 0.186 |
| Specialist | **23.19%** | 49/85/105 | 0.366 | 0.318 | 0.340 |
| 融合 | **22.55%** | 49/92/105 | 0.348 | 0.318 | 0.332 |

**回答 Q1：修复后的 AP50 是 BDD 10.87%、Specialist 23.19%。**

## 2. P1：ROI 方案对照

### 2.1 板端实际输入

板端实际 Specialist 预处理：

```text
1280×432 ROI（前视上方 60%）
→ 512×192
```

不是假定始终有 1280×720 原始前视；这是当前 local/quad 前视图的有效坐标。

### 2.2 测试方案

同一 Specialist、同一 50 张图：

| 方案 | ROI |
|---|---|
| baseline | 全宽 0..1280，上 0..432 |
| center960 | x=160..1120，上 0..432 |
| center800 | x=240..1040，上 0..432 |

所有方案仍使用 512×192，不做虚假上采样。

### 2.3 结果

| 方案 | AP50 | TP/FP/FN @.25 | Recall | Precision | 8–16px Recall | <8px Recall |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 23.19% | 49/85/105 | 31.8% | 36.6% | 23.8% | 11.8% |
| center960 | **25.75%** | 54/75/100 | **35.1%** | 41.9% | **31.0%** | 11.8% |
| center800 | 23.40% | 46/58/108 | 29.9% | 44.2% | 28.6% | 11.8% |

center960 对 16–32px 也有提高（48.9%→51.1%），但 >32px 从 66.7% 降到 33.3%（只有 6 个 GT，样本小）。

### 2.4 侧边漏检

center960 丢掉的区域：

```text
GT 位于 x<160 或 x>1120：20 / 154
baseline 侧边 Recall：7/20 = 35%
center960 侧边 Recall：0/20 = 0%
```

这是不能直接替换正式配置的原因：中心区域更好，但路侧灯完全丢失。

**回答 Q2：小目标 Recall 有提高（8–16px +7.2 个点，总 Recall +3.3 个点），但以丢失两侧灯为代价，不能直接采用。**

## 3. 颜色与“有框无颜色”

### 3.1 颜色 GT

154 个 traffic light：

```text
green 81, red 33, yellow 3, none 37
```

只使用 red/green/yellow；none 不当作颜色真值。

### 3.2 基线（全宽 ROI）

预测框与 GT IoU≥0.5 后：

| GT 色 | 匹配 | 正确 | RED→GREEN | GREEN→RED | UNKNOWN | NO_DET |
|---|---:|---:|---:|---:|---:|---:|
| GREEN 81 | 38 | 30 | 0 | — | 5 | 43 |
| RED 33 | 19 | 16 | — | 2 | 1 | 14 |
| YELLOW 3 | 3 | 3 | 0 | 0 | 0 | 0 |

### 3.3 center960

| GT 色 | 匹配 | 正确 | RED→GREEN | UNKNOWN | NO_DET |
|---|---:|---:|---:|---:|---:|
| GREEN 81 | 38 | 33 | 0 | 4 | 43 |
| RED 33 | 13 | 11 | 2 | 0 | 20 |
| YELLOW 3 | 3 | 3 | 0 | 0 | 0 |

中心 ROI 让绿颜色更稳定，但红灯检测匹配从 19 降到 13，且有 2 个 RED→GREEN。  
**颜色可靠性没有随 ROI 变好而解决。**

**回答 Q3：有框无颜色有局部改善，但仍存在；且红灯漏检/红绿误判仍是安全风险。**

## 4. P2：Tracker 帧同步与旧 GREEN

修改：

```text
competition_ai/traffic_light_tracking.py
tests/test_b05b_traffic_light.py
```

新增/修复：

- `detection_frame_id`、`color_frame_id`、`color_age_ms`；
- `color_ttl_ms=500` 独立于检测框 TTL；
- 颜色证据过期后 `valid=False`，即使 track 仍在；
- 新颜色与旧颜色冲突时，先 INVALID/UNKNOWN，连续确认后才切换；
- 可配置 `confirm_updates`；
- 连续 2 次 RED 才输出 RED，不在第一次 RED 时继续显示 GREEN。

测试覆盖：

```text
GREEN 已确认 -> 新 RED 第一次 -> UNKNOWN/invalid
第二次 RED -> RED
颜色证据 600ms 后 -> valid=False
```

**回答 Q4：研究代码中已解决；正式运行时未接入，当前 HUD 仍使用旧逻辑。**

**回答 Q5：无法给“从多少 ms 降到多少 ms”的实测延迟。**  
能做到的是：

- 检测框 TTL 1200ms；
- 颜色证据 TTL 500ms；
- 证据过期强制 UNKNOWN；
- 不复制旧 bbox 冒充 10Hz 新识别。

实际 10Hz 颜色更新需要在 A 线接入后测，不在本轮伪造。

## 5. P3：1Hz vs 2Hz

板端短时把 service 改为：

```text
--specialist-hz 2
```

观察到：

- 服务可启动；
- AI 进程约 77% CPU；
- 2Hz 会话很短，没有取得统一时间轴的主 YOLO FPS、Qt 唯一帧 FPS、首次颜色时间；
- 随后恢复 1Hz。

因此：

**没有证据表明 2Hz 值得替换 1Hz。**  
在缺少完整性能和收益对照前，保留 1Hz，不直接提高到 10Hz。

## 6. P4：真实视频验收

本轮未完成可信的视频时间轴验收：

- 没有把不同时间截图当作同帧；
- 没有伪造首次检出/首次颜色时间；
- 没有声称 RED/GREEN/YELLOW 实时稳定。

固定 50 图离线结果是可信的；板端实时结果仍标记为**未验收**。

## 7. 回答交付问题

1. **修复后 AP50**：BDD 10.87%，Specialist 23.19%。
2. **小目标 Recall**：center960 的 8–16px 从 23.8% 到 31.0%，总 Recall 31.8% 到 35.1%；但侧边 20 个 GT 全部丢失，不能直接采用。
3. **有框无颜色**：局部改善；红灯匹配下降，仍有 UNKNOWN/NO_DET，未彻底解决。
4. **旧框与新帧错位**：Tracker 研究代码已加入检测帧/颜色帧/500ms 颜色证据过期和红绿冲突失效；正式 HUD 未接入。
5. **颜色延迟**：没有实测数字；设计上颜色证据 500ms 过期，检测框 1200ms 保留，但不再把旧颜色标 valid。
6. **1Hz/2Hz**：没有可信完整对照；暂保留 1Hz。
7. **新的红绿误判**：center960 仍有 2 个 RED→GREEN；baseline 也有 2 个；没有消除。
8. **RK3566 性能**：2Hz 短测 AI 约 77% CPU，但缺少主 FPS/Qt FPS 对照；不能声称无退化。
9. **可交 A 线**：
   - 可评审：AP 修复脚本、ROI 对照结论、Tracker 新状态机测试；
   - 不可直接交：center960 正式配置（丢侧边）、任何实时颜色结论。

## 8. 修改边界

未训练、未换模型、未改 FPGA/PCIe、未改 A 线 Qt、未覆盖正式权重。  
板端 service 2Hz 测试后已恢复 1Hz；保留备份。

## 9. 复现

```bash
# BDD/Specialist 离线预测由板端脚本生成
python tools/b05c_diagnose.py \
  outputs/b05c_predictions.json \
  --images D:/Datasets/adas_edge_training/b03c_rknn_validation_20261021/board_holdout50 \
  --gt-json D:/Datasets/adas_edge_training/b03d_gt_hf_20261021 \
  --out outputs/b05c_summary.json
```

ROI 变体：

```text
tools/b05d_roi_infer.py
```

测试：

```text
tests/test_b05c_ap.py
tests/test_b05b_traffic_light.py
```

## 10. 交付文件

```text
tools/b05c_diagnose.py
tools/b05d_roi_infer.py
tests/test_b05c_ap.py
tests/test_b05b_traffic_light.py
competition_ai/traffic_light_tracking.py
research/rk3566_models/reports/B05D_traffic_light_optimization.md
```

未提交模型权重、BDD 原图或大型视频。
