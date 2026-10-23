# B-05B — RK3566 交通灯检测与红黄绿状态优化

日期：2026-10-23  
分支：`research/rk3566-adas-models`  
模型：`tl_specialist_512x192_hybrid2.rknn`（不重训、不换模型）

## 0. 结论（先读）

本轮完成了链路审计、Specialist 结果保留时间修复、多灯颜色/时序模块和测试，但**没有完成对 A 线的集成，也没有取得可信的颜色实测指标**。

关键事实：

1. Specialist 只在 **1 Hz** 更新，但 HUD 原来只保留 **400 ms**，约 60% 时间没有 specialist 结果；
2. 颜色不是 Specialist 模型输出的，而是 `competition_ai.traffic_light.classify_traffic_light` 的 HSV 后处理；
3. 9000 张 ROI 的 labels 只有 `0: traffic light`，**没有颜色标签**，不能拿它们计算颜色 Precision/Recall；
4. 当前只暴露一个 `TrafficLightEstimate`，没有多灯统一接口；
5. 因此本轮不能声称 RED/GREEN/YELLOW 在真实道路上可靠，也不能交给 A 线正式集成。

## 1. 现有 Specialist 的真正瓶颈

### 1.1 Specialist 结果窗口过短

`FrontPipeline.snapshot()` 原逻辑：

```python
self.specialist if sa is not None and 0 <= sa < 400 else []
```

Specialist 频率：

```text
--specialist-hz 1
```

400 ms TTL + 1 Hz 更新意味着：

```text
每秒只有约 400 ms 有 specialist 框
约 60% 时间 HUD 看到 NO_SIGNAL_ROI
```

这不是颜色算法本身，而是**检测结果在管线里被过早丢弃**。

本轮板端修改：

```text
400 ms -> 1200 ms
```

文件：

```text
/home/cat/competition_cockpit_port_20261018/tools/competition_console/pipeline.py
```

备份：

```text
pipeline.py.bak-b05b
```

这是本轮唯一直接改板端的运行逻辑；不涉及 A 线 Qt、FPGA 或正式权重。

### 1.2 颜色来源

`fpga_perception.py` 的 `build_hud_perception()`：

- 收集 `snapshot.specialist` 和 `snapshot.main` 中的 `traffic light`；
- 用 `classify_traffic_light(..., require_lamp_evidence=True)` 做 ROI HSV；
- 多灯只在**当前帧**去重，没有跨帧 Track/确认；
- 只输出一个 `TrafficCode`（RED/GREEN/OFF），YELLOW 被 HUD 协议视为 unsupported。

`BoundSignal` 是另一条显式绑定路径，要求人工选定灯壳；它不是自动多灯识别。

### 1.3 数据集

路径：

```text
D:\Datasets\adas_edge_training\tl_specialist\data_512x192
```

`dataset.yaml`：

```yaml
names:
  0: traffic light
```

`dataset_card.json`：

```text
classes: ["traffic light"]
candidate_images_with_tl: 5080
total_images: 9000
val_images: 487
boxes.val: 1185
```

结论：**有灯框 GT，没有 RED/GREEN/YELLOW 颜色 GT**。  
BDD100K 原始颜色属性没有随这个 ROI 数据集保存，按现有文件不能恢复可靠的颜色标签。

## 2. 本轮代码

新增：

```text
competition_ai/traffic_light_tracking.py
tests/test_b05b_traffic_light.py
```

功能：

- 每个 ROI 做亮/饱和 blob 的 HSV 红/黄/绿分类；
- 蓝牌 hue 不在 R/Y/G 范围，直接 UNKNOWN；
- 同一目标需连续 2 次相同颜色才确认；
- IoU 关联，最多保留 1200 ms；
- 多灯分别保留，不合并成一个状态；
- 输出字段满足 A 线接口：
  ```text
  source_id, source_frame_id, timestamp_ns,
  bbox, confidence, color, color_confidence,
  valid, ttl_ms, track_id, age_ms
  ```

测试结果：

```text
5 passed
```

覆盖：

- 红/黄/绿合成灯；
- 黑 ROI 和蓝牌 → UNKNOWN；
- 两次确认；
- 两个灯保持两条记录；
- 过 TTL 删除。

注意：测试是合成图像，不是真实道路颜色精度。

## 3. RED/GREEN/YELLOW 是否可靠

**没有足够证据说可靠。**

原因：

- ROI 数据没有颜色 GT；
- 真实道路的逆光、白平衡、远距离小灯、灯壳反光没有标注；
- 本轮没有完成 30–50 个有颜色 GT 的人工核查；
- 合成图测试只能证明代码路径，不证明真实精度。

因此只能报告：

```text
颜色代码路径：已实现
真实颜色 Precision/Recall：无法计算
RED/GREEN/YELLOW 可靠性：未验证
```

不编造指标。

## 4. 远距离交通灯

没有实质改善。

本轮没有：

- 提高 Specialist 输入分辨率；
- 重训；
- 新增检测模型；
- 改用更大的前视裁剪。

1200 ms TTL 只减少“已有检测被过早丢弃”，不会提高远距离小灯的检测能力。  
远距离漏检仍然是 Specialist 输入分辨率/训练数据问题。

## 5. 路牌误报

- 主 YOLO 的 traffic light 已从主模型输出中移除；
- Specialist 仍可能把蓝色路牌等误检为 traffic light；
- 新颜色模块对蓝牌 hue 返回 UNKNOWN，但这只影响颜色，不会删除 Specialist 的框；
- 没有颜色 GT，无法量化路牌误报下降。

因此不能声称路牌误报已解决。

## 6. 多灯与时序

新增模块按 IoU 保留多个灯，每灯独立颜色、独立 TTL；冲突/低置信度输出 UNKNOWN。  
但它**尚未接入** `ai_bridge`/`fpga_perception` 的 A 线数据路径，所以当前 Qt HUD 仍使用旧的单状态逐帧 HSV 逻辑。

本轮没有擅自改 A 线 Qt 工程；接入需要 A 线确认接口和显示位置。

## 7. RK3566 验收

板端尝试：

- Specialist 1 Hz；
- TTL 400→1200 ms 已改；
- Local Publisher 在重启后进入长时间 raw/CRC 准备，短时窗口内未稳定产出 PGVF；
- 8 秒回放短测出现 `AI sidecar 未连接`，没有取得可信的交通灯颜色截图；
- 因此本轮不报告板端颜色准确率或性能收益。

当前不能声称：

```text
RED/GREEN/YELLOW 在 RK3566 实时稳定
增加颜色识别后性能无退化
```

能确认的是：

- 颜色模块本身是纯 CPU HSV，单 ROI 开销很小；
- 没有新增模型；
- 未修改正式权重；
- pipeline TTL 修改可回滚。

## 8. 回答交付问题

1. **真正瓶颈**：Specialist 1 Hz + 原 400 ms 结果窗口（约 60% 丢弃），以及没有颜色 GT/多灯时序；不是模型直接输出颜色。
2. **RED/GREEN/YELLOW 是否可靠**：不能证明；只有合成测试，真实颜色 GT 缺失。
3. **远距离是否改善**：没有；未改分辨率/模型。
4. **路牌误报是否减少**：未量化；蓝牌颜色会 UNKNOWN，但 Specialist 框仍在。
5. **性能影响**：颜色模块开销小，但未完成板端可信测量，不能给百分比。
6. **是否达到 A 线集成条件**：**没有**。需要颜色 GT、30–50 实例人工核查、多灯接口接入和板端验收。

## 9. 建议下一步

1. 从 BDD 原始 `det_20` 恢复 `trafficLightColor`，或人工标注 100–200 个 ROI 颜色；
2. 将新 `TrafficLightTracker` 接入 `ai_bridge` 输出，保留多灯记录；
3. 将 Specialist TTL 与频率作为正式参数评审（1 Hz + 1200 ms 仍是折中）；
4. 对远距离小灯单独评估输入裁剪/分辨率，不在本轮擅自改模型；
5. 完成板端性能与颜色验收后再交 A 线。

## 10. 交付文件

```text
competition_ai/traffic_light_tracking.py
tests/test_b05b_traffic_light.py
research/rk3566_models/reports/B05B_traffic_light.md
```

未提交模型权重、BDD 原图或大型视频。
