# B-05F — 交通灯实时链路接入及 A/B 验收

日期：2026-10-24  
分支：`research/rk3566-adas-models`  
基线：`54367019d19bd06d6463033f7d43b8b90531b826`

## 0. 结论

本轮完成了：

- 异步 Tracker：支持主视频 105 帧、Specialist 返回 100 帧的迟到检测；
- 帧缓存、检测帧到当前帧的 LK 传播、一对多匹配拒绝、source/generation 重置；
- `effective_color` 安全字段；
- 真实视频帧上的自动 Specialist 初始框 + LK 跟踪 + 颜色观察；
- 29 个测试通过。

但尚未完成：

- 没有把 Tracker 接入正式 `ai_bridge`/`fpga_perception`；
- 没有 A 线 HUD 的 A/B；
- 没有 RK3566 的约 10Hz 颜色观察性能数据；
- 没有小目标（<16px）真实视频跟踪统计；
- 没有真实红绿切换视频。

因此本轮不是“A 线正式接入完成”，而是**异步 Tracker 和真实帧验证完成，正式集成待评审**。

## 1. P0：异步迟到检测

新增：

```text
competition_ai/traffic_light_async.py
```

核心流程：

```text
push_frame(frame, frame_id, timestamp, generation)
→ 缓存最近 60 帧
→ update_detections(detections, detection_frame_id=100,
                    current_frame_id=105, ...)
→ 从 100 到 105 逐帧 LK 传播 bbox
→ 在当前帧关联 Track
```

特点：

- 检测时间使用 `detection_timestamp_ns`，不是推理完成时间；
- 旧检测不能覆盖较新的 Track；
- 同批次检测使用 `used` 集合，不能重复匹配同一 Track；
- source 变化、generation 变化、frame_id 回绕/倒退时 reset；
- 无法从检测帧推进到当前帧时拒绝，返回 `tracking_from_detection_failed`；
- 检测超过 1200ms 拒绝。

输出字段：

```text
source_id, generation, track_id,
detection_frame_id, color_frame_id,
detection_timestamp_ns, color_timestamp_ns,
detection_age_ms, color_age_ms,
track_valid, color_valid, effective_color,
debug_color
```

`effective_color` 规则：

```text
color_valid=True  -> 实际颜色
color_valid=False -> "UNKNOWN"
```

历史颜色只保留在 `debug_color`，不允许作为有效输出。

## 2. 真实 Specialist 自动检测

在真实路口片段的首帧上，用板端 Specialist 自动推理，得到：

```text
绿灯 bbox：565.7, 367.0, 582.8, 395.4
confidence：0.318
```

不是手工 bbox。

随后在 30 帧 1280×720 真实视频上：

- 首帧用该自动检测初始化 Track；
- 后续 29 帧全部用 LK 跟踪；
- 每帧尝试颜色识别。

结果：

```text
跟踪成功：30/30
首次有效颜色：第 1 帧
GREEN：29 帧
UNKNOWN：1 帧
```

这证明“自动 Specialist 框 → LK → 连续颜色”的路径可运行。

## 3. A/B 结果

同一 30 帧、同一个绿灯：

| 项目 | 旧 1Hz + 单帧 HSV | 新 1Hz + LK + 逐帧颜色 |
|---|---|---|
| 初始框 | Specialist 自动 | Specialist 自动 |
| 颜色更新 | 检测帧一次 | 30 帧逐帧 |
| 首次有效颜色 | 检测帧（本样本 GREEN） | 第 1 帧 |
| 连续有效颜色帧 | 1 | 29 |
| RED/GREEN 切换 | 本片段无 | 本片段无 |

**不能声称首次颜色延迟降低**：本样本旧链路在检测帧已经得到 GREEN，新链路反而到第 1 帧才确认。  
真正改善是**后续每帧持续观察**，不是首次颜色更快。

片段没有红绿灯切换，不能虚构 RED/GREEN 切换实验。

## 4. 多灯/匹配测试

测试：

```text
tests/test_b05f_async.py
```

覆盖：

- 105/100 迟到检测并传播；
- 两个检测不能匹配同一 Track；
- 旧检测不能覆盖新 Track；
- source/generation 切换；
- 重复帧；
- 小 ROI/特征不足；
- effective_color 强制 UNKNOWN；
- 颜色时效。

结果：

```text
5 passed
```

## 5. 小目标 LK

本轮**没有完成**按 <8px/8–16px/16–32px/>32px 分组的真实视频 LK 统计。

自动检测的样本是约 17×28px 的灯，不是 <16px 小目标。  
因此：

- 不能报告小目标 LK 成功率；
- 不能声称远距离有框无颜色已解决；
- 不能把 30/30 的大灯样本外推到小灯。

这是本轮主要缺口之一。

## 6. 红绿互判与安全

状态机：

- 新颜色与旧颜色冲突时先 UNKNOWN；
- 2 次确认后才输出；
- 颜色证据 500ms 过期；
- track 丢失时 effective_color=UNKNOWN；
- 不使用旧 GREEN 刷新证据。

测试覆盖 GREEN→RED 第一次失效、第二次确认。  
但本视频片段没有红绿切换，因此没有新增真实道路红绿误判统计。

B-05D 的离线颜色结果仍显示 RED→GREEN 误判存在；本轮没有消除 HSV 的根本误差。

## 7. RK3566 性能

**未完成正式集成性能验收。**

原因：

- Tracker 尚未接入 `ai_bridge`/`fpga_perception`；
- 没有约 10Hz 颜色输出在板端的真实运行数据；
- 没有主 YOLO FPS/Qt FPS 对照。

因此不能报告 CPU/NPU/内存增量，也不能声称无退化。

## 8. 回答交付问题

1. **Specialist 异步检测是否正确接入**：研究模块正确处理迟到检测；正式 AI Bridge 未接入。
2. **是否实现约 10Hz 颜色观察**：PC 30 帧测试逐帧颜色可运行；板端 10Hz 未验收。
3. **首次正确颜色延迟降低多少**：没有降低；本样本旧链路检测帧已 GREEN，新链路第 1 帧才确认。
4. **远处有框无颜色是否改善**：没有完成小目标验证；本样本是较大灯。
5. **多灯跟踪是否稳定**：单元测试稳定；真实多灯视频未验收。
6. **<16px LK 成功率**：未测，不能报告。
7. **红绿互判是否增加**：状态机降低旧色残留；HSV 根本误判未消除，本片段无切换。
8. **RK3566 主 YOLO/视频 FPS 是否退化**：未接入，不能回答。
9. **哪些情况强制 UNKNOWN**：
   - 检测帧无法推进到当前帧；
   - 检测超过 1200ms；
   - LK 特征不足/位移异常/边界损失；
   - 颜色证据超过 500ms；
   - 颜色冲突未确认；
   - track 丢失；
   - `color_valid=False`。
10. **是否达到 A 线接口评审条件**：没有。需要接可选开关、真实 10Hz 端到端、小目标统计、性能验收和真实红绿切换。

## 9. 代码与测试

新增：

```text
competition_ai/traffic_light_async.py
tests/test_b05f_async.py
research/rk3566_models/reports/assets/B05F_ab.json
```

现有：

```text
competition_ai/traffic_light_tracking.py
tests/test_b05e_tracking.py
```

测试：

```text
tests/test_b05f_async.py: 5 passed
tests/test_b05e_tracking.py: 6 passed
tests/test_b05b_traffic_light.py: 7 passed
tests/test_b05c_ap.py: 3 passed
tests/test_b05a_bdd_postprocess.py: 8 passed
```

## 10. 修改边界

未训练、未换模型、未改 FPGA/PCIe、未改 A 线 Qt 正式输出、未覆盖正式权重。  
预览/测试使用 PC 离线 LK；板端正式服务未接入新 Tracker。

## 11. 交付与复现

报告：

```text
research/rk3566_models/reports/B05F_realtime_integration_ab.md
```

A/B JSON：

```text
research/rk3566_models/reports/assets/B05F_ab.json
```

完整视频/模型不提交 Git。
