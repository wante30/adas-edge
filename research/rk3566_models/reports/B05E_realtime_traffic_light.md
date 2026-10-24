# B-05E — 交通灯实时跟踪与高频颜色识别

日期：2026-10-24  
分支：`research/rk3566-adas-models`  
基线：`2f18cb64d0f5bc27920bb367d87be16c18b2a521`

## 0. 结论

本轮实现了“低频检测 + 新帧 LK ROI 跟踪 + 独立颜色观察”的研究代码，并补了真实视频帧测试。

已验证：

- 同一个真实路口视频片段 30 帧：LK 跟踪 **30/30 成功**；
- 颜色 **29/30 GREEN**，首帧确认在第 1 帧；
- 重复帧、乱序帧、generation 切换不会刷新证据；
- GREEN→RED 第一次观察立即 UNKNOWN，第二次 RED 才确认；
- 颜色证据 500ms 过期、检测框 1200ms 过期。

但必须明确：

- **尚未接入正式 `ai_bridge`/`fpga_perception` HUD**；
- 没有完成真实 A/B（旧链路 vs 新链路）的统一视频时间轴对比；
- 30 帧的真实跟踪是手工初始化 bbox，不是 Specialist 自动检出；
- 没有 RK3566 的 10Hz 颜色端到端性能数据。

因此本轮是**可运行的独立研究模块 + 验证**，不是 A 线正式功能。

## 1. 运行链路审计

板端实际：

```text
BDD KL：512×288，main-hz 10
Specialist：512×192，specialist-hz 1
前视 ROI：上方 60%（1280×432 → 512×192）
```

实际 Python 包根：

```text
ADAS_EDGE_ROOT=/home/cat/competition_cockpit_port_20261018
```

不是 `/home/cat/adas-edge-rect`。

当前：

- BDD class_id=8 在 `RknnConsoleDetector` 仍被 `continue`；
- HUD 颜色仍走 `fpga_perception.py` 的单状态 HSV；
- B-05D 的 Tracker 未接入；
- 1Hz Specialist + 1200ms 检测框保留已存在；
- 没有改动 A 线 Qt/FPGA。

## 2. 新 Tracker 接口

文件：

```text
competition_ai/traffic_light_tracking.py
```

新增两个独立接口：

```python
update_detections(frame, detections, source_frame_id=..., timestamp_ns=..., generation=...)
update_frame(frame, source_frame_id=..., timestamp_ns=..., generation=...)
```

### 2.1 检测更新

- Specialist 结果记录：
  - `detection_frame_id`
  - `detection_timestamp_ns`
  - bbox / confidence
- 首次颜色在检测对应帧上尝试；
- 新 track 保持独立 ID；
- 检测框 TTL 默认 1200ms。

### 2.2 帧更新

- 每个新视频帧使用 LK 光流更新 ROI；
- 只使用当前真实帧做颜色观察；
- 颜色证据 TTL 默认 500ms；
- 颜色确认默认 2 次；
- 跟踪失败、ROI 异常、证据过期时 `color=UNKNOWN` / `color_valid=False`。

### 2.3 LK 跟踪

`track_bbox_lk()`：

- `goodFeaturesToTrack` 在旧 ROI 内取特征；
- 前后向 LK；
- forward-backward 误差 ≤1.5px；
- 至少 6 个有效点；
- 位移超过 ROI 宽/高 30% 判失败；
- 边界裁剪后面积损失超过 40% 判失败。

这不是模板匹配；选择 LK 是因为 CPU 成本低、可直接验证运动。

## 3. 状态机

新增字段：

```text
detection_frame_id
color_frame_id
detection_age_ms
color_age_ms
track_valid
color_valid
detection_valid
color_ttl_ms
```

规则：

- 检测框和颜色证据分开计时；
- 颜色确认必须来自不同真实帧；
- 同一 `source_frame_id` 重复调用直接拒绝；
- 时间戳倒退/乱序拒绝；
- generation 变化清空 tracks；
- 新颜色与旧颜色冲突时立即 UNKNOWN；
- 旧 GREEN 不因 1200ms 检测框 TTL 继续 valid；
- 颜色证据超过 500ms 后 `color_valid=False`。

## 4. 测试

新增：

```text
tests/test_b05e_tracking.py
tests/data/b05e_real_light_0.png
tests/data/b05e_real_light_1.png
```

覆盖：

1. GREEN→RED，旧 GREEN 第一次即失效；
2. 两次颜色确认来自不同帧；
3. 500ms 颜色证据过期；
4. 1200ms 检测框过期；
5. 无特征帧导致 ROI 跟踪丢失；
6. 目标移动后 ROI 位置更新；
7. 多灯独立 track；
8. generation 变化后重置；
9. 重复/乱序帧不刷新。

测试结果：

```text
tests/test_b05e_tracking.py: 6 passed
tests/test_b05b_traffic_light.py: 7 passed
tests/test_b05c_ap.py: 3 passed
```

真实帧测试：

- 从 `video_20260929_113421.mp4` 路口片段抽 30 帧；
- 初始 bbox 手工给定；
- LK 跟踪 30/30 成功；
- 颜色 29/30 GREEN，首帧确认在第 1 帧。

这是**真实视频 ROI 跟踪**，但不等于 Specialist 自动检出 + HUD 全链路。

## 5. 频率/延迟

在 30 帧、30 FPS 的 PC 测试中：

```text
跟踪成功：30/30
有效颜色帧：29/30
首次有效 GREEN：第 1 帧（约 33ms 后）
```

但这是手工初始化 bbox，不代表真实首次检测延迟。  
不能把它写成“端到端颜色延迟从 X 降到 33ms”。

## 6. 远距离/有框无颜色

B-05D 的离线结论仍有效：

- 全宽 ROI 8–16px Recall 23.8%；
- center960 31.0%，但丢失两侧 20 个 GT；
- 当前没有把 center960 替换成正式配置。

本轮的 LK 跟踪只解决“已有正确框后的颜色更新”，不能提高 Specialist 对远处小灯的检测率。  
因此“远处有框、近处才有颜色”在检测层面仍未解决；颜色层面只对已有正确框有效。

## 7. 红绿互判

新状态机：

- 新 RED 第一次观察 → UNKNOWN；
- 第二次 RED → RED；
- 旧 GREEN 不会继续保持 valid。

这降低了“旧 GREEN 被当成新证据”的风险，但不能消除 HSV 本身的红绿误判。  
B-05D 的 center960 仍有 2 个 RED→GREEN，本轮没有新的真实颜色 GT 结果推翻它。

## 8. 多灯与 UNKNOWN

- 多灯独立 track；
- 不自动选“本车车道灯”；
- 跟踪失败/证据过期/颜色冲突 → UNKNOWN；
- 不把 UNKNOWN 当作识别正确。

## 9. RK3566 验收

未完成正式 10Hz 颜色链路接入，因此没有：

- 真实 Specialist 1Hz + 10Hz 颜色观察的端到端 FPS；
- 主 YOLO FPS/Qt FPS 对照；
- CPU/NPU 增量。

B-05D 的 2Hz 短测曾观察到 AI 约 77% CPU，但没有可信主 FPS/Qt FPS 对照；已恢复 1Hz。  
本轮没有把 2Hz 或 10Hz 写成正式性能结论。

## 10. 回答交付问题

1. **是否真正实现独立高频颜色观察**：研究模块实现了，真实片段 30 帧 30 次跟踪/29 次有效颜色；**未接入正式 HUD**。
2. **ROI 跟踪方法/失败率**：LK + 前后向校验；30 帧真实片段 0 失败；样本太小，不能给总体失败率。
3. **首次正确颜色延迟改善**：手工初始化下第 1 帧有效；没有可信端到端 A/B，不能给相对旧链路的改善数字。
4. **远距离有框无颜色是否改善**：检测层面没有；颜色层面只对已有框有效。
5. **红绿互判**：状态机防止旧 GREEN 继续有效；HSV 红绿误判仍存在，未增加新的安全放行逻辑。
6. **多灯/UNKNOWN**：多灯独立 track、UNKNOWN 回退已测试。
7. **RK3566 性能**：未接入正式链路，不能声称无退化。
8. **是否具备 A 线评审条件**：不具备正式接入条件；可作为研究模块评审。需要接入可选开关、10Hz 端到端测试、颜色 GT 复核后才能评审。

## 11. 修改边界

未训练、未换模型、未改 FPGA/PCIe、未改 A 线 Qt、未覆盖正式权重。  
板端 2Hz 测试后已恢复 1Hz。

## 12. 复现

```text
tests/test_b05e_tracking.py
tests/test_b05b_traffic_light.py
tests/test_b05c_ap.py
```

真实帧：

```text
tests/data/b05e_real_light_0.png
tests/data/b05e_real_light_1.png
```

## 13. 交付

```text
competition_ai/traffic_light_tracking.py
tests/test_b05e_tracking.py
tests/data/b05e_real_light_0.png
tests/data/b05e_real_light_1.png
research/rk3566_models/reports/B05E_realtime_traffic_light.md
```

未提交模型权重、大型视频或 BDD 原图。
