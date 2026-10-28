# ADAS 事件引擎计划（最小侵入式）

更新日期：2026-10-28  
目标：在**不破坏当前稳定 HMI/RKNN/PCIe** 的前提下，把现有 YOLO 检测升级为“带 track_id 的感知 + 场景事件”，再把事件交给 FPGA 状态机。  
原则：先软件后硬件、先 HMI 后协议、先可解释规则后学习模型。

---

## 1. 现有基础

| 能力 | 现有实现 | 状态 | 可直接复用 |
|---|---|---|---|
| 检测 | BDD Raw-head INT8 / P2 / Specialist | 实板 A 运行 10 Hz | ✅ |
| 检测框 | MVA2 boxes + status | 实板运行 | ✅ |
| 轻量跟踪 | `competition_ai/image_tracks.py`（IoU + 18 帧历史） | 实验 | ✅ 作为 baseline |
| HMI 跟踪 | `tools/adas_hmi_sim/backend/object_tracker.py`（IoU + EMA） | PC 仿真 | ✅ 思路 |
| 危险事件 | `competition_ai/hazard_events.py`（图像空间） | 实验 | ✅ 骨架 |
| 交通灯 | `traffic_light.py` HSV + B05L Specialist | 部分 | ✅ |
| 信号绑定 | `bound_signal.py` ROI 灯组 | 部分 | ✅ |
| 场景决策 | `scenario_policy.py` | 软件完成 | ✅ |
| 车道 | FPGA HSST 邮箱 0x680–0x68c | 实机有数据，HMI 未消费 | ✅ 读取 |
| 导航 | `q2_navigation_reader.py` + OCR | 1.2 s，常 UNKNOWN | ⚠️ |
| HUD 回写 | HUD V2 | 实板 applied | ✅ |
| 协议 | ADSF/ADSR mock、ADST v1、HUD V2 | 分层 | ✅ 参考 |

### 1.1 现有事件骨架的边界

`hazard_events.py` 已经能产生：

- `PEDESTRIAN_IN_PATH`
- `PEDESTRIAN_CROSSING_PATH`
- `LEAD_RAPID_CLOSING`
- `LEAD_BRAKE_LAMP_RISE`
- `SUSPECTED_BRAKING_WITH_CLOSING`

但它明确标注：

- 路径是**图像空间假设多边形**，不是检测到的车道；
- `visual_tau_s` 是图像膨胀估计，不是标定 TTC；
- `sudden_braking = UNKNOWN_NO_EGO_SPEED_OR_CALIBRATED_RANGE`；
- `pedestrian_red_violation = UNKNOWN_NO_ASSOCIATED_PEDESTRIAN_SIGNAL`；
- 不能把“刹车灯变亮”称为确认急刹；
- 不能从车辆灯色认定行人闯灯。

这些边界必须保留到新事件引擎中。

---

## 2. 建议的统一感知消息

### 2.1 逻辑 schema

每条检测/跟踪目标：

```json
{
  "stream_id": 0,
  "frame_id": 123456,
  "timestamp_ns": 9004786595905,
  "class_id": 0,
  "class_name": "person",
  "confidence": 0.31,
  "bbox": [x1, y1, x2, y2],
  "track_id": 42,
  "track_state": "confirmed",
  "age": 7,
  "missed": 0,
  "history": [[t_ns, cx, cy, w, h], ...],
  "motion": {
    "vx_px_s": 12.3,
    "vy_px_s": -4.1,
    "expansion_per_s": 0.08,
    "lane_relative": "ego"
  },
  "valid": true,
  "source": "yolo_raw_head_int8",
  "ttl_ms": 250
}
```

每条事件：

```json
{
  "event_id": "evt-...",
  "kind": "PEDESTRIAN_CROSSING_PATH",
  "severity": "WARNING",
  "stream_id": 0,
  "frame_id": 123456,
  "timestamp_ns": 9004786595905,
  "track_id": 42,
  "ttc_s": null,
  "confidence": 0.62,
  "source": "IMAGE_TRAJECTORY_EXPERIMENTAL",
  "valid_for_ms": 250,
  "reason": "lateral_motion_confirmed"
}
```

### 2.2 二进制承载

短期（不改 FPGA 协议）：

- 继续用 MVA2 的 box V2（7 floats + class_id + confidence）；
- 把 `track_id`、`track_state`、`events[]` 放进 MVA2 的 status JSON（≤4096 B）；
- Qt 只读 JSON，向后兼容。

中期（需要 FPGA 消费）：

- 新增 `ADST v2` 或 `EVT1` 消息，建议字段：

```text
magic "EVT1", version u16, header_len u16, sequence u32
stream_id u32, frame_id u64, timestamp_ns u64
event_count u16, source u8, valid_for_ms u16
per event:
  event_kind u8, severity u8, track_id u16
  ttc_q8 u16 (0xffff = unknown), confidence_q8 u8
  bbox 4×u16, flags u16
CRC32
```

- 不要直接把事件塞进 HUD V2 的 0x638 位域；HUD 是显示控制，不是事件总线。
- 若 FPGA 只需要“刹车/行人/红灯”三个布尔，可先映射到现有 HUD 字段；完整事件另走新邮箱。

---

## 3. 最小侵入式接入点

### 3.1 跟踪器接入点

```text
ai_bridge.py
  RknnConsoleDetector.infer()
    → Raw-head decode/NMS → detections[]
    → [新增] tracker.update(detections, frame_id, timestamp_ns)
    → tracks[]
    → MVA2 boxes + status(track_id/events)
```

- 不改 RKNN 模型/输入/输出；
- 不改 PGVF/PCIe；
- 不改 Qt 视频渲染；
- 跟踪器 CPU 开销目标 < 3 ms/帧（10 Hz 下 < 3% 单核）。

### 3.2 事件引擎接入点

```text
FrontPipeline
  → HazardEvents.update(frame, detections, frame_id, timestamp_ns)
  → [扩展] EventEngine.update(tracks, ego_state, lane_state, signal_state, navigation)
  → events[]
```

- 现有 `HazardEvents` 保留为“图像空间兜底”；
- 新 `EventEngine` 用 track 历史 + 本车速度 + 车道 + 信号；
- 事件输出到 MVA2 status JSON；
- HMI 只在“危险事件”卡增加条目，不改变布局。

### 3.3 FPGA 接入点

```text
Qt/AI events
  → hud_bridge.py（现有 HUD V2 字段：pedestrian/collision/red/green/lane）
  → 或新 EVT1/ADST v2 邮箱
  → FPGA 状态机
```

- 先映射到现有字段，验证“事件→HUD applied”；
- 再新增事件邮箱，避免 HUD 位域膨胀；
- FPGA 端需要状态机消费、超时/失效、优先级、去抖。

---

## 4. 事件逐项设计

### 4.1 行人进入本车道 / 横穿

| 项目 | 内容 |
|---|---|
| 当前感知基础 | YOLO person 框；`PEDESTRIAN_IN_PATH`/`PEDESTRIAN_CROSSING_PATH` 实验 |
| 可复用开源 | ByteTrack/BoT-SORT（跟踪）、Supervision（标注/区域）、Kalman（平滑） |
| 缺少输入 | 本车道多边形/消失点、行人脚点、自车速度、行人信号灯归属 |
| RK 端 | tracker + 脚点投影 + 横向速度/进入区域确认 + 2 帧去抖 |
| FPGA 端 | 接收 `PEDESTRIAN_DANGER`/`PEDESTRIAN_CROSSING`，刹车/报警；保留超时 |
| HMI | 危险卡显示“行人横穿/进入本车道”，框高亮 |
| 资源风险 | CPU +1–2 ms；无 NPU 增量 |
| 验收 | 真实/标注视频正例、反例、首次预警时间、误报率；不能只用单车灯色 |

### 4.2 前车 Cut-in

| 项目 | 内容 |
|---|---|
| 当前基础 | 车辆 track 历史 + 横向速度 |
| 可复用 | ByteTrack/BoT-SORT + Kalman + lane assignment |
| 缺少 | 本车道边界、自车速度、相邻车道归属 |
| RK 端 | 横向位移阈值 + 连续 3 帧 + 进入本车道；区分正常变道/加塞 |
| FPGA 端 | 接收 `CUT_IN` 事件 + TTC；制动准备/报警 |
| HMI | 前车框变色 + “前车切入”提示 |
| 资源风险 | 低（CPU） |
| 验收 | 正例/负例（邻车未切入）、首次预警时间、误报率 |

### 4.3 前车碰撞风险 / TTC

| 项目 | 内容 |
|---|---|
| 当前基础 | 图像膨胀 `visual_tau_s`（实验，非标定） |
| 可复用 | 单目距离估计/车牌/车宽先验；Kalman；TTC = distance / closing speed |
| 缺少 | 相机标定、前车真实距离/相对速度、自车速度 |
| RK 端 | 先做“图像空间 TTC 候选”，明确标注 `UNVERIFIED`；有标定后升级 |
| FPGA 端 | 接收 `COLLISION_WARNING` + TTC/severity；刹车 |
| HMI | 危险卡显示 TTC/“碰撞风险”，不伪造精确米数 |
| 资源风险 | 低–中（CPU） |
| 验收 | 标定距离误差、TTC 误差、正例/负例；无标定前只能验收“候选” |

### 4.4 交通灯状态与停车判断

| 项目 | 内容 |
|---|---|
| 当前基础 | YOLO TL 框 + HSV + B05L Specialist + `bound_signal.py` |
| 可复用 | 多帧投票、ROI 绑定、颜色空间规则 |
| 缺少 | 本车道控制灯归属、箭头灯、夜间/远距离颜色 GT |
| RK 端 | 检测 + 颜色 + 多帧确认 + 归属；UNKNOWN 时禁止通行 |
| FPGA 端 | 接收 RED/GREEN/YELLOW/UNKNOWN；红灯停、绿灯行、强制红行报警 |
| HMI | 信号灯卡显示颜色/置信度/来源；UNKNOWN 显示“未识别” |
| 资源风险 | Specialist 1 Hz 约 40–50 ms/次；NPU 争用 |
| 验收 | 路口视频逐灯组 Precision/Recall、颜色混淆矩阵、首次正确时间；当前“未识别”不能算完成 |

### 4.5 前方拥堵

| 项目 | 内容 |
|---|---|
| 当前基础 | 车辆数/密度/低速 track |
| 可复用 | 区域计数、密度估计、跟踪 |
| 缺少 | 自车速度、车道区域、时间窗口 |
| RK 端 | 本车道车辆数 + 平均速度 + 持续 2 s |
| FPGA 端 | 接收 `CONGESTION`，降速/提示 |
| HMI | 导航/危险卡显示“前方拥堵” |
| 资源风险 | 低 |
| 验收 | 拥堵/畅通片段、误报率；不能只按车辆数 |

### 4.6 前车急减速候选

| 项目 | 内容 |
|---|---|
| 当前基础 | `LEAD_BRAKE_LAMP_RISE` + 图像膨胀 |
| 可复用 | 刹车灯检测（红亮对）、track 历史、TTC |
| 缺少 | 自车速度/距离/标定；刹车灯样式泛化 |
| RK 端 | 刹车灯 + 膨胀 + 持续帧确认；输出“候选” |
| FPGA 端 | 接收 `HARD_BRAKING_CANDIDATE`，制动准备 |
| HMI | 显示“前车急减速候选”，不写“已确认急刹” |
| 资源风险 | 低–中 |
| 验收 | 正例/负例、首次预警、不能把夜间尾灯当刹车灯 |

### 4.7 车道偏离 / 合法变道

| 项目 | 内容 |
|---|---|
| 当前基础 | FPGA HSST 左右数量/类型；`scenario_policy.py` 实线/虚线规则 |
| 可复用 | HSST 读取、车道时间序列、转向灯状态 |
| 缺少 | 本车横向位置/车道 ID、转向灯、车道几何 |
| RK 端 | 读 HSST + 转向灯 + 横向运动；判定合法/非法变道 |
| FPGA 端 | 接收 `LANE_DEPARTURE`/`SOLID_LINE_CHANGE`；报警/拒绝 |
| HMI | 车道卡显示左右类型/数量；变道按钮反馈 |
| 资源风险 | 低（只读 BAR1） |
| 验收 | 实线/虚线/遮挡/弯道/路口；HSST 超时/未知时拒绝许可 |

### 4.8 导航 + FPGA 状态机联合

| 项目 | 内容 |
|---|---|
| 当前基础 | 导航 OCR（慢）、`scenario_policy.py` 导航提示 |
| 可复用 | OCR/结构化导航输入、状态机 |
| 缺少 | 实时导航源、路口/车道映射、过期策略 |
| RK 端 | 导航解析（或外部结构化输入）+ 与视觉/车道融合 |
| FPGA 端 | 接收转向/直行/距离；与红灯/危险优先级仲裁 |
| HMI | 导航卡显示方向/距离/来源；无效时“暂不可读” |
| 资源风险 | OCR 1.2 s，必须异步/降频；不能阻塞主循环 |
| 验收 | 真实导航视频、方向/距离正确率、过期/冲突处理 |

---

## 5. 事件优先级与仲裁

建议优先级（高→低）：

```text
1. 碰撞风险 / 行人横穿（安全）
2. 红灯/未知信号（法规）
3. 实线变道/车道偏离（法规）
4. 前车急减速/Cut-in（安全辅助）
5. 拥堵/导航（舒适/路线）
```

规则：

- 任一安全事件有效 → 覆盖导航/巡航；
- 事件过期（> TTL）→ 不得继续授权；
- 多源冲突 → 取更保守动作；
- UNKNOWN 不得当成“安全”；
- 同一 track 的事件去抖/滞回，避免闪烁。

---

## 6. FPGA 协议容量分析

| 协议 | 现有字段 | 能否容纳 track_id/TTC | 建议 |
|---|---|---|---|
| HUD V2 0x638 | red/green/pedestrian/static/overspeed/lane/LKA | 否（位域满，且无 track_id/TTC） | 只放布尔/灯色/车道；事件另走新消息 |
| ADST v1 60B | light/left_lane/right_lane/action/valid/warnings | 否（无 track_id/TTC/event list） | 可作“整车状态”；事件用 ADST v2 |
| ADSR 42B | 12B detection（class/conf/bbox） | 否（无 track_id） | 可加 v2 detection：track_id/event_flags |
| ADSF 52B | frame metadata | 否 | 保持不变 |
| 新 EVT1 | 自定义 | 是 | 推荐：事件总线，CRC，TTL，sequence |

建议：

1. 先不改 FPGA 协议，用 MVA2 status JSON 做 HMI 事件展示；
2. 需要 FPGA 消费时，新增 `EVT1`/`ADST v2`，不要挤 HUD；
3. 事件消息必须带 `frame_id/timestamp_ns/valid_for_ms/sequence`；
4. FPGA 端必须实现超时/失效/去重/优先级；
5. 保留 HUD V2 作为显示控制，不把事件当显示位。

---

## 7. 分阶段路线

### Phase 1（1–2 周，最小改动）

- 在 `ai_bridge.py` 接入 ByteTrack/BoT-SORT（或先用现有 `ImageTracker`）；
- MVA2 status JSON 增加 `track_id`/`track_state`/`events[]`；
- HMI 检测框显示 ID；危险卡显示事件；
- 不改 FPGA 协议、不改 RKNN、不改 PGVF；
- 验收：10 Hz 下 CPU 增量 < 3 ms，ID 稳定，事件不闪烁。

### Phase 2（2–4 周）

- 接入 HSST 车道（0x680–0x68c）；
- 事件引擎加入 TTC/Cut-in/行人横穿/拥堵/急减速候选；
- 用现有 HUD 字段做“事件→FPGA applied”验证；
- 导航 OCR 异步/降频，或接外部结构化导航；
- 验收：每个事件有正例/负例/首次预警时间。

### Phase 3（比赛前）

- 定义 EVT1/ADST v2；
- FPGA 状态机消费事件；
- 实车/台架 1080P 3 分钟演示；
- 失败回退：事件引擎关闭时，HMI/RKNN/PCIe 保持原状。

---

## 8. 资源与风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| NPU 已 100% | 加跟踪/事件不能加 NPU 负载 | 跟踪/事件只用 CPU；不新增模型 |
| CPU 单核 69.5% | 跟踪/事件可能挤 Qt | 限制 10 Hz、用轻量 tracker、Nice/绑核 |
| 无标定 | TTC/距离不可信 | 明确 `UNVERIFIED`，只做候选；不伪造米数 |
| 无本车速度 | 急减速/拥堵不可确认 | 用模拟速度/FPGA 状态，标注来源 |
| 无车道 ID | Cut-in/偏离误判 | 先用 HSST 边界 + 图像区域，保守拒绝 |
| 交通灯 UNKNOWN | 不能授权通行 | 保持红灯/UNKNOWN 停车策略 |
| 导航 OCR 慢 | 阻塞/过期 | 异步、降频、外部输入 |
| 协议膨胀 | FPGA 不愿改 | 先 JSON/HMI，后新消息；HUD 不动 |
| 事件闪烁 | 驾驶员不信任 | 去抖/滞回/TTL/优先级 |

---

## 9. 验收指标

每个事件至少记录：

- 正例/负例片段与逐帧 GT；
- 首次预警时间（相对人工标注）；
- Precision/Recall/F1（按事件，不按框）；
- 误报/漏报样本；
- 事件持续/闪烁次数；
- CPU/NPU/内存增量；
- 事件→HUD applied 延迟；
- 失效/超时行为。

**没有这些，不得写“已完成事件识别”。**

---

## 10. 相关代码位置

| 功能 | 位置 |
|---|---|
| 跟踪 baseline | `competition_ai/image_tracks.py` |
| 危险事件 | `competition_ai/hazard_events.py` |
| 场景策略 | `competition_ai/scenario_policy.py` |
| 交通灯 | `competition_ai/traffic_light.py`、`bound_signal.py` |
| AI 结果组装 | `deployment/lubancat/qt_gpu_probe/ai_bridge.py` |
| HMI 框/危险卡 | `deployment/lubancat/qt_gpu_probe/main.qml`、`video_item.cpp` |
| 3D BEV | `deployment/lubancat/qt_gpu_probe/tesla_scene_item.cpp` |
| HUD | `deployment/lubancat/qt_gpu_probe/hud_bridge.py` |
| HSST | `deployment/lubancat/qt_gpu_probe/pcie_protocol.py`（待接） |
| 协议 | `competition_io/protocol.py`、`adas_status_v1.py` |
