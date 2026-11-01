# 统一危险事件协议

## 身份与坐标

每个事件包含 `event_id`、`event_type/kind`、`severity`、`risk_state`、`lifecycle_state`、`source_id`、`inference_mode`、`mode_epoch`、`frame_id`、`timestamp_ns`、`track_id`、`target_class`、`bbox`、`coordinate_space`、`confidence`、`evidence`、`path_valid`、`control_eligible`、`valid`、`expires_at_ns`、`confirmed_ns` 和 `synthetic`。

`event_id = bridge_id:epoch:sequence`。同一来源、主跟踪 ID 和规则类型在升级/降级时保持 ID。模型、模式、epoch 或 AI 实例变化会清除旧上下文。所有时间戳为板端单调纳秒；板端墙钟不准确，不用它比较风险时序。

规则处理 Q1 原生 640×360 的实测框。ROAD_Q1 只映射一次到 `ROAD_VIEW_1280x720`；MOSAIC_FULL 输出完整画面的 `MOSAIC_1280x720`。二进制 MVA2 维持原格式，JSON 的 `box_identities` 按二进制框顺序提供主 ID、来源和显示速度。

Qt 先验证 epoch、模式、坐标空间和有效期，再以来源及主 ID 匹配。已有显式 ID 时不能退回其他 ID；没有主 ID 的旧协议才允许正确坐标空间中的同类 IoU 后备。跨象限框保留普通显示，来源标记为 UNASSIGNED，不能控制。

## 生命周期

NORMAL 由没有当前风险的普通目标表示。WARNING 显示黄框，CRITICAL 显示红框；事件面板、视频标签、Tesla 风险色和控制检查共用事件严重程度。

EventStore 是唯一保持层：一般两次连续缺失或超过约 250 ms 未再观测就解除；任何事件最多保持到实测时间后 400 ms。一次缺失期间可短暂保留显示，但立刻令 `control_eligible=false`。GUI 在同一锁内取得检测框、事件和控制结果快照，面板及框只消费该快照，避免接收线程完成下一帧时颜色错位。Qt 不再添加一秒 latch，并用原有 200 ms 状态时钟处理过期。每次解除记录同 ID 的 CLEARED、`cleared_ns` 和原因。

事件未匹配到当前实测框时，面板继续显示且 UI 日志记录 `unmatched_hazard`，禁止把相邻车染色。风险色过期和 Tesla 视觉淡出相互独立，淡出的普通目标不保留过期红色。

## 日志

- `<label>_ai.jsonl`：检测帧、epoch、模型、实测框数量、危险事件、各阶段耗时、规则及语义统计。
- `<label>_events.jsonl`：事件转换、控制检查、请求、ACK、拒绝、超时、点刹开始/结束和控制解除。
- `<label>_risk_ui.jsonl`：GUI 实际应用时刻、事件 ID、目标 ID、严重程度、已标色目标及未匹配事件。
- `<label>_schedule.jsonl`：请求间隔、帧写入/发送耗时、GUI 忙碌比例、渲染 FPS 和退让时间。
- `<label>_scene.jsonl`：原生 Tesla 帧率和对象数。

请求发送、模拟接收以及实际硬件执行分别记录。DRY_RUN 始终 `executed=false`、`simulated=true`，不能据此声称真实车辆刹车成功。
