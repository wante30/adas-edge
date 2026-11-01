# 控制模式与恢复

## 本地回放

`tools/start_local_replay.sh` 显式选择 `ADAS_CONTROL_MODE=DRY_RUN`。AutoBrake 使用内存适配器，不创建控制 socket；Qt HudControl 在该模式下也不绑定或发送到硬件控制端口。AI 图像/模式协议继续使用本机 38260/38261 通信，二者不是制动模拟端口。

DRY_RUN 控制检查使用 `DRY_RUN_VIRTUAL_VEHICLE`，其 `simulated=true`。这仅用于覆盖请求与 ACK 流程，不修改 `/dev/shm/adas_vehicle_status.json`，不把 LVT1 离线状态改成在线。界面显示 DRY RUN、模拟 ACK、点刹结束或危险解除。

NORMAL / WARNING 不请求紧急制动。CRITICAL 每次都进入统一决策记录，检查 ACTIVE/valid、新鲜度不超过 400 ms、有效期、Q1、主 ID、可制动类型、实测而非预测、路径、模式/epoch、车辆状态以及控制授权。请求默认每个事件至少间隔 400 ms，最多一个待 ACK 请求。ACK 等待为非阻塞轮询，250 ms 后记录超时，AI 推理不等待 80 ms。

日志含 event_id、track_id、frame_id、严重程度、来源、request_id 和请求/接收/超时/解除时间。模拟 ACK 明确 `executed=false`；只能证明软件链路。

## 斑马线点刹

生产快速层不会把单次斑马线检测绑定到制动。合成测试辅助函数要求 synthetic 的行人横穿 CRITICAL 与 MTL 关联证据；独立 UI 夹具也在屏幕注明合成。DRY_RUN 对每个点刹事件只请求一次，记录模拟执行、ACK、300 ms 结束和 CLEARED；禁止 HARDWARE 执行该测试类型。

## 硬件适配器

`tools/start_pcie.sh` 显式选 HARDWARE，保留原 HUD Bridge 38262 和现有命令格式，增加 request_id 以关联 ACK。`ADAS_AUTO_BRAKE=1` 和 `ADAS_CONTROL_AUTHORIZED=1` 表示启用/授权，可设 0 停止自动请求。真实车辆状态从 `LVT1_STATUS_FILE`（默认 `/dev/shm/adas_vehicle_status.json`）读取，要求 schema 匹配、文件新鲜、online、传感器不陈旧且非 simulated；车辆拒绝制动时不能请求。

硬件 ACK 需回传 request_id；旧桥若不回传会被记录为未匹配 ACK/超时，不能将它视为已确认执行。当前没有设备，没有验证真实动作和断链恢复。本次没有修改 PCIe 驱动、FPGA RTL 或 RKNN Runtime。

## 恢复

`tools/restore_afternoon.sh` 终止当前开发会话，按当前 boot 的记录恢复原自启动父进程和原服务。原程序、PR #6 独立目录、冻结运行包和 Git 提交均保留。恢复记录仅属于原板当前 boot，运行包不包含活动 PID；新板需从其正常 HMI 获取启动参数。
