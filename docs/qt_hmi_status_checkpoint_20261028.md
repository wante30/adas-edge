# Qt 上位机状态 Checkpoint

更新时间：2026-10-28

## 当前结论

Qt 上位机和模型选择面板仍在运行，界面当前显示方案 A，状态为 `APPLIED`。方案选择功能此前在板端实测过 A → B → A，界面均收到 `APPLIED`。

当前端到端视频与 AI 不可用：PCIe 状态为断开，视频为 0 FPS，AI 状态为等待，界面提示 AI sidecar 未连接。主视区仍显示道路画面，但在新帧计数为 0 的情况下不能视为实时画面，检测框也不能视为当前有效结果。

## 当前观测

- Qt 窗口和模型面板可见，当前选择 A；截图：`C:\Users\JingyuWang\AppData\Local\Temp\qt_status_check_20261028.png`。
- `/dev/shm/pango-video/status.json`：`state=error`、`frames=0`、`fps=0`、`error=capture_io`。
- `pango-video-capture.service` 正在自动重试；采样时 `NRestarts=5272`。`qt_gpu_probe` 进程存在。
- PCIe 端点配置读回为全 `FF`，`DLLActive` 低。AER 中可见 `Completion Timeout` 和 `RxErr`；这些错误状态具有粘滞性，单独不足以判定根因或责任端。
- 板端日志先报 `slot=0 generation=4 frame=0 length=0 commit=0` 和 `STOP: Input/output error`，随后反复报 `/dev/pango_video0: No such device`。

## 模型切换与 PCIe 断链

模型切换器实现位于 `deployment/lubancat/qt_gpu_probe/model_profiles.py`、`ai_bridge.py` 和 `video_item.cpp`。切换逻辑负责请求/替换 detector，没有直接操作 PCIe 驱动、设备节点或 Root Port。

本次 A → B → A 操作与故障在时间上相隔约十来分钟，存在时间关联，但当前证据不能证明模型切换导致 PCIe 断链，也不能排除推理负载等间接诱因。模型切换功能出现前，10 月 20 日和 24 日的记录中也已有 PCIe 断链。

因此应分开判断：

- **已确认：** 界面、方案选项和板端切换应答此前工作过；目前采集链路断开，完整模型检测链路不可用。
- **未确认：** FPGA 错误码 `1101` 的定义，以及 PCIe 断链的最终根因。现有协议资料没有找到该错误码的链路错误定义。
- **因果判断：** 没有证据把断链归因于模型切换器；目前只能说时间相关，不能当作因果结论。

## 下一步

1. 先恢复并验证 PCIe 端点、`/dev/pango_video0` 和持续采集状态；确认 status 中帧数与 FPS 持续增长。
2. 确认 AI sidecar 已连接，再检查实际活动模型 ID 和新检测结果；不能只依据 UI 的 `APPLIED` 标签。
3. 在相同采集条件下分别观察不切模型和切换模型时的 PCIe 链路、采集错误与系统资源，收集复现数据后再判断是否存在负载关联。

本 checkpoint 仅记录状态，没有重启板卡、停止采集服务或改动模型/驱动配置。
