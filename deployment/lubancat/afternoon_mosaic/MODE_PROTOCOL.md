# 推理模式协议

基线提交：`9b396e8b346bfbbfa9856950937e1cd3f54516f4`。物理输入始终为 1280×720 RGB565 四宫格。

## 模式与坐标

| 推理模式 | 检测输入 | 返回坐标 | 道路安全语义 |
|---|---|---|---|
| ROAD_Q1 | Q1，640×360 | ROAD_VIEW_1280x720，兼容原道路主视的 2 倍坐标 | Q1 |
| MOSAIC_FULL | 完整 1280×720，一次推理 | MOSAIC_1280x720，全局坐标 | 只接受完整落在 Q1 内的目标；跨边界目标保留显示 |

旧 MVS1 象限包继续选择显示/导航象限，不改变推理模式。道路推理固定 Q1；选择其他道路预览象限时，隐藏 Q1 检测框和分割，避免叠到其他来源。

## UDP 数据包

沿用请求端口 38260 和结果端口 38261。所有整数使用小端编码。

- MVI1：4 字节 magic、uint64 epoch、uint32 mode（0=ROAD_Q1，1=MOSAIC_FULL），共 16 字节。
- MIA1：4 字节 magic 加 JSON，包含实际 mode、epoch 字符串、box_space、bridge_id、state、request_epoch。
- MVR3：4 字节 magic、uint64 frame_id、uint64 capture_timestamp_ns、uint64 epoch、float input_fps、float submitted_fps，共 36 字节。
- MVA2：保留原二进制头和目标框数组，在 JSON 中增加 inference 对象，包含 mode、epoch、box_space、bridge_id、frame_id。

HMI 在启动和用户切换时发送目标模式。尚未确认时每 250 ms 重发，快速重试最多 8 次，此后每 2 秒尝试重连；已确认时每秒发送幂等保活，以便 AI 重启后同步。

AI 监听线程只排队模式命令，推理线程在原生推理调用结束后应用切换。旧 epoch 不能覆盖新 epoch，同一模式/epoch 的重复命令直接确认，不重复加载模型。

HMI 仅显示与当前目标模式、epoch、坐标空间、AI 实例和已提交 frame_id/时间戳全部匹配的结果。模式变化时清空旧检测框、跟踪状态和不适用分割。迟到结果不参与当前模式的显示和 FPS。

## 制动输出

原有 AutoBrakeController 判定条件与正式默认端口 38262 保持不变。测试会话默认可将输出转到模拟端口 38362。2026-11-01 用户明确要求恢复正常指令后，板端测试通过 `--action-port 38262` 使用原 HUD Bridge，并回读到 ok:true、last_error:0。

## 阶段与恢复

阶段一可恢复提交：`b2e4cc1c`；实测 1719 个唯一结果，223.65 秒，7.68 FPS。截图见 validation/stage1.png。

独立运行目录：`/home/cat/adas_mosaic_dev_20261101`。正式归档及正式 HMI 二进制没有被覆盖。test_session.py 保存启动参数、显示环境、服务状态与 boot_id，拒绝向上一次开机遗留 PID 发信号。恢复原启动会话使用该工具的 restore 子命令。
