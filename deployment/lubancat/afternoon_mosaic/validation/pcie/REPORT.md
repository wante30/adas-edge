# Q4 真实 PCIe 比赛链路测试（2026-11-02）

本轮使用真实 PCIe 采集输入、真实 BAR1 控制命令与真实 LVT1 反馈。功能闭环通过；单路 AI 与 Qt 的性能底线未通过，所以总体验收 `passed=false`。机械执行效果和 PCIe 热恢复没有独立验证。

## 配置与版本

- 基于 PR #7，运行源码 `6bf1093986acdd59b36dba7e9c880a231a206179`；独立硬件分析工具提交 `55084944`。
- RK3566 4 GB，Qt5 / Mali / 原 RKNN Runtime。独立部署 `/home/cat/adas_risk_dev_20261102`。
- 使用 `/dev/shm/pango-video/latest.pgv`；控制 HARDWARE，真实 UDP 38262 → BAR1，未使用模拟端口或 DRY_RUN。
- 用户指定默认单路及比赛前向通道为 **Q4**（零基参数 3）；四路模式也只允许 Q4 的有效 CRITICAL 前向危险下发制动。Q1/Q2/Q3 保持显示和检测，不进入前向制动。
- `ROAD_Q1` 保留为单路协议名，实际来源明确记录 `forward_source_id=Q4`。画面缩略图选择不改变固定的可制动前向通道。
- 原服务与原稳定目录保留。`tools/start_pcie.sh` 在独立目录启动新版并临时接管 HUD 服务；`tools/restore_afternoon.sh` 恢复原服务及原自启动。新版启动脚本默认 Q4；原稳定版自启动仍是保留的历史配置。

## 真实运行结果

610.303 秒，约 60 秒交替单路 / 四路，导航 OCR、危险规则、Tesla、HUD、LVT1 与真实控制均开启。源输入与历史回放不同，不能直接视为同输入回归。

| 指标 | 实测 | 原验收底线 | 结论 |
|---|---:|---:|---|
| FAST-MOSAIC 完成率 | 13.9480 FPS | 13.8 | 通过 |
| MTL-FAST 单路完成率 | 6.0774 FPS | 6.5 | 未通过 |
| Qt 全程 200 ms 指标均值 | 18.2772 FPS | 28 | 未通过 |
| HMI 接收源帧计数 | 41.8081 FPS | 单独统计 | 有采集丢弃 |
| 纹理上传计数 | 17.4585 FPS | 单独统计 | 尚需优化 |
| 实际运行时长 | 610.303 秒 | 600 | 通过 |

完成率按 PR #6 的 bridge / epoch / model 分组，累计 `(n-1)` 除以组内首尾发布时间差。AI 单路 1823 个、四路 4321 个结果。Qt 截图头部“视频”数值来自输入源帧率，不替代 Qt 场景图实测值。

整机 30 秒独立 `/proc/stat` 采样平均 CPU 忙碌约 96.65%；Qt 自身进程约占四核总能力的 22.92%，两者不能混为一谈。温度单次读取约 39°C。该测试存在真实采集 CPU 负载和界面处理瓶颈，尚未达到全部性能要求。

## 自动制动与显示一致性

- 770 条事件转换，117 条 CRITICAL 决策；45 次真实请求、45 次匹配 ACK、45 次 FPGA 接收和动作激活、45 次 LVT1 `brake_allow=true`；超时 0。
- 45 次 ACK 的 LVT1 动作序号均与硬件 sequence 一致。全部请求来源 Q4、`simulated=false`。
- 框／事件身份与严重程度不一致 0，WARNING 请求 0，CLEARED 请求 0，非前向请求 0，缺失 CRITICAL 决策 0。
- 17 条决策拒绝为未再观测期间的 `ineligible_or_invalid_path`，没有发送对应的新指令。
- UI 记录有 9 次事件暂未匹配框：保留在面板，没有染色到其他目标。
- 风险确认 → UI p95 206.93 ms；解除 → 框恢复 p95 215.34 ms。
- 真实输入观测到前车冲突、加塞、行人路径 CRITICAL，及斑马线附近行人 WARNING。未出现的两轮车、对向等场景沿用独立合成验证，不能称为本轮实拍覆盖。
- 305 次采样 LVT1 在线且传感器不陈旧；305 次采集状态 running，采集帧持续增加。
- 导航 VALID 5237 / UNREADABLE 791 / STALE 116，OCR 持续工作。

真实命令探针也确认 FPGA 接收、动作激活、允许制动和约 1001 ms 后动作结束。以上证明 FPGA 控制链路，`physical_actuator_verified=false`，不声称已测得车辆机械制动效果。

## 构建与回归

本地及板端 49 项 Python 测试通过；原生 Q4 事件关联、ARM CRC 与 zlib 在不同长度及非对齐地址一致性、LVT1 原始计数变化不重绘／制动变化立即更新／状态过期转离线测试均通过。186 个部署源码与本地规范化哈希全部匹配。实际 7 次模型菜单选择和 AI 停止／重启恢复通过。长测运行错误、重复完成、语义错误和 CRC 错误均为 0。

实时路径调整包括 8 个预分配帧槽、合并重复 GUI 唤醒、保留逐帧 CRC 的 ARM 指令加速、模型菜单关闭时停止创建条目、LVT1 可见状态变化通知与原始快照通知分开。OCR 使用绝对 nice 10，保留原 RKNN 推理线程调度。CPU/GPU performance governor 的短时对照帮助有限，测试后已恢复原 governor；没有修改驱动、FPGA RTL 或 RKNN Runtime。

## PCIe 断线及恢复

首轮用户报告 PCIe 断开时，实测采集 `capture_io`、LVT1 标识失效、PCIe revision=ff。普通重启请求超时，按用户指令执行强制板端重启；boot ID 更新后，设备重新枚举、视频恢复。该事实只能证明重启恢复，热恢复仍未解决。新桥在标识失效后停止继续访问 BAR1，等待链路恢复并重启桥。

原始 Q1 实测帧 9525 的黑色像素比例为 1.0，用户因此指定改用 Q4。该现象源于 PCIe 原始输入，并非 Qt 裁剪黑屏。

## 运行包

独立运行包位于 `releases/adas_risk_control_q4_pcie_20261102/adas-risk-control-q4-pcie-20261102.zip`，29,350,021 bytes，247 个文件。SHA-256 为 `dbebb09d1190cce3aebb784d48f19096997efb9572076be72c2917094efa84a4`。本地逐文件哈希验证通过；未额外从新解压目录启动该包，因此不将源码部署实测冒充为包启动实测。

## 证据

- [最终机器可读结论](FINAL_ACCEPTANCE.json)：功能与性能分别判定，完整指标。
- [真实日志及截图归档](actual_pcie_q4_logs.zip)：43 个文件，内含逐文件 SHA-256 清单。该归档只对应真实硬件运行。
- `competition_q4_final_model_menu_acceptance.json`：实际模型选择。
- `competition_q4_final_restart_acceptance.json`：AI 重启恢复。
- `competition_pcie_v2_manual_control.json`：真实命令、动作与结束反馈。
- `pcie_q4_source_sync.json`：186 文件部署校验。
- `interrupted_run.json`：断链与重启证据摘要。

![Q4 四路实测](q4_mosaic_final.png)

![Q4 单路实测](q4_road_final.png)

上一轮无 PCIe 的 1350 帧回放结论保留在 `../followup/`，不与本轮混用。交付后板端运行 `competition_q4_delivered`，默认 Q4 单路、MTL-FAST、HARDWARE。
