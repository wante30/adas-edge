# ADAS RK3566 危险感知与控制后续版

本目录在 PR #6 的 `d397af411d592447197fa2f194144dca52c765dd` 上继续开发。独立分支为 `feature/adas-risk-control-20261102`，独立板端目录为 `/home/cat/adas_risk_dev_20261102`。PR #6、下午版归档及正式自启动环境保留。

当前板端没有 PCIe 设备。本地录制回放使用进程内 **DRY_RUN**：制动请求、模拟 ACK 和点刹时序写入日志；不会向控制端口发送命令。LVT1 界面仍反映真实设备可用状态，DRY_RUN 的虚拟车辆状态仅在控制日志中明确标注。

## 运行

运行包位于仓库 `releases/adas_risk_control_20261102/`；先在同一 RK3566 的空目录解压 ZIP。SHA-256 和逐文件清单与 ZIP 同目录提供。

在 RK3566 原有 Qt5 / Mali / RKNN / OCR / 模型环境中运行：

```sh
sh tools/build.sh
sh tools/start_local_replay.sh /home/cat/qt_gpu_probe_ai_20261019/local_quad_full.rgb
sh tools/restore_afternoon.sh
```

回放使用完整 1350 帧、1280×720 RGB565 录制，循环请求 60 FPS。源帧率、Qt 场景图帧率、实际 AI 完成率分别计数。模型菜单保留 B、B-capi、MTL-FULL、MTL-FAST，并新增可直接选择的 FAST-MOSAIC。四路使用一次完整帧 BDD 推理；切回道路模式恢复上一个道路模型。切换通过模式/epoch/模型握手，旧结果不能覆盖新模式。

硬件恢复后，可使用原有适配器和 `sh tools/start_pcie.sh`。本次没有验证真实 PCIe 恢复、真实制动执行或 LVT1 在线链路。

## 事件与控制

[统一事件协议](RISK_PROTOCOL.md) 定义 ACTIVE / CLEARED 生命周期和 WARNING / CRITICAL 严重程度。普通目标保持类别颜色，WARNING 为黄色，CRITICAL 为红色。视频框、面板、Tesla 和 HUD 消费同一份事件。没有匹配到目标的事件留在面板及 `unmatched_hazard` 日志，不给其他目标上色。

[危险规则](RISK_RULES.md) 覆盖前车接近、行人、两轮车、切入、方向已确认的对向冲突、制动灯变化与接近，以及 MTL 斑马线附近行人。几何路径为明确的图像空间假设，当前没有标定距离或方向测量。不能只凭框变大认定对向来车。

[控制配置](CONTROL_MODES.md) 解释 CRITICAL 候选的检查、非阻塞 ACK、虚拟车辆状态及独立点刹测试。WARNING、CLEARED、预测目标、其他象限和过期事件都不能产生新的制动请求。

## 跟踪与显示

复用恢复源码中的 Windows ByteTrack-Lite，按检测时间戳进行摄像头内两阶段关联，类别族隔离，漏检保留 600 ms。Qt 使用同一组主跟踪 ID。Tesla 仅消费 Q1；在 Qt 原生渲染中预测最多 250 ms，平滑纠正，淡入约 180 ms，淡出最长 350 ms，600 ms 后删除。预测和渐隐对象不作为控制证据。

模型、模式或 epoch 变化会清理旧身份与语义。MTL 掩码、导航 OCR、LVT1、HUD 接口和原生 BEV 保留；没有加入 Python/OpenCV 图像渲染或额外模型推理。

## 性能与验证

快速危险规则按每次有效完成执行；MTL 斑马线和少量车辆制动灯裁剪使用最新单槽后台任务，最多 2 Hz。性能统计每秒汇总一次。调度只保留一个未完成请求，始终使用最新帧；默认 PRESSURE 策略按 GUI 忙碌比例和渲染帧率选择最多 8 ms 退让，`QT_AI_PACING_POLICY=LEGACY` 可对照旧 30 ms 策略。

```sh
python3 -m unittest discover -s tests -v
python3 tools/validate_scenarios.py
python3 tools/test_session.py start --build build-stage4 --label final_replay --ai-hz 0 --control-mode DRY_RUN --external-qml --raw-replay /home/cat/qt_gpu_probe_ai_20261019/local_quad_full.rgb
python3 tools/validate_followup.py --label final_replay --seconds 610 --switch-every 60
```

`validate_scenarios.py` 是明确标记的合成数值序列，不是实拍检测精度证明。`validate_synthetic_ui.py` 使用独立合成 AI 协议夹具在真实 Qt 界面验证黄框、红框、行人、点刹 ACK 与解除，并在屏幕显示合成测试标记。夹具仅使用 AI 通信协议，控制仍完全在进程内。

完整报告见 `validation/followup/REPORT.md`。验收材料位于 `validation/followup/`；PR #6 的历史验收保留在 `validation/`。最终报告区分实际回放、合成测试及没有硬件无法验证的事项。

## 恢复与打包

运行包包含不含 PID 的 `config/baseline_launch.json`，当前板无 PCIe、原 HMI 未启动时仍可获取正确回放参数。启动器记录当前 boot ID、基线参数、原自启动父进程和本测试进程，只停止本测试拥有的会话。跨启动的旧 PID 记录不能用于发信号。运行包不分发活动会话 PID 文件。

`tools/package_followup.py` 在板端打包源码、原生 HMI、独立 FAST-MOSAIC C API、配置、测试及验收材料，并生成 SHA-256 清单。模型、OCR 资产和系统 RKNN/Mali/Qt 运行库使用原有外部安装；清单记录依赖，不重新安装或修改。
