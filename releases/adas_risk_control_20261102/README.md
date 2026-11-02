# RK3566 ADAS 危险感知回放版（2026-11-02）

基线：PR #6 / `d397af411d592447197fa2f194144dca52c765dd`。源码验收点：`be56effd`。运行包为 `adas_risk_control_rk3566_20261102.zip`，27,757,852 bytes。

SHA-256：`c01bbd0ec616ea44e0a0bbf16f6868c08cbcc424662d4bf1cecf3d10647b49c7`。

`*.manifest.json` 与 ZIP 内 `RUNTIME_MANIFEST.json` 记录 230 个文件的大小、SHA-256、架构和外部依赖。ZIP 不含活动会话 PID、巨型 RGB565 录制或 RKNN 模型；模型、OCR、原 C API 和系统 Qt/Mali/RKNN 使用当前板原有安装。

同板若有正在运行的开发回放，先用该会话自己的 `tools/restore_afternoon.sh` 结束，再从空目录解压。当前没有 PCIe 时，可用不含 PID 的 `config/baseline_launch.json` 启动。

```sh
sha256sum -c adas_risk_control_rk3566_20261102.zip.sha256
unzip adas_risk_control_rk3566_20261102.zip -d adas-risk-control
cd adas-risk-control
sh tools/start_local_replay.sh /home/cat/qt_gpu_probe_ai_20261019/local_quad_full.rgb
sh tools/restore_afternoon.sh
```

启动器显式选择进程内 DRY_RUN；制动不会创建或使用模拟控制端口，HUD 在 DRY_RUN 不发送硬件指令。AI 图像与模型协议的本机通信继续使用 38260/38261。

包已在 RK3566 的新目录解压，逐项验证 230 个文件 SHA，确认实际执行包内的 HMI，可在原 HMI 未启动、无 PCIe 的条件下完成至少 30 次真实模型结果。独立证明见 `deployment/lubancat/afternoon_mosaic/validation/followup/runtime_package_smoke.json`。

610.672 秒长测：FAST-MOSAIC 14.0374 FPS，MTL-FAST 6.6475 FPS，Qt 全程 28.0948 FPS。完整 24 项对照、实际与合成证据和未验证硬件项目，见 `deployment/lubancat/afternoon_mosaic/validation/followup/REPORT.md`。

真实硬件重连与制动尚未验证。硬件恢复后 `tools/start_pcie.sh` 使用 HARDWARE 配置；旧桥需要能回传 request_id 才可严格匹配 ACK。
