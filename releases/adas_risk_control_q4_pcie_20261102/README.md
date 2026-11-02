# RK3566 Q4 真实 PCIe 比赛链路版（2026-11-02）

源码：`5508494417c825ecacfcbc3c6b6250d41ef0b60e`，基于 PR #6 / PR #7。ZIP 为 29,350,021 bytes，共 247 个文件，带逐文件 SHA-256 清单。旧回放包独立保留。

SHA-256：`dbebb09d1190cce3aebb784d48f19096997efb9572076be72c2917094efa84a4`。

使用原 RK3566 Qt5 / Mali / RKNN / Python / OCR / 模型环境。包不含 RKNN 模型或 1350 帧录制。先在正在运行的旧开发目录执行 `sh tools/restore_afternoon.sh`，再解压到新空目录启动：

```sh
unzip adas-risk-control-q4-pcie-20261102.zip -d adas-q4-pcie
cd adas-q4-pcie
sh tools/start_pcie.sh
sh tools/restore_afternoon.sh
```

`start_pcie.sh` 默认 **Q4 单路 / MTL-FAST / HARDWARE**。控制桥真实映射 BAR1，回传 request_id、实际 accepted word、FPGA active 与 LVT1。会话临时接管原 HUD 服务，恢复脚本恢复原服务及原 HMI。原稳定版自启动文件保留；该包需要通过新启动脚本使用。

历史回放仍可 `sh tools/start_local_replay.sh /home/cat/qt_gpu_probe_ai_20261019/local_quad_full.rgb`，显式 DRY_RUN、默认历史 Q1，避免混用真实硬件与回放结果。

真实 PCIe 610.303 秒：45/45 自动制动请求均被 FPGA 接收并激活动作，LVT1 允许制动，0 超时，0 非 Q4 请求，0 身份/颜色不一致，0 CRC 错误。FAST-MOSAIC 13.9480 FPS，MTL-FAST 6.0774 FPS，Qt 18.2772 FPS。单路及 Qt 未达到原验收底线，总验收未通过；机械制动效果及 PCIe 热恢复未验证。

本地和板端各 49 项 Python 测试、3 项原生测试、实际模型菜单和 AI 重启恢复通过。包内全部文件哈希已验证，但本轮未从新解压包另外启动实测。完整报告和独立硬件日志见仓库 `deployment/lubancat/afternoon_mosaic/validation/pcie/REPORT.md`。
