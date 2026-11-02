# RK3566 真实 PCIe 性能版运行说明

基线为 `a9054ec21fb4a97637f06bbf0ece1d36f0f08c2e`。默认模型仍是 MTL-FAST，物理前向摄像头 Q4。FAST-Q4 是新增的 BDD 单路检测选项，不提供 MTL 道路分割。

## 安装与启动

运行包适用于当前 RK3566 aarch64 板端，包含 Qt 可执行文件、通用 BDD C API 动态库、源码、工具及验证摘要。继续使用板上已有的 Qt5/Mali、RKNN Runtime、Python/OpenCV、OCR 环境及模型文件；没有重新训练、量化或分发模型权重。

解压到一个新目录，并给可执行文件添加权限：

```sh
unzip adas-performance-rk3566.zip -d /home/cat/adas_perf_release
cd /home/cat/adas_perf_release
chmod +x build-stage4/qt_gpu_probe
python3 tools/verify_source_hashes.py
sh tools/start_pcie.sh --label release
```

若当前已经由测试会话管理 HMI/AI，接管同一启动周期的会话：

```sh
sh tools/start_pcie.sh --label release --adopt-session /home/cat/adas_perf_20261102/src/evidence/integration_session.json
```

接管工具检查 boot_id 和进程命令行。不要跨重启复用旧 PID。已有正式自动启动的环境也可直接运行启动脚本，由会话工具保存原来的启动参数。

启动脚本保持真实 PCIe 与 HARDWARE 遥测模式，默认 `ADAS_CONTROL_AUTHORIZED=0`。危险状态机、控制决策日志、LVT1 和 HUD 继续工作，但不发送自动制动请求。本轮交付没有进行机械制动性能测试。

## 模型与功能

| 模型 | 物理输入 | 功能 |
|---|---|---|
| MTL-FAST（默认） | Q4 | 检测、道路/车道/斑马线分割 |
| FAST-Q4 | Q4 的 640×360 packed RGB565 裁剪 | BDD 检测、跟踪、基于目标/轨迹的危险规则；分割明确不可用 |
| FAST-MOSAIC | 1280×720 四路拼图 | 四路 BDD 检测；前向控制判断仍仅 Q4 |

使用右上角模型菜单切换。等待 APPLIED 和新结果，不以菜单高亮代替实际模型运行确认。原有 B、B-capi、MTL-FULL 及其他模型选项保留。

AI OpenCV 保留 4 线程；OCR OpenCV 和 ORT 使用 1 线程，识别距离 ROI 最小宽度 128。CPU/GPU/NPU/DRAM governor 沿用板端原值，未固化绑核或优先级调参。

## 复现实机验证

在完整真实 PCIe 会话中运行，禁止并行编译或独立推理基准：

```sh
ADAS_VALIDATION_LABEL=release python3 tools/validate_model_menu.py
python3 tools/validate_restart.py
sudo sysctl -w kernel.sched_schedstats=1
python3 tools/validate_performance.py --label release --seconds 610 --window 100
python3 tools/summarize_acceptance.py evidence/release_system.jsonl evidence/release_performance_acceptance.json --output evidence/release_summary.json
sudo sysctl -w kernel.sched_schedstats=0
```

完整窗口包含切换；稳态窗口剔除每段开始 10 秒及结束 2 秒。AI FPS 由同 bridge/epoch 内真实结果完成时间计算。界面顶部的视频 FPS 是接收源帧率，不能当作 Qt Scene Graph FPS。

`validate_performance.py` 操作可见菜单、截图、检查事件与框身份及实际控制请求数。若遇到异常会中止并保存失败原因，不能把中止的窗口算作 10 分钟验收。

## 恢复原版

当前板上完整原版目录保留在 `/home/cat/adas_risk_dev_20261102`，原版可执行文件额外保存在 `/home/cat/adas_perf_20261102/baseline/qt_gpu_probe-a9054ec2`。

```sh
python3 tools/restore_performance.py
```

恢复脚本使用原版源码、可执行文件及 Q4 前向配置，保持自动制动发送关闭。与之独立，`test_session.py restore` 恢复会话接管前的桌面启动方式；对于此次性能版回退，优先使用上述专用恢复脚本。

## 已知边界

性能目标和最终实测见 `validation/performance/REPORT.md`。FAST-Q4 的整机结果不能用无 Qt/OCR 的独立 C API 峰值代替。OCR 的 24 帧人工标注对照仅覆盖本次真实 Q2 样本，不能推断通用准确率。

开发期间 PCIe 两次出现 revision ff、capture_io 与 LVT1 掉线，其中一次发生在约一小时运行后；重启恢复。没有修改驱动/RTL，也没有声称已经修复热恢复。最终 610 秒结果仅证明对应窗口的稳定性，不证明长时间链路问题消失。
