# ADAS 前视优先落地路线

更新时间：2026-10-01

## 目标

本项目面向紫光同创 FPGA 创新设计赛道选题三，先把前视道路感知和车载大屏做成可在 PC、再在 RK3566 上复现的闭环，再扩展后、左、右三路。主画面应是前视道路驱动的 ADAS 3D 驾驶场景，红绿灯、车辆、行人、车道、路径和告警必须来自同一帧链路。

赛题要求的硬件边界保持不变：FPGA 负责视频采集、同步、缓存及预处理，PCIe 负责帧/元数据传输，RK3566 负责轻量 AI 和场景状态，显示侧负责最终大屏/HDMI 输出。PC 模拟只能验证软件协议和画面，不能代替板端证据。

## 阶段与验收门

| 阶段 | 交付目标 | 验收证据 | 当前状态 |
|---|---|---|---|
| 0. 赛题与边界 | 固定选题三、前视主线、四路后置、FPGA/RK 分工 | 官方 PDF 17 页核对、需求矩阵 | 已完成 |
| 1. PC 前视感知 | 真实前视视频驱动 YOLO、跟踪、红绿灯检测/颜色状态和基础车道；记录帧率、延迟、空帧率 | 原始视频、JSONL/CSV 指标、带框视频、可复现命令 | 进行中：目标/候选灯态和 CPU 车道候选已跑通，正式准确率和真实车道/灯归属未通过 |
| 2. PC 车机大屏 | 前视视频为主画面，SceneFrame 同时承载目标、灯态、车道、路径、告警；3D 画面不依赖 mock 数据 | 浏览器运行、截图/录屏、WebSocket 指标、3D 场景检查 | 进行中 |
| 3. RK3566 前视 | 同一前视片段在真实 RK3566 上运行；主检测器和红绿灯专项支路分别测 NPU、端到端、温度、丢帧 | 板端命令、模型哈希、完整日志、结果视频/CSV | 进行中：主模型和低频车道支路已实测，专项信号归属仍未通过 |
| 4. FPGA 单路闭环 | FPGA→PCIe→RK3566→PCIe/OSD 单路帧号闭环，先冻结帧再连续视频 | endpoint/DMA/OSD 日志、frame_id 对齐、超时 fail-safe | 依赖硬件链路 |
| 5. 前视质量冻结 | 在统一道路集上冻结准确率/速度/稳定性和 3D 视觉版本 | precision/recall/F1 或明确的定性边界、P50/P95、长测、渲染录屏 | 未完成 |
| 6. 四路扩展 | 后/左/右接入，同步、调度、融合和环视作为后续能力，不牺牲前视新鲜度 | 四路连续实测、带宽/帧率/延迟/丢帧、AVM 标定证据 | 后置 |
| 7. 比赛交付 | 覆盖红灯停车、绿灯通行、强制通行报警、变道、行人危险和异常断流 | 实物、1080P 三分钟视频、源码和技术报告 | 未完成 |

## 关键工程决策

1. Tesla-BevCam-Clone 作为真实前视感知参考、PC 验证入口和可切换的开源 BEV 对比渲染器；它的 BEV 代码不直接等于 RK3566 实时方案。项目自己的 `SceneFrame` 是 PC、RK3566、FPGA OSD 之间的稳定边界。
2. 主 YOLO 负责车辆、行人、两轮车等通用目标；红绿灯必须保留 FRONT-only 专项支路或等价的高分辨率 ROI 支路，并通过时间滤波输出 `RED/YELLOW/GREEN/UNKNOWN`。仅有 COCO 的 `traffic light` 框不算完成红绿灯判断。
3. 所有距离先标记为 `PSEUDO_DISTANCE`，没有相机标定、深度或雷达证据时不能写成真实测距。3D 画面精致度和感知真实性分开验收。
4. PC 的 FPS、RK3566 的 FPS、FPGA/PCIe 的端到端延迟分别记录，不能互相替代。任何旧报告只作为历史参考，必须注明模型、画布、输入和运行日期。

## 当前真实基线

- 官方赛题要求至少四路视频、图像拼接/融合模拟全景、实时低延迟识别、红绿灯行为、变道和行人危险场景，并要求实物、1080P 三分钟视频和源码。
- 当前 PC 已用真实前视 MP4 运行 Tesla-BevCam-Clone；YOLO 可检出 `traffic light`，但上游 BEV 版本曾把该类别过滤掉，红黄绿状态尚未接入该链路。
- 当前工程已有 `SceneFrame`、交通灯 HSV 分类器、RKNN 解码器、PCIe 软件 descriptor 和 HMI，但真实前视视频到 SceneFrame、板端专项支路和 FPGA 实链仍需闭环。
- RK3566 实机已通过 USB RNDIS 确认：`<board-ip>`，`EmbedFire LubanCat-1`，设备树为 `rockchip,rk3566`。板上已有矩形 YOLO11n RKNN 模型和交通灯 specialist 模型，可用于本轮前视实测；此事实不代表 PCIe/FPGA 已连通。
- PC 真实路口片段（YOLOv8n、512×288、169 帧、原视频 30 FPS）最新语义版在 `outputs/front_pc_signal_semantics_20261001/`：检测器检测到 74 个 traffic-light 框；车辆形状信号候选输出 `GREEN 18 / UNKNOWN 151`，垂直行人信号单独记录为 `RED 8 / GREEN 6`。这避免把行人红灯冒充本车红灯，但仍不能替代本车道归属标注。
- PC 已接入可选 FRONT-only specialist（`D:/Datasets/adas_edge_training/tl_specialist/tl-512x192/weights/best.pt`），评测命令写入 `tools/benchmark_front_pipeline.py`，按冻结的顶部 60% ROI、512×192、3 FPS 协议运行。补充人工正负样本审核后，specialist-only/旧双检结果仍只作为对照；本轮高分辨率信号支路（顶部 55%、横向 10%–90%、640×384、ROI 阈值 0.05、车辆 hold12、行人 hold4）在同一窄片段上车辆绿灯为 precision `1.0000`、recall `0.8293`、F1 `0.9067`（34/41），行人红灯为 precision `1.0000`、recall `0.7692`、F1 `0.8696`（10/13）。169 帧墙钟约 `7.72 FPS`、感知均值约 `99.27 ms`；因此它先作为评测/展示配置，不把速度与召回的折中误写成最终方案。这些指标不是数据集级准确率，也不包含本车道归属。
- RK3566 最新语义版在 `outputs/rk3566_front_eval_20261001_signal_semantics/`：135 个 traffic-light 框；车辆形状候选 `GREEN 38 / UNKNOWN 262`，行人信号旁路 `RED 5 / GREEN 1`。主模型仍保持空帧率 1.67%；NPU mean 39.90 ms、P50 39.20 ms、P95 45.47 ms；车道每 3 次检测更新一次后，感知端到端 mean 72.43 ms、P95 123.80 ms，墙钟 6.55 FPS。
- RK3566 已用板上专用模型完成同一基准脚本的 300 帧复现，最新产物在 `outputs/rk3566_front_specialist_fuse_20261001_pedhold4/`：主模型 NPU mean 40.04 ms/P95 46.37 ms，specialist 3 FPS 支路 mean 32.52 ms，整体墙钟 6.37 FPS、端到端 mean 76.05 ms/P95 127.31 ms；采用“主模型候选 + specialist 候选”融合后，车辆形状候选为 `GREEN 50 / UNKNOWN 250`，行人旁路为 `RED 8 / GREEN 22`。融合比 specialist-only 的车辆 `GREEN 17` 更完整，但该片段没有逐帧人工灯态标注，故这些是候选运行统计，不是板端准确率。
- RK3566 长时循环复测已完成 600 帧，产物在 `outputs/rk3566_front_stability_loop_20261001_600/`：视频循环 1 次、空帧率 0、墙钟 6.397 FPS、主 NPU mean 39.94 ms/P95 45.97 ms、端到端 mean 75.96 ms/P95 128.01 ms；车道任一侧覆盖 100%、双侧 89%。进程在整段循环中保持运行，但这仍不是温度/内存/多小时稳定性验收，也没有板端逐帧准确率标签。
- 接入实验性车道支路后的 PC 结果（`outputs/front_pc_benchmark_signal_lane_20261001_320/`）曾为端到端 mean 44.90 ms、墙钟 15.12 FPS；语义版最新结果（`outputs/front_pc_signal_semantics_20261001/`）为 mean 47.33 ms、P95 46.32 ms、墙钟 14.37 FPS。双侧车道覆盖率约 89.9%（PC）和 88%（RK3566）。这些是显示候选覆盖率，不是正式车道精度。
- 本轮板端复现哈希：主模型 SHA-256 `7a697ccc476954f6f36f7441b842257a2d7a019653cc4bad2fbf8de327c50591`；specialist SHA-256 `becfaada5da49c4dd6c143e815f8f957fb7b7ed759040d4e18c685a9c434e6f8`；板端目标为 `embedfire,lubancat-1` / `rockchip,rk3566`、aarch64、Python 3.8.10、RKNN Lite 2.3.2。
- 交通灯 specialist 首轮同片段产生 636 个候选、186 帧有候选，抽查发现禁停牌误检且真实行人灯有漏检。因此当前不能声称“红绿灯稳定识别”，下一门必须使用带本车信号归属的专项模型/规则和标注片段重测。新的 specialist 适配器已支持 PC `.pt` 和 RK3566 `.rknn`，但 `decision_state` 仍保持 `UNKNOWN`，直到本车道归属有标注证据。
- PC 真实前视链路现在同时输出 `lanes.left/right.points`，由 CPU OpenCV Hough/颜色候选产生；在 45 个均匀抽样帧上覆盖率为任一侧 100%、双侧 86.7%。这是显示用候选，尚未做相机标定、曲线拟合或车道语义验证。
- 本轮把车道候选短时保持从 3 次漏检放宽到 6 次，PC 窄片段双侧覆盖从约 89.9% 提升到 97.6%，RK3566 300 帧复测达到双侧 100%；这只改善显示/关联输入的连续性，不等于车道检测准确率。
- `geometry-experimental` 关联对照产物在 `outputs/signal_audit_20261001/geometry_assoc_lanehold_wide_decision.json` 和 `outputs/rk3566_front_geometry_assoc_20261001_300/`。PC 关联决策在人工审核片段上 precision `1.0000`、recall `0.6341`（26/41）；RK3566 300 帧墙钟 `6.41 FPS`、NPU mean `39.99 ms`、端到端 mean `76.27 ms`，关联状态为 `GEOMETRIC_EXPERIMENTAL` 31 帧。该模式仍是未标定几何实验，默认 HMI 不开启。

## 本轮执行顺序

1. 将真实前视源接入项目 `LiveSource`，补齐 traffic-light ROI 颜色状态、时域滤波、指标和 SceneFrame metadata。（已完成软件接线；`decision_state` 仍保守为 `UNKNOWN`。）
2. 用同一段含红绿灯的前视视频完成 PC 端带框/状态/延迟产物，并用浏览器主界面检查前视 3D 场景。
3. 将可复现的输入片段、主模型和 specialist 模型送到 RK3566，运行同一指标协议，记录模型哈希和板端资源。
4. 只有 PC 与 RK3566 前视门通过后，才进入 Apollo/Three.js 渲染层评估、FPGA 单路 PCIe/OSD 和三路扩展。

