# 2026 FPGA 创新设计赛道选题三：比赛项目交接总览

更新日期：2026-09-08

本文件只记录与“以紫光同创 FPGA 为核心的道路环境感知、辅助驾驶与高速通信系统”赛题直接相关的工作。MOT17 研究、论文实验和其他仓库改动不作为本项目完成度证据。

## 1. 当前判断

软件侧已经有一条可以演示和继续验收的链路：视频帧可以进入 `CompetitionDetector`，由 RK3566 上的 RKNN 模型输出检测框，再由场景适配和规则引擎产生停车、通行、变道或报警元数据；帧号、流号和时间戳可随结果返回。

但比赛系统尚未闭环。最重要的三项缺口是：真实 FPGA 视频采集/OSD/PCIe endpoint、合法且有代表性的道路视频、以及在道路数据上验证并可能微调的检测模型。因此当前状态应称为 **软件与单板模型原型完成，整机比赛验收未完成**，不能称作整套赛题已经完成。

## 2. 已具备的比赛能力

| 子系统 | 已具备内容 | 证据 | 状态 |
|---|---|---|---|
| 检测模型 | YOLO11n COCO 预训练权重已导出为 RK3566 INT8 split-output 320/416 两个候选 | `deployment/rknn/`；模型构建记录 | 已完成 |
| RK3566 推理 | 两个候选模型均已在真实 RK3566 上完成 load、inference、decode 的单图 smoke | 320 约 24.14 ms；416 约 38.35 ms | 已完成，非视频验收 |
| Detector API | `CompetitionDetector` 支持 320/416 letterbox、RKNN 推理、split/raw 输出解码、NMS、坐标还原及 stream/frame/timestamp 透传 | `competition_ai/`、单元测试 | 已完成 |
| 场景与决策 | 交通灯 HSV 状态适配、行人/障碍危险区、红黄绿灯、虚实线变道的确定性规则已实现 | `competition_ai/traffic_light.py`、`competition_ai/adas.py` | 软件逻辑完成 |
| PCIe 软件契约 | 定义了 FPGA→RK3566 帧描述符、RK3566→FPGA 元数据描述符，并有 mock 往返、异常包与超时语义测试 | `competition_io/`、`docs/competition_pcie_protocol.md` | 软件 mock 完成 |
| 多路预算 | 根据已有 NPU 服务率，四路轮询的保守起点为每路 6 FPS、总计 24 FPS | `docs/competition_multistream_budget.md` | 规划值，未端到端实测 |
| 外部数据验证 | 已下载 575 MB BDD100K validation-only 图像镜像，利用逐框标注对 1,000 张图像完成 320/416 PC FP32 对比 | `docs/competition_bdd_external_validation.md` | 已完成，图像级 |
| 本地视频预览 | 用户夜间道路视频、密集交通视频已生成 PC FP32 带框结果，便于人工观察 | `outputs/user_video_preacceptance/`、`outputs/local_preacceptance/` | 定性预验收 |
| 软件质量 | CompetitionDetector/PCIe/场景测试已通过；最近完整结果为 `62 passed` | `tests/test_competition_*.py` | 已完成 |

### 模型候选及速度

| 候选 | RKNN 文件 | SHA-256 | RK3566 单图粗测 | 当前定位 |
|---|---|---|---:|---|
| 320 INT8 | `yolo11n_rk3566_int8_split_320.rknn` | `60F17E9BFE20B27DC7D0BE068BC9852024E4A941847F881E1890BF96B2C977DA` | 24.14 ms | 默认实时候选 |
| 416 INT8 | `yolo11n_rk3566_int8_split_416.rknn` | `87DFD6D480760D105E0ABD7D04829D5059C8C5721BD7E3D6C53784DE9793BB43` | 38.35 ms | 小目标质量候选 |

两个模型都来自同一个未经 fine-tune 的 `yolo11n.pt` COCO checkpoint，RKNN Toolkit2 2.3.2，RK3566 INT8 校准。它们的单图 smoke 证明部署链路可运行，**不能**证明道路准确率或四路系统实时性。

### 当前检测效果：已知事实

BDD100K 外部图像预验收使用固定的 1,000 张验证图像、IoU >= 0.5、置信度 0.25，PC RTX 4060 上运行 FP32 `yolo11n.pt`。它不是 RK3566 INT8 测试，也不是完整 BDD100K 官方排行榜结果。

| 类别 | 320 recall | 416 recall | 结论 |
|---|---:|---:|---|
| car | 23.0% | 29.6% | 416 更好 |
| person | 9.4% | 16.1% | 小行人仍是薄弱点 |
| truck | 14.8% | 19.9% | 416 更好 |
| bus | 16.6% | 20.4% | 416 更好 |
| traffic light | 1.0% | 3.3% | 两者都不足，必须重点处理 |
| motorcycle | 3.7% | 9.3% | 两者都不足 |
| bicycle | 1.1% | 5.3% | 两者都不足 |

结论：416 的确提升了小目标召回，但绝对水平仍不足以宣称比赛道路感知可靠。当前数据支持进行小规模、可复现的 fine-tune 试验；不能据此直接开始大规模训练，也不能把 PC FP32 数据当作板端结果。

## 3. 尚未具备的赛题能力

| 缺口 | 为什么重要 | 当前状态 | 解除条件 |
|---|---|---|---|
| FPGA bitstream 与视频链路 | 赛题需要视频输入、处理、显示闭环 | 仓库未见可用 bitstream、采集或 HDMI OSD 实机证据 | FPGA 工程、时序报告、实机输入/输出演示 |
| 真正 PCIe endpoint/DMA | mock 只验证软件协议，不能代替物理链路 | 现有 Host 参考为冻结帧 BAR0 PIO/DMA bring-up，未承诺连续 30 FPS | FPGA endpoint、RK3566 host 驱动、连续帧收发实测 |
| FPGA OSD 与返回控制 | 检测框、箭头、告警需要回写显示 | 未接入 | 用相同 frame_id 的检测元数据驱动 FPGA overlay |
| 四路实机能力 | 规划的 4×6 FPS 不是系统证明 | 未实测 | 四路输入、PCIe、NPU、OSD 同时跑的延迟/丢帧日志 |
| 道路视频正式验收集 | 需要真实场景比较模型、验证状态机 | 只有用户素材和图像镜像；BDD100K 原始视频未取得 | 合法的 6–10 段短视频及 manifest |
| 训练数据 | 当前 BDD 镜像是 validation-only，不能拿来训练 | 未取得合法 train 子集 | 官方或可信来源的 train 数据及许可证记录 |
| 障碍物/斑马线语义 | COCO YOLO11n 没有专用 cone、road-obstacle、zebra crossing 类 | 未解决 | 有标注数据后的模型/传统视觉方案与视频验证 |
| 交通灯颜色可靠性 | COCO 只给 traffic light 框；HSV 在小框、夜间会不稳定 | 软件 adapter 有，实景未验证 | 道路视频标注与红黄绿测试 |
| 最终演示材料 | 赛题要求 1080P 三分钟演示和源码 | 未开始 | 完整系统录像、说明和可复现部署包 |

## 4. 决策边界：哪些结论已经能下，哪些不能

- 已确认：320/416 INT8 模型能在 RK3566 上运行；416 的单图推理更慢；416 在固定 BDD 图像样本中各重点类别 recall 高于 320。
- 合理推测：经过针对道路和小目标的训练后，320 的效果可能接近当前 416，或 416 可进一步改善小目标。这需要同一外部 holdout 实验验证。
- 尚不确定：微调后能否满足比赛准确率、模型量化后是否保留提升、夜间/交通灯颜色是否可靠、PCIe 是否连续传输、FPGA 是否能四路闭环。
- 明确不能声称：当前 PC FP32 结果不是 RK3566 INT8 准确率；单图 smoke 不是连续视频 benchmark；协议 mock 不是 PCIe 实机通信；四路服务率预算不是四路系统实测。

## 5. 接下来按优先级执行

### P0：先把模型选择和训练决策做成可验证实验

1. 获取合法 BDD100K detection train 小子集，或同等权威、带框道路训练数据。不要使用现有 validation-only 镜像训练。
2. 将当前 1,000 张 BDD 图像冻结为外部 holdout；再准备按原始视频/序列分组的 train/val，避免相邻帧泄漏。
3. 以原始 `yolo11n.pt` 做第一轮 320 输入 fine-tune pilot：3,000–8,000 图像、20–40 epoch，重点观察 person、traffic light、bicycle、motorcycle 的 recall/F1。
4. 用完全相同的 holdout 比较：未训练 320、微调 320、未训练 416、微调 416。只在指标和人工视频检查都提升时保留新模型。
5. 将胜出 checkpoint 重新按既有 split-output pipeline 导出 ONNX、INT8 RKNN 320/416，并在 RK3566 做 load/inference/decode 与同一道路素材 smoke。不能沿用旧模型的速度或准确率代替新模型。

建议的模型选择规则：若微调 320 在关键小目标上接近微调 416，则固定 320 作为比赛主模型；只有 416 在 traffic light、远处 person、两轮车上有显著且可重复的优势，才作为高质量模式。不要为提高准确率直接换大模型，RK3566 的实时性余量有限。

### P0：建立道路视频验收集

1. 优先收集官方/比赛相近视频，其次团队自采，最后才是许可证清楚的公开道路视频。
2. 目标为 6–10 段，每段 20–60 秒、720p 或 1080p、25–30 FPS：普通道路、十字路口、交通灯、行人过街、密集交通、移动相机、夜间、路障/锥桶。
3. 每段原始文件只读保存于 `data/competition_eval/raw/`，并在 `data/competition_eval/manifest.csv` 记录来源、许可证、场景、时长和分辨率。
4. 同一素材运行 320/416，输出带框视频、空帧比例、关键类别检测统计和粗略延迟。道路视频没有人工标注时，只能作为定性验收；从中抽取 200–500 帧标注重点类别后，才可计算公平的 precision/recall/F1。

### P1：等待 FPGA 条件到位后的联调顺序

1. 先以冻结帧/单路跑通 FPGA → PCIe → RK3566 → PCIe → FPGA OSD，核对 stream_id、frame_id、timestamp、坐标和超时 fail-safe。
2. 再验证连续单路视频的帧率、端到端延迟、丢帧与结果错配。
3. 最后加入四路，先按每路 6 FPS 轮询起跑；只有有完整日志时才提高到 8 FPS 或更高。
4. 使用道路视频逐项演示：红灯停车、绿灯通行、行人危险、实虚线变道限制、夜间/复杂路口与异常断流。

## 6. 最小完成路线

```text
合法训练小子集 + 固定外部 holdout
            ↓
320/416 微调对照实验 → 选择比赛模型 → 重建 RKNN
            ↓
道路视频 320/416 验收 + 少量人工标注
            ↓
FPGA 单路 PCIe/OSD 闭环 → 连续单路 → 四路压力测试
            ↓
覆盖赛题场景的 1080P 三分钟实机演示与源码材料
```

## 7. 关键文件索引

| 用途 | 文件 |
|---|---|
| 本总览 | `docs/competition_project_handover.md` |
| 赛题逐项完成度 | `docs/competition_requirement_matrix.md` |
| 模型构建、哈希、量化参数 | `docs/competition_model_build.md` |
| BDD100K 外部验证 | `docs/competition_bdd_external_validation.md` |
| 公共数据集调查与下载计划 | `docs/competition_public_dataset_survey.md`、`docs/competition_eval_download_plan.md` |
| PCIe 软件契约与 Host 限制 | `docs/competition_pcie_protocol.md` |
| 320/416 640 历史性能口径 | `docs/competition_640_benchmark_note.md`、`docs/competition_multistream_budget.md` |
| 系统架构与风险 | `docs/competition_system_architecture.md`、`docs/competition_remaining_risks.md` |
| 带框视觉结果 | `outputs/bdd100k_external/`、`outputs/user_video_preacceptance/` |
