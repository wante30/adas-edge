# 2026 FPGA 创新设计赛道选题三：比赛项目交接总览

更新日期：2026-09-08

本文件只记录与“以紫光同创 FPGA 为核心的道路环境感知、辅助驾驶与高速通信系统”赛题直接相关的工作。MOT17 研究、论文实验和其他仓库改动不作为本项目完成度证据。

> **2026-09-17 更新提醒**：本文件第 2 节的"模型候选"与"当前检测效果"两张表已被本轮的矩形输入工作取代。
> 现行结论见文末"## 2026-09-17 更新"一节，以及
> [检测器最终配置](competition_final_detector.md)、[指标协议](competition_detection_metric_protocol.md)。

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


---

## 2026-09-17 更新：矩形输入、标准指标与 RK3566 四路实测

本节取代前面关于"320/416 方画布候选"和"PC recall 表"的内容。**本轮没有 fine-tune**，
检测权重仍是原始 COCO-80 `yolo11n.pt`。

### 指标口径改了

之前的表只报 recall（单点、贪心一对一），不能和论文的 mAP 比较。现在两套口径分列，禁止混用：
`macro recall`（历史口径）/ `mAP@0.5` / `mAP@0.5:0.95`。协议见
[competition_detection_metric_protocol.md](competition_detection_metric_protocol.md)。

### 画布：640×360 非法

`640×360` 不是合法模型画布（360 不是 32 的倍数），`check_imgsz` 会静默取整成 `640×384`。
本轮改用 `640×352` 并同时测了 `640×384` 作对照。

### 冻结 holdout 上的标准指标（COCO yolo11n.pt，1000 张）

| 画布 | 像素 | mAP@0.5 | mAP@0.5:0.95 | macro recall@0.25 | 小目标 recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 320×192 | 61,440 | 0.1122 | 0.0672 | 0.0993 | 0.0009 |
| 416×256 | 106,496 | 0.1675 | 0.0991 | 0.1484 | 0.0097 |
| 512×288 | 147,456 | 0.1999 | 0.1183 | 0.1792 | 0.0235 |
| 640×352 | 225,280 | 0.2329 | 0.1393 | 0.2096 | 0.0522 |
| 640×384（对照） | 245,760 | 0.2506 | 0.1469 | 0.2249 | 0.0547 |

### RK3566 实测：旧方画布候选被支配

| 旧候选 | 旧延迟 | 矩形替代 | 新延迟 | 变化 |
| --- | ---: | --- | ---: | --- |
| 320×320 INT8 | 24.14 ms | 320×192 | **16.12 ms** | −33% |
| 416×416 INT8 | 38.35 ms | 416×256 | **23.86 ms** | −38% |
| — | — | 512×288 | 30.42 ms | 比旧 416 快 21% 且更准 |

### 四路：异构配置 + 加权轮询

| 配置 | 聚合 FPS | 达成率 |
| --- | ---: | ---: |
| CONFIG-A 全 320×192 | 25.72 / 28 | 0.919 |
| CONFIG-B FRONT 512×288 | 23.86 / 28 | 0.852 |
| CONFIG-C 侧路 416×256 | 22.91 / 28 | 0.818 |
| CONFIG-D FRONT 640×352 | 22.42 / 28 | 0.801 |
| **CONFIG-B-WRR-FAIR（推荐）** | **27.91 / 28** | **0.997** |

推荐配置：**FRONT 512×288 @10 FPS + LEFT/RIGHT/BACK 320×192 @6 FPS，加权轮询权重 10:6:6:6，队列深度 3**。
10 分钟长测（CONFIG-B）稳定无崩溃。详见 [异构四路实测](competition_heterogeneous_multistream.md)
与 [四路预算](competition_multistream_budget.md)。

### 新的缺口表

| 缺口 | 现状 |
| --- | --- |
| traffic light | **主检测器确认不可用**（512×288 下 78.6% 的 GT 框无任何候选）。需 front-only 专用支路，本轮不训练 |
| 四路验证的边界 | 本轮的"四路"是**文件视频回放**，不是 PCIe 四路输入。调度/队列逻辑可直接复用，输入源不同 |
| FPGA 视频采集 / OSD / 连续 PCIe | 仍无进展，维持原判断 |
| BDD100K train | 仍不可得，本轮所有训练相关实验都用 nuImages 或未做 |

---

## 2026-09-18 更新：长测定稿、ROI 复核与 INT8 per-class 复核

三个收尾 Gate 的结果（详见对应文档）：

1. **300s=27.91 vs 600s=23.66 已归因**：measurement protocol difference——600s 那次
   （CONFIG-B-10MIN）误用了 rate 调度器，而 27.91 是 WRR-FAIR。同协议下 600s 对 300s 仅 -0.8%。
   修复后的 20 分钟 WRR-FAIR-D1 长测：**27.30/28（97.5%），全程平坦，SoC ≤38.9 °C，
   NPU 恒 600 MHz**，无热降频/调度/队列问题。正式配置以长测为准。
2. **Traffic-light ROI 复核：FAIL。** 数据驱动 ROI（覆盖 94.4% GT、面积 57%）+ 640×352 + conf 0.05
   下 recall 仅 0.149（全帧 0.141），67.7% 的 GT 仍无任何候选。
   **TRAFFIC_LIGHT_SPECIALIST_REQUIRED 定稿**；专项支路若按 512×288@3FPS 追加约 +9% NPU 占用
   （5FPS@640×352 不可行），但这只在换用专项小模型时才需要重新评估。
3. **INT8 per-class 复核：接受。** 100 张道路 holdout 图、真实板端 INT8 vs PC FP32：
   部署工作点 per-class recall Δ≤0.008（car 0.326→0.324，person 0.175→0.168）；
   AP50 下降 0.01–0.11 属置信度标定漂移。工具 `tools/int8_perclass_eval.py`。

新增工具：`tools/telemetry_sampler.py`（SoC 温度/频率遥测）、`tools/traffic_light_roi_eval.py`、
`tools/int8_perclass_eval.py`。原始数据在 `D:/Datasets/adas_edge_training/phase10/`（LONG20-*、
int8check/）与 `D:/Datasets/adas_edge_training/roi_eval/`。

---

## 2026-09-18 晚间更新：TL specialist 交付，AI 软件侧冻结

FRONT-only traffic-light specialist 全流程完成并通过 Gate（此前"MAIN YOLO NOT SUFFICIENT"的缺口就此闭环）：

- 数据：BDD val 非 holdout 9000 张（23468 TL 框），512×192 ROI 画布，与验收集严格隔离；
- 训练：YOLO11n 单类头，rect 训练，PC FP32 holdout TL AP50 0.136（主模型 0.083）、R@0.05 0.272（主 0.149）；
- 量化：普通 INT8 因 box 分支 2 px 步长对 5×8 px 目标致命（AP50 0.101→0.029，混搭实验定位），
  **混合量化**（框链 25 层 fp16）恢复到 AP50 0.069 / 板端 R@0.05 0.199 = 主模型板端 2.1 倍；
- 预算：3 FPS 接入 FRONT，推理线程 78%→85.6%（5 FPS 90.7% 亦可）；
- 主 YOLO / 调度 / 四路配置零改动。

详见 [competition_tl_specialist.md](competition_tl_specialist.md)。**RK3566 AI 软件侧就此冻结**
（主检测器 2 个 RKNN + specialist 1 个 RKNN + 调度器 + 队列参数），剩余为 FPGA 硬件侧事项。

---

## 2026-09-21 更新：FRONT 强模型评估关闭，维持 YOLO11n

按比赛方要求做了「是否把 FRONT 主检测器升级为 YOLO11s」的最后一轮评估，结论 **KEEP CURRENT YOLO11N**：

- BASE-S（不训练）精度确实显著更好：mAP50 0.1999 → 0.2848（+8.5pp），person recall 0.216 → 0.295，
  small-object recall 0.0235 → 0.0590，每类都涨、无退化；
- 但 11s 的算力是 11n 的 **3.31×**（矩形输入精确 FLOPs：7.74 GF vs 2.34 GF），
  板端标定外推 full pipeline ≈ 7.9 FPS（乐观口径 11.0 FPS）；
- **决定性约束是四路预算**：现有配置已占推理线程 87.2%，换 11s 后 FRONT 8 FPS 需要 105%（乐观）～134%（FLOPs 口径），
  结构性装不下。要装下只能把 FRONT 降到 5 FPS，破坏已冻结的四路形态，因此不做。
- 本轮板子未开机，11s 的板端实测（NPU/pipeline/INT8 per-class）**未取得**；开机后可用
  [benchmark 文档 §5 的命令](front_strong_model_benchmark.md)在 10 分钟内复核外推（判读区间 62–101 ms）。

AI 主模型就此冻结。SIDE/BACK、specialist、调度器、队列参数全程零改动。
