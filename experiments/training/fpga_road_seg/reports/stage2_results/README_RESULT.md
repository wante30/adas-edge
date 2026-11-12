# 第二阶段真实训练结果

Candidate-plus 从原最佳权重继续混合微调，网络未改变：32,524 参数、84,718,080 MACs（320×192，Conv 算术统计）。
第一阶段所有模型和量化产物保留。本轮实际完成 12 个混合采样 epoch，附加蒸馏 0 轮。
当前选择混合微调分支最佳 epoch 12，分支比较见 refinement.json。

| 数据集 / 指标 | 微调前 FP32 | 微调后 FP32 | INT8 |
|---|---:|---:|---:|
| bdd100k drivable_area iou | 55.41% | 55.45% | 54.71% |
| bdd100k solid_lane f1 | 56.49% | 55.13% | 54.88% |
| bdd100k dashed_lane f1 | 45.05% | 44.82% | 44.26% |
| rlmd crosswalk iou | 0.01% | 38.94% | 38.71% |
| rlmd crosswalk f1 | 0.01% | 56.05% | 55.81% |
| rlmd crosswalk recall | 0.29% | 55.70% | 55.79% |

BDD 表格是 10,000 张官方 val held-out；模型选择使用另有 7,012 张内部验证。
RLMD 是 439 张独立分组验证，训练 1698 张；没有额外独立 RLMD 测试集，不能把它称为测试精度。
前模型 crosswalk 未训练，数值仅用于展示未训练输出，不是旧模型有效斑马线精度。
RLMD 标注覆盖整个过街区域、含条纹间隙；未标 drivable 的 loss 始终屏蔽。
BDD 旧曲线 crosswalk 在本轮屏蔽。数字文件名分组和近似图像排查减少邻帧泄漏，但原数据缺少完整录像 ID，仍不能证明场景完全独立。
所有分割指标在 320×192 letterbox 网格上统计，padding 不计分；不是原始分辨率逐像素评分。

## 量化对照

先完成 FP32 微调，再在真实 TRAIN 样本上进行 5 种 PTQ 对照。
校准 192 张，BDD / RLMD 平衡且包含 crosswalk 正负样本；不用验证/测试图校准。
量化器选择使用内部验证，选择 `qat`。完整候选结果见 exports/search.json。
PyTorch / ONNX 最大 logit 误差 7.34329e-05。
PTQ 未通过内部逐项 2 点保护时，自动尝试最多 3 轮真实混合数据 QAT（冻结 BN/校准范围，FP32 logits 蒸馏正则），结果及异常均记录于 exports/。
本轮显式要求 QAT：True；实际 QAT 是否成功、是否被采用见 exports/search.json，不能把已请求等同于已成功。
实际完成 QAT 3 轮，内部验证最佳第 1 轮；最终采用 `qat`。QAT 权重位于 checkpoints/candidate-plus-qat/，
最佳 FX QAT checkpoint SHA-256：`48e6a2dccf1dc98095b41e895cd9c99f7ab044b3c8a343cb4a364275c6fceffc`。FX 权重键与原始 FP32 不同，恢复方法见该目录 README.md。
若本轮从之前已评估 PTQ 的结果继续，先前的留出集结果已经看过，后续同集合复评不能称为新的盲测；量化参数/候选选择仍只使用训练校准和内部验证。
表中关键 IoU/F1/Recall 最大量化下降 0.74 个百分点；2 点软件门槛：通过。
ONNX 只有 Conv W8A8，Add / Resize / I/O 仍浮点，不能直接烧录 PG2L200H。
当前导出有 14 层 Conv 使用浮点 bias，须按 input_scale×weight_scale 转为 INT32 并逐层验证；quantization.json 不会把参考图 bias 存储误写成板端格式。
如选用 unsigned asymmetric 激活，RTL 需明确支持零点修正。QDQ 文件不是队友硬件权重格式。

## 真实道路检查与性能

comparisons/ 包含真实道路 GT、微调前后对照、原始连续视频预测；failures/ 按真实 RLMD crosswalk IoU 排序并记录 FP/FN。
本地原始视频已生成连续预测，详情见 comparisons/video.json，没有逐像素视频真值，不报告视频精度。
RTX 4060 FP32 batch1 前向 p50/p95：1.324/2.733 ms；不含预处理/传输/绘图，不是 FPGA 延迟。
CPU ONNX Runtime（4 线程，真实道路输入、20 次预热/100 次测量）前向 p50：FP32 1.097 ms，
INT8 7.172 ms。这是该 CPU/运行时的实测，不能推断 INT8 必然更快或推断 FPGA 吞吐；
comparisons/quantization/ 提供 FP32/INT8/GT 和阈值分歧图，quantization_comparison.png 为内部验证候选对照。

## 交接与限制

best checkpoint SHA-256：`f12fac988aa29d7ba68d94af2bd0261a6eacf35b111b4c58141119d40bef6498`。训练/优化器/AMP/随机状态保存在 checkpoints/，可恢复。
官方 RLMD 无明确许可证；公开副本 MIT 声明未获原作者确认。本地科研实验，数据与预测对照均不随代码提交。
权重、原数据、视频不进普通 Git。详细算子及接口合同见 FPGA_HANDOFF.md。
本轮仅形成待硬件验收的软件候选；是否满足道路实际可用性需结合错误图及场景验证，不能由 loss 单独断言。
中间真实图发现 RLMD 仪表台区域 drivable 误检和斑马线碎片；RLMD 无 drivable 真值，不能为消除误检而伪造背景监督。未提供这类场景的可行驶区域实测精度。
最终失败图 rlmd-28180 的雨天过街区域完全漏检（对应样本 crosswalk IoU=0）；视频只做抽帧定性检查，不能把保存预测视频等同于验证连续稳定性。
给 PG2L200H 的建议：保留 Candidate-plus FP32 为当前四通道软件基线，先确认算子和量化合同，再做板端逐层对齐；现有斑马线质量不足以宣布模型已满足实际道路使用要求。
