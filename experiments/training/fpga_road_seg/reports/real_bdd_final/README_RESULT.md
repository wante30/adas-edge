# FPGA 道路分割真实结果

状态：三模型真实训练与 held-out 评估完成。所有精度来自本分支新模型、新 checkpoint，没有复制 PR #4/B-2a 精度或权重。

## 数据与监督

完整恢复 BDD100K 100K Images 与每图 JSON，70k train、10k val 一一配对；20k test 无公开真值。
数据来源、大小和 MD5 见 `reports/real_bdd_recovery.md` 与本地下载 manifest；公开镜像不是官方签名背书。
从 JSON 的 poly2d 渲染原生 Mask，10px 原图线宽；direct/alternative 合并，solid/dashed 根据属性区分。
62,984 张训练，7,012 张内部验证，10,000 张官方 val 用作 held-out。与官方 val 视频前缀重叠的 4 张 train 帧隔离，
其原图、JSON 和 Mask 均保留；3 对同 split 相同 JPEG 保留，没有跨 split 相同内容。
混合检测 JSON 中缺失某项道路标注时，对应有效性为零，不生成全背景监督。核查：`{"version": 1, "samples": 79996, "missing_drivable": 3533, "missing_lane": 3834, "policy": "Mixed objects JSON without lane/area annotations has zero validity for that task; no invented background labels"}`。

crosswalk 仅有 BDD 稀疏曲线：主模型屏蔽该通道；独立 weak-crosswalk checkpoint 为正像素弱监督试验，
正监督存在全阳性退化解，不能据此证明定位能力。Mapillary Vistas 的密集斑马线标签尚未获得。
**全部 dense crosswalk IoU/Recall 为 N/A**，也没有伪造全背景斑马线标签。

## 真实 held-out 指标

| 模型 | 参数 | MACs | drivable IoU | solid F1 | dashed F1 | lane IoU | GPU p50/p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| candidate | 8508 | 25351680 | 0.5146 | 0.5127 | 0.3748 | N/A | 1.28/2.39 |
| candidate-plus | 32524 | 84718080 | 0.5541 | 0.5649 | 0.4505 | N/A | 1.14/3.82 |
| twin-small | 121552 | 323968256 | 0.6742 | N/A | N/A | 0.4929 | 6.51/10.00 |

IoU 是前景 IoU；F1 在下采样后的有效像素上按全数据 TP/FP/FN 累计。
320×192 RGB letterbox，固定阈值 0.5，padding 无效。线宽影响细线指标，不能直接与旧 B-2a 不同口径对比。
TwinLiteNet+ Small 保留原始两头结构，只输出 drivable/lane，solid/dashed F1 没有定义。
MACs 是 Conv/ConvTranspose/Linear/MatMul 的软件计算量，不包含所有逐元素开销。
GPU 延迟是已预处理 batch=1 FP32 模型前向，不含解码、传输和 FPGA 时延。

## 训练证据

每模型先做 512/128 张真实 train/val 的 3 epoch 试训，再继续全量训练。
小样本与全量验证范围不同，最佳损失分别选择；history 逐 epoch 保留样本量。

- candidate：12 个当前规模 epoch，train=62984、internal-val=7012，AMP=True，GPU=NVIDIA GeForce RTX 4060 Laptop GPU；checkpoint SHA-256 `77435a51e1e9338267f4f6d674b7e42e286a9f1cef6276b236fb6cb2ba9c4634`。
- candidate-plus：12 个当前规模 epoch，train=62984、internal-val=7012，AMP=True，GPU=NVIDIA GeForce RTX 4060 Laptop GPU；checkpoint SHA-256 `40aecc70bbc5d9f7a1e16712e2e6c5d4dc56d6b41f8145253c3ec5e5757d4308`。
- twin-small：12 个当前规模 epoch，train=62984、internal-val=7012，AMP=True，GPU=NVIDIA GeForce RTX 4060 Laptop GPU；checkpoint SHA-256 `477f9f0e8d5a7b43070376ba514ca31980aeb86c0608be46ea97539de0391758`。

最佳权重仅由内部验证 loss 选择。`training_curves.png` 包含试训到全量的边界，边界处损失不可直接视为回退。
历史无效监督试训已移入 `failures/invalid_pilots_before_task_availability_fix/`，不参与当前训练、指标和导出。

## 推荐与硬件依赖

软件交接优先候选：**candidate-plus**。在两个能直接输出实/虚线的 FPGA-friendly 候选中，按内部验证的 drivable IoU、solid F1、dashed F1 三项平均值选择，未使用 held-out 指标选模型。最终硬件推荐需通过下列板端适配条件。

Candidate 系列：Conv（普通/Depthwise/Pointwise）、ReLU、Add、nearest Resize 等。
TwinLiteNet+ Small 还含 PReLU、dilated Conv、ConvTranspose、CAAM 注意力、MatMul/Linear/Softmax、池化、reshape/transpose。
软件导出成功不等于 PG2L200H 支持；LUT/DSP/BRAM、目标时钟/时序、最大张量、dilation/resize 语义、
INT8 累加/四舍五入/饱和、权重打包格式和传输协议仍需队友确认。板端 FPS/功耗/时延没有实测。
交接接口与逐层校验步骤见 `docs/fpga_road_seg_handoff.md`。现有 FAST-Q4、PCIe、HUD、Tesla BEV 和 FPGA 工程未修改。

## 导出与量化对照

### candidate-plus INT8 PTQ/QDQ
校准：256 张 train 图像；比较：10000 张 test。
| 类别 | FP32 IoU | INT8 IoU | IoU 差值 | FP32 F1 | INT8 F1 |
|---|---:|---:|---:|---:|---:|
| drivable_area | 0.5541 | 0.4505 | -0.1036 | 0.7131 | 0.6212 |
| solid_lane | 0.3936 | 0.3521 | -0.0415 | 0.5649 | 0.5208 |
| dashed_lane | 0.2908 | 0.2209 | -0.0699 | 0.4505 | 0.3618 |
PyTorch/FP32 ONNX 最大 logit 差 9.72748e-05；FP32/INT8 有效像素阈值一致率 97.6883%。
FP32 实际算子：`{"Conv": 14, "Relu": 11, "Shape": 3, "Constant": 12, "Slice": 3, "Concat": 3, "Resize": 3, "Add": 2}`。
仅 Conv 权重/激活为 W8A8；Add、Resize、I/O 等仍浮点。该 ONNX 不能直接烧写 FPGA。

### twin-small INT8 PTQ/QDQ
校准：256 张 train 图像；比较：10000 张 test。
| 类别 | FP32 IoU | INT8 IoU | IoU 差值 | FP32 F1 | INT8 F1 |
|---|---:|---:|---:|---:|---:|
| drivable_area | 0.6742 | 0.6616 | -0.0126 | 0.8054 | 0.7964 |
| lane | 0.4929 | 0.4870 | -0.0059 | 0.6603 | 0.6550 |
PyTorch/FP32 ONNX 最大 logit 差 0.000234604；FP32/INT8 有效像素阈值一致率 99.2534%。
FP32 实际算子：`{"Constant": 17, "Conv": 91, "PRelu": 31, "AveragePool": 3, "Concat": 15, "Add": 31, "BatchNormalization": 13, "Sigmoid": 1, "Reshape": 10, "Transpose": 6, "Unsqueeze": 3, "Softmax": 2, "MatMul": 7, "Mul": 1, "ConstantOfShape": 1, "Expand": 1, "Tile": 1, "ConvTranspose": 6, "Gather": 4, "Sub": 2}`。
仅 Conv 权重/激活为 W8A8；Add、Resize、I/O 等仍浮点。该 ONNX 不能直接烧写 FPGA。


## 文件

- `metrics.csv`：三模型实际指标；空值表示无定义/无监督。
- `training_curves.png`：真实训练/内部验证曲线。
- `comparisons/all_models/`：同一 held-out 图像的输入、GT 和三模型预测。
- `failures/<model>/`：固定前 512 张 held-out 中的低 IoU 案例及 TP/FP/FN；这是子集筛选，未覆盖全体错误。
- `checkpoints/<model>/`：PyTorch best/last/周期 checkpoint；`pilot/` 为有效试训。
- `exports/`：FP32 ONNX、INT8 QDQ、校准清单、算子和量化参数、精度对照。
- `run.log`：运行日志；旧失败保留以供审计，不能将重试前的损失混入本次结果。

数据、权重及完整日志留在独立 outputs/data 目录，不提交到普通 Git。
