# PG2L200H 第二阶段交接

- 架构未改：Conv 3×3 / depthwise / pointwise，BN 导出折叠，ReLU、残差 Add、nearest Resize；无注意力。
- 输入 RGB FP32 NCHW [1,3,192,320]；uint8/255；等比 letterbox，填 114；标签 padding 不计分。
- 输出 logits [1,4,192,320]，顺序 drivable_area、solid_lane、dashed_lane、crosswalk；独立 sigmoid≥0.5，可重叠，非互斥 softmax。
- 参数 32524、MACs 84718080；模型校验和 f12fac988aa29d7ba68d94af2bd0261a6eacf35b111b4c58141119d40bef6498。
- exports/fp32-operators.json / int8-operators.json 包含实际导出算子与 tensor 形状，quantization.json 包含逐层 scale、zero_point、int32 bias 与 multiplier/shift 近似。
- 量化器 u8_percentile999，关键指标最大降 2.07 点；软件量化门槛未通过，需继续 QAT/误差定位。
- 待确认：depthwise groups、各通道倍数、padding/stride、Add 跨支路 scale 对齐、nearest 坐标、INT32 累加溢出、舍入/饱和、非零 zero_point 修正、片上缓存/DDR/带宽、权重打包与 PCIe 协议。
- 当前 Conv QDQ 外仍有浮点，必须确认完整整数图适配；没有板端资源、频率、FPS、功耗、准确率实测。
- crosswalk 定义为过街区域（含条纹间隙），不能解释为单条白纹；数据许可状态见结果报告。
- 不涉及 FAST-Q4、PCIe、HUD、Tesla BEV、队友 FPGA 工程或队友模型权重修改。
