"""Chinese engineering specification generated from the pinned graph export."""
import json
from pathlib import Path
import shutil

from .handoff_integer import load_json


def documents(root, model_root, repo):
    root,model_root,repo=map(Path,(root,model_root,repo))
    contract=load_json(root/'04_量化参数/integer_contract.json')
    summary=load_json(root/'03_网络结构与算子/architecture_summary.json')
    identity=load_json(root/'MODEL_IDENTITY.json')
    samples=load_json(root/'05_硬件验证向量/samples.json')
    search=load_json(model_root/'exports/search.json')
    stages=contract['stages'];byid={s['id']:s for s in stages}
    versions=summary['runtime_versions']
    table=[];quant=[];convtable=[]
    for s in stages:
        ins=' / '.join('×'.join(map(str,byid[k]['shape'][1:])) for k in s['inputs']) or 'RGB 3×192×320'
        shape='×'.join(map(str,s['shape'][1:]))
        kernel='×'.join(map(str,s.get('kernel',[]))) or '—'
        table.append(f"| {s['id']} | {s['op']}{'+ReLU' if s.get('relu') else ''} | {ins} | {shape} | {kernel} | {s.get('strides','—')} | {s.get('pads','—')} | {s.get('groups','—')} | {s.get('weight_parameters',0)}+{s.get('bias_parameters',0)} | {s['macs']:,} | {s['logical_io_bytes']:,} |")
        sw=s.get('weight_scales',[])
        quant.append(f"| {s['id']} | {s['scale']:.17g} | {s['zero_point']} | {min(sw):.17g} ～ {max(sw):.17g}"+' |' if sw else f"| {s['id']} | {s['scale']:.17g} | {s['zero_point']} | — |")
        if s['op']=='Conv':
            convtable.append(f"| {s['id']} | `{s['name']}` | {s['weights']['shape']} | {s['accumulator_bound']:,} | {s['minimum_safe_signed_accumulator_bits']} | {s['bias_rounding_max_abs_real_error']:.9g} | {s['multiplier_max_abs_error']:.9g} |")
    metrics=[]
    for domain,cls,metric in [('bdd100k','drivable_area','iou'),('bdd100k','solid_lane','f1'),('bdd100k','dashed_lane','f1'),
                              ('rlmd','crosswalk','iou'),('rlmd','crosswalk','f1'),('rlmd','crosswalk','recall')]:
        fp=search['comparison']['fp32'][domain]['classes'][cls][metric]*100
        iq=search['comparison']['int8'][domain]['classes'][cls][metric]*100
        metrics.append(f'| {domain} {cls} {metric} | {fp:.6f}% | {iq:.6f}% | {iq-fp:+.6f} pp |')
    vector_table='\n'.join(f"| {s['id']} | {s['frame_zero_based']} | {s['time_seconds']:.3f} | {s['description']} | {s['model_grid_mask_agreement']*100:.6f}% | {s['qdq_vs_integer_max_logit_error']:.9g} |" for s in samples)
    diagram=(root/'03_网络结构与算子/network.mmd').read_text(encoding='utf-8')
    document=f'''# Candidate-plus 第二阶段最终 QAT：PG2L200H 技术交接文档

版本：{identity['version']}。模型与权重冻结；生成时间（UTC）：{identity['generated_utc']}。

本交付已经具备软件图、参数、标准二进制、全整数参考和真实场景逐层向量，可开始独立 CNN 加速器开发。**尚未进行 PG2L200H RTL 仿真、综合、布局布线、板端推理或资源/频率验收。QDQ ONNX 不能直接烧录。**

## 1. 项目定位、来源和冻结范围

ADAS Edge 全国 FPGA 决赛项目的第二套 FPGA CNN 加速器，用独立四通道道路结构分割辅助已有系统。现有 FAST-Q4、RK3566、Qt、HUD、PCIe、Tesla BEV 和队友第一套 FPGA 车道模型均不在本次修改范围。这里交付的是新模型的计算规范，不能推断现有 RTL 已支持相同数据排列或寄存器协议。

Candidate-plus 是仓库 `experiments/training/fpga_road_seg/models.py` 中独立定义的 `RoadCandidatePlus`，来源为同仓库 8,508 参数 Candidate 的加宽版本，不是 TwinLiteNet+ Small 的原版网络或其权重。宽度 32→56→96，编码至 1/8，再用两个残差跳连恢复到 1/2，最后四通道 head 和最近邻恢复到原输入大小。没有注意力、反卷积、Softmax、Concat 特征融合或动态分辨率。ONNX 中的三个 Concat 仅构建固定 Resize 的尺寸向量，可在硬件离线常量化。

训练模型：32,524 可训练参数，84,718,080 Conv MAC/帧，169,436,160 FLOPs/帧（2 FLOPs/MAC；不计 BN、ReLU、Add、Resize、量化和访存）。14 层卷积：1 普通 3×3、5 Depthwise 3×3、8 Pointwise 1×1。训练有 11 个 BN；QAT 导出时已融合，不需要硬件 BN。

第二阶段从既有 Candidate-plus 最佳 checkpoint 继续 BDD100K/RLMD 混合微调 12 epoch；最终 QAT 共 3 epoch，最佳为第 1 轮。QAT 冻结 BN/观测范围，AdamW LR=1e-5，真实监督加 0.2 的带有效像素掩码 FP32-logit 正则。观测初始化使用 192 张真实训练图，不使用验证图更新范围。BDD drivable/solid/dashed 有效、crosswalk 弱曲线本轮屏蔽；RLMD solid/dashed/密集 crosswalk 有效、drivable 缺失监督屏蔽。

### 1.1 模型身份

| 文件 | SHA256 |
|---|---|
| 最终 `model.int8.qdq.onnx` | `{identity['sha256']['int8']}` |
| `qat.best.pt`，FX QAT 最佳权重 | `{identity['sha256']['qat_checkpoint']}` |
| `model.fp32.onnx`，QAT 前第二阶段基线 | `{identity['sha256']['fp32']}` |
| `fp32.best.pt` | `{identity['sha256']['fp32_checkpoint']}` |

原电脑绝对路径见包根 `MODEL_IDENTITY.json`；包内路径均可相对运行。QAT checkpoint 键是 FX 变换后的网络，不能直接塞进原始 `RoadCandidatePlus`。包内 `restore_checkpoint.py` 重建相同 QConfig/FX 后严格载入，另提供已可加载的 QAT TorchScript。

## 2. 完整网络结构与算子

以下图、表由真实最终 QAT ONNX 提取。stage 指激活量化边界，不是 ONNX 节点编号；s00 是输入量化。全部 shape 为 NCHW、batch=1，下表省略 N。

```mermaid
{diagram.rstrip()}
```

### 2.1 完整计算阶段表

| 阶段 | 算子 | 输入 C×H×W | 输出 C×H×W | 核 | Stride H,W | Pad T,L,B,R | Groups | 权重+Bias 元素 | MACs | 逻辑读写字节/帧 |
|---|---|---|---|---|---|---|---|---|---:|---:|
{chr(10).join(table)}

逻辑读写 = 所有输入 UINT8 + 输出 UINT8 + 当前权重 INT8 + Bias INT32，各张量只读/写一次的未融合阶段账本；不含 scale/乘子元数据、量化器、外部 RGB 输入和实际缓存重复访问，不能作为实测 DDR 带宽。s00 表中的输入流读成本未计入该账本。

权重 OIHW、Depthwise 为 [C,1,3,3]，无翻转的 cross-correlation，所有 dilation=[1,1]。Pad 在量化字节域必须填该层输入 zero_point，减零点后才等于数学零。普通输入 letterbox 的 114 是图像预处理，与卷积 padding zero_point 是两件不同的事。

| 阶段 | 真实 ONNX Conv 名称 | 权重 Shape OIHW | 最坏 acc 绝对值上界（含 Bias） | 最少有符号位数 | Bias 整数化最大实值误差 | Requant 比例最大近似误差 |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(convtable)}

累加上界按任意 UINT8 输入和真实权重绝对和计算，未利用 ReLU 输入范围收紧；所有层均静态验证 INT32 可容纳。参考程序仍用 INT64 作精确累加 oracle，落到每层 INT32 时检查溢出，发生溢出直接报错，绝不默认绕回。

### 2.2 源图与 CPU 优化图

原 QDQ ONNX 有 {summary['onnx_nodes']} 节点：{json.dumps(summary['onnx_operators'],ensure_ascii=False)}。没有未提取的神经网络算子。

- 普通 3×3：RGB 三通道 MAC 与两行行缓存，可做 output-channel 并行。
- Depthwise：groups=C，各通道独立 3×3 窗口；不可按普通卷积误做通道求和。
- Pointwise：占主要 MAC，适合 input/output-channel 分块、DSP MAC 阵列与权重复用。
- ReLU：11 处，Bias 相加后先对累加值取 max(acc,0)，再重缩放；无 ReLU 的 project/project2/head 允许负数，其 UINT8 zero_point 非零。
- Add：两个残差分支量化域不同，必须对齐尺度、减各自零点；不能直接加 UINT8 字节。跳连 s01、s03 必须保留到 s15、s10。
- Nearest Resize：三个 2×上采样，asymmetric + floor，每个像素复制 2×2；量化 scale/zp 不变，可用重复读/广播代替存储完整放大图。
- Shape/Slice/Concat：仅固定 shape 元数据，离线展开，不需要通用动态 shape 引擎。
- Q/DQ：源图表达量化/反量化，CPU Runtime 可融合；不是 RTL 必须每层浮点 DQ/Q。

**实测提取注意事项：**给原始 QDQ 所有节点加输出会阻止 ORT 融合，结果可与历史默认 CPU 结果不同。本包先保存 ORT 优化图（14 QLinearConv、2 QLinearAdd、3 Resize、2 Transpose、输入 Q 和输出 DQ），再暴露其整数边界，逐样本断言最终 logits 与未插桩默认 CPU Session 完全相同。`ort_cpu.optimized.onnx` 是该 CPU/ORT 的追踪证据，包含 com.microsoft 算子及 NHWC 优化，不能当成可移植 FPGA 文件。交付标准张量统一转回 NCHW。

## 3. 输入输出与 Mask 语义

1. 原图为 RGB uint8；OpenCV 读取的 BGR 必须先转 RGB。目标宽=320、高=192，NCHW [1,3,192,320]。
2. scale=min(320/W,192/H)，nw=max(1,round(W×scale))，nh 同理；round 为 Python ties-even。OpenCV INTER_LINEAR 缩放，左=(320-nw)//2，上=(192-nh)//2，余量放右/下；填充值 RGB(114,114,114)，无 crop，严格保持原始宽高比。
3. 软件 ONNX 输入 float32 RGB/255，mean=0、std=1。硬件核心入口另提供完成激活量化的 UINT8 NCHW；不要重复 /255 或误认为 ONNX 输入本来就是 UINT8。输入 256 项 LUT 从真实 scale/FP32 预处理编译，可在 host 或输入模块完成量化。
4. 四通道输出顺序：0 `drivable_area`、1 `solid_lane`、2 `dashed_lane`、3 `crosswalk`。四个独立 sigmoid，而非含背景的互斥类别；不能 argmax，也没有第五个背景通道。源 ONNX 输出 float32 logits [1,4,192,320]；integer-v1 输出 UINT8 同 shape，scale={stages[-1]['scale']:.17g}、zp={stages[-1]['zero_point']}。
5. 模型网格上 sigmoid(logit)≥0.5 等价于 logit≥0，即整数 q≥{stages[-1]['zero_point']}。等号包含在前景中。原图可视化必须先 sigmoid，在概率域裁去 letterbox padding，再双线性还原 W×H，最后 ≥0.5；不要在原图还原时将“先阈值再放大”误写成相同规则。
6. drivable 代表标注可行驶区域，不保证安全可行驶；solid/dashed 是像素标线而非拟合几何曲线。RLMD crosswalk 是整个过街区域，包含条纹间隙；不可只期待白色条纹像素。覆盖颜色只用于显示：绿、红、青、洋红，显示覆盖顺序不改变四通道独立预测。

1920×1080 示例变为 320×180，上下各 6；所有验证样本 sample.json 保存精确 geometry。

## 4. 量化规范：现有软件模型与全整数 v1 的边界

### 4.1 从 ONNX 提取的真实量化参数

权重 signed INT8 [-128,127]，per-channel axis=0，symmetric，所有权重 zp=0。激活 unsigned UINT8 [0,255]，per-tensor affine，非对称（ReLU 后通常 zp=0，带负数的投影/残差/head 不为 0）。BN 已折叠。每层 bias 在源 QDQ 中仍为 float32。

| 激活阶段 | 精确 float32 scale（十进制 round-trip） | zero_point | 该 Conv 权重通道 scale 范围 |
|---|---:|---:|---|
{chr(10).join(quant)}

全部 {summary['folded_bias_elements']} 个输出通道的精确权重 scale/zp、原 FP32 Bias、INT32 Bias、bias_scale、重缩放乘子和移位在 `04_量化参数/per_output_channel_quantization.csv`；JSON 数值来自 ONNX float32 提升至 float64，二进制 float32 scale 是最终权威位模式，不允许从表中缩短小数重新计算。

### 4.2 软件 QDQ 的实际含义

Q=clip(RNE(x/scale)+zp)，DQ=(q-zp)×scale。源图 Conv 输入/权重反量化后做 float Conv，并加 float bias，Add/Resize/I/O 含 float。实际 CPU ORT 可融合成 QLinearConv/Add，其 bias/requant 内核与浮点路径不保证逐位相同。因此“INT8 QAT 已通过软件精度门槛”不代表“全整数 RTL 已对齐”。历史精度对应最终 ONNX 默认 CPU Session，不对应本次新定义 integer-v1。

### 4.3 新交付 integer-v1 计算规则

这是由最终 QAT 的同一权重和范围确定的 **RTL 建议规范**，没有改训练权重、没有重校准；它确定了原软件图尚未固定的整数 Bias/Add/Requant 规则。

- 输入 UINT8、权重 INT8；先 qx-zx（范围 [-255,255]，需要 signed 9 bit），权重 zp=0，因此乘积至少 signed 17 bit 足够，累加 signed INT32。
- 每输出通道 bias_scale[c]=float64(sx)×float64(sw[c])；b32[c]=RNE(float64(bias_float32[c])/bias_scale[c])，检查 INT32 范围。实值 Bias 舍入误差和全幅累加上界逐层记录。
- Conv：acc32[c,y,x]=b32[c]+Σ((qx-zx)×qw)。越界像素以 qx=zx 补零。若该层有 ReLU，在 acc 域 max(acc,0)，再重缩放。
- Conv 比例 r[c]=sx×sw[c]/sy，frexp 得到 mantissa×2^exponent；M[c]=RNE(mantissa×2^30)，shift[c]=30-exponent。实际比例为 M×2^-shift。乘子 INT32、shift INT32 元数据，acc×M 使用 signed INT64，静态上界已检查。
- 对 signed product 作**一次**带符号 round-to-nearest ties-to-even 右移，然后加 zy，最后 saturate 到 [0,255]。负数先取绝对值判断余数/半值/奇偶，再恢复符号；不能用无条件加半后算术右移代替 ties-even。禁止中途饱和/截断。
- Add：固定同一 Q31 域，Mi=RNE((si/sout)×2^31)；tmp64=(qa-za)×Ma+(qb-zb)×Mb，先相加，后一次 RNE(tmp64/2^31)，加 zout 并 saturate。不能分别舍入两支再相加。两支乘子均已验证可放 INT32，和使用 INT64。
- Resize：asymmetric/floor 整数地址索引，保持原 scale/zp，不再次改变量化数值。
- I/O：规范核心从 input_quantized_u8 到 output_integer_uint8，无浮点运算。FP32 RGB 预处理、sigmoid、去 padding、原图双线性放大在软件 host；若要迁入 FPGA，需另立协议和数值规范，不能隐含把它们算作 integer-v1 CNN 已支持。

`handoff_integer.py` 的运行核心仅依赖 NumPy，整数卷积使用 INT64 精确 oracle 检查最终 INT32，所有 stage 输出及 acc 均可与 RTL 精确比较。向量生成会同时保存生产 ORT QDQ 和 integer-v1 结果，以实测记录差异；不能拿 QDQ 某层差一个 LSB 就直接宣判 RTL 错误，也不能对 integer-v1 放宽逐位验收。

## 5. 容量、内存、精度和实测性能

### 5.1 精确容量与计算

- 训练可训练参数 32,524 = Conv weight 31,152 + BN gamma/beta 1,280 + 原 project/project2/head bias 92。
- 导出折叠后 31,152 个 INT8 权重 = 31,152 B；732 个折叠 Bias = 2,928 B INT32。原 BN 的均值/方差 buffer 不是训练参数计数的一部分。
- 标准 weight+bias 合计 34,080 B（33.28125 KiB）；不含 per-channel scale/zp、乘子/移位、字节对齐或控制表。实际 ONNX 容器 QDQ {summary['qdq_onnx_bytes']:,} B，FP32 {summary['fp32_onnx_bytes']:,} B；不能用容器大小当作纯权重容量。
- 最大单个 UINT8 特征图 {summary['largest_uint8_feature_bytes']:,} B；s01=491,520 B、s03=215,040 B 两个跳连同时保留需 706,560 B。
- 非 alias、输出先分配再释放已消费输入的 stage 调度，UINT8 活跃特征峰值 {summary['peak_live_uint8_features_bytes']:,} B；再为当前 Conv 分配完整 INT32 acc，峰值 {summary['peak_features_plus_full_int32_accumulator_bytes']:,} B。逐阶段生命周期在 architecture_summary.json。可用 tile 局部 acc 降低峰值；这些不是 PG2L200H BRAM 实测或容量承诺。
- 未融合逻辑特征/权重/bias 读写账本合计 {summary['logical_unfused_io_bytes_per_frame']:,} B/帧，不等于外存实际流量。输出 UINT8 logits 245,760 B；若四 mask 每像素每通道 1 bit，则紧凑负载 30,720 B/帧，bit packing/对齐尚未约定。

### 5.2 已验证模型指标（不是 FPGA 指标）

| 数据域 / 指标 | 第二阶段 FP32 ONNX | 最终 INT8 QAT ONNX | QAT-FP32 |
|---|---:|---:|---:|
{chr(10).join(metrics)}

BDD 是 10,000 张官方 val held-out，模型选择另用内部验证。RLMD 是 439 张分组内部验证，无额外独立测试集；不能称为 RLMD 独立测试精度。分割指标在 320×192 letterbox 网格，padding 排除。BDD crosswalk 和 RLMD drivable 无有效真值，不报告这些项精度。最大关键指标下降约 0.74 pp，通过 2 pp 软件门槛；对照包含 QAT 微调，不是固定相同权重下纯量化误差。

RTX 4060 Laptop GPU FP32 batch1 forward 30 warmups/200 runs p50/p95=1.324/2.733 ms。视频 1349 帧同输入 CPU ORT 1.23.2、4 intra/1 inter：FP32 mean/p50/p95=1.379/1.321/1.842 ms；QAT=8.691/8.495/10.528 ms。CPU 为 i9-13980HX。预处理、视频编码、数据传输不在 model forward 数值内，完整流程另有日志。此 CPU 上 INT8 更慢，不能推断 FPGA 频率/吞吐。

### 5.3 本次整数迁移的真实差异（6 个样本）

| 样本 | 帧（0 起） | 秒 | 场景 | integer-v1 / 生产 QDQ mask 一致率（模型网格） | 最大 logit 差 |
|---|---:|---:|---|---:|---:|
{vector_table}

以上一致率包含大量背景，没有密集真值，不是准确率，也不能替代全验证集迁移验收。本次没有对 integer-v1 重新跑 BDD/RLMD 全集，因此不得把上一节精度直接冠给 integer-v1/RTL。交接方必须在板端数值逐层一致后另跑同一数据划分，尤其检查少数类。

视频定性失败：白车尾部、道路方向箭头和树木/建筑背景出现 crosswalk 误检；真实斑马线有碎片/漏检，虚线可能连成粗带并局部跳变。FP32 也有类似问题。`07_性能与可视化` 和真实向量有对照；不因硬件对齐成功就宣称语义质量达到了道路部署要求。

## 6. 标准文件格式与权重加载

统一规范是 **小端、连续无 padding、无自动 alignment**。NPY 提供 shape/dtype 可读头，BIN 提供纯 payload；两者数据一致。

- 特征图：NCHW，byte_offset=(((n×C+c)×H+y)×W+x)，UINT8。
- Conv 权重：OIHW，byte_offset=(((o×(Cin/groups)+i)×Kh+ky)×Kw+kx)，INT8 two's-complement。Depthwise [C,1,3,3]，输出通道与组一致，无核翻转。
- `weights.oihw.int8.bin`：按 s01,s02,…,s18 的 Conv 顺序连接，准确 offset/长度/shape 在 weights.offsets.json；无间隙，无 DDR 地址含义。
- Bias：输出通道连续 INT32 little endian；scale 数组 float32 little endian；权重 zp INT8；乘子和右移量 INT32 little endian。JSON 同时保存对应数值和文件 SHA256。
- 验证 mask：四通道 NCHW UINT8 0/1；PNG 每通道 0/255。不是 argmax 类别图。核心 output_integer_uint8 是量化 logits，不是 mask 或概率。

布局转换工具明确接受 OIHW NPY，支持 OIHW/OHWI/HWIO 输出（分组 Conv 的 I 始终是 Cin/groups）；不包含任何猜测的 RTL lane 打包、通道补齐或 burst 对齐。请 FPGA 方确认格式后新增单独转换目标，并做往返校验，不要修改本标准 golden。

```powershell
python 06_参考推理与转换工具/handoff_integer.py convert-weight --input 02_模型与权重/standard_weights/s01.weight.npy --output ../stem_ohwi.bin --layout OHWI
```

## 7. 推荐 FPGA 架构与实现顺序

建议先做可复用 INT8/UINT8 MAC 阵列、Depthwise 窗口模块、Pointwise 分块、INT32 Bias/累加、INT64 固定重缩放、ReLU、双输入尺度对齐 Add 和最近邻地址重复模块。CPU 或独立控制模块负责预处理/输出可视化。所有建议是起点，非对 PG2L200H 资源的实测结论。

数据流优先采用 tile 内 output-stationary 累加；1×1 权重/输入按通道 tile 复用，3×3 用行缓冲。Depthwise 采用通道独立流水；同一 MAC 阵列分时复用 14 Conv，先保证功能再增加并行度。跳连存外部 RAM 或专用缓存，确定生命周期后才能覆盖。卷积+ReLU+Requant 可融合写回 UINT8，Resize 可以地址广播到 Add/后继计算，减少中间 DDR；每次融合仍需保持相同舍入边界。

权重标准 blob 很小，可在初始化通过 host/DMA 装入片上或可寻址权重 RAM，是否常驻由 BRAM/DSP 预算确定。激活峰值远大于纯权重，需要先确认板上实际外存容量/带宽/接口；本包不假设已有 DDR 控制器或寄存器地址。

84,718,080 MAC×目标 FPS 是最低有效算力。示例 30 FPS 要求 2.5415424 GMAC/s；若设计选择 200 MHz，则至少 12.707712 个有效 MAC/cycle，16-lane 理想上限约 37.77 FPS，尚未扣气泡、访存、Depthwise 利用率和控制开销。**200 MHz、16 lane 均只是估算示例，不是确认时钟或资源配置。**

建议按以下顺序验收：

1. 在解压目录运行包自检，确认模型/数据 SHA256 和 PC 参考可复现。
2. 实现 s01 单层标准 Conv，用同一 sample 的 input_quantized_u8 对齐 s01 accumulator INT32；再对齐 s01 UINT8（必须 exact）。
3. 对齐 s02 Depthwise、s03 Pointwise，然后其余编码器；每层先 acc 后重缩放输出。
4. 单独验证 s09 nearest、s10 Add 和 skip；保留原量化域，测试负值和非零 zp；再做第二残差。
5. 对齐 head、最后 Resize，六个样本所有 20 stage UINT8 和 14 acc32 exact；比较脚本 tolerance 默认 0，故意改单字节必须失败。
6. RTL 仿真后做综合/时序/资源报告，再板端跑同一向量及端到端流；记录硬件 revision、bitstream hash、频率、batch、IO/纯核心延迟、吞吐、功耗（如具备测量）。
7. 板端与 integer-v1 数值完全一致后，执行代表性视频和原 BDD/RLMD 验证。integer-v1 相对生产 ONNX 的完整指标尚未测；需双方约定迁移精度容忍值，不可擅用 6 样本背景一致率验收语义质量。

没有 RTL 输出时不填“通过”硬件验收。参考程序在数学规则上确定 exact，若 FPGA 采用另一种舍入/布局，必须形成 v2 规范并重新生成对应向量，不能静默改 tolerance 掩盖差异。

## 8. 验证包、运行方式与可复核性

6 个样本来自用户提供的真实 MP4，不包含完整训练集或原 MP4；包括正常道路、实线、虚线、斑马线、遮挡和转弯/反射困难场景。每个 sample.json 记录源视频 hash、帧序号/时间、letterbox 几何、全部 BIN/NPY 的 dtype/shape/scale/zp/hash；保存 20 个生产 QDQ UINT8 边界、20 个 integer-v1 UINT8 边界、14 个 acc32、原图、四 mask 与可视化。没有伪造 GT。

在包根目录（Python 命令为队友当前环境，不依赖原电脑绝对路径）：

```powershell
python -m pip install -r 06_参考推理与转换工具/requirements.txt
python 06_参考推理与转换工具/handoff_verify.py --root .
python 06_参考推理与转换工具/handoff_integer.py infer --root . --input 05_硬件验证向量/000_crosswalk_front/input_quantized_u8.npy --output ../integer_recheck_000
python 06_参考推理与转换工具/handoff_integer.py compare --expected 05_硬件验证向量/000_crosswalk_front/integer/s01.npy --actual ../rtl_s01.bin
python scripts/infer.py --model model/model_int8.onnx --input golden/input_000.bin --output ../onnx_recheck_000.bin --config config/model_config.json
```

整数推理仅需 NumPy；完整自检额外依赖 ONNX、ORT、PyTorch，恢复 FX 优先使用包中实测版本。BIN compare 以 expected NPY 的 dtype/shape 解析 RTL 原始字节，小端 NCHW，误差输出含首个坐标。输出目录/新文件默认拒绝覆盖。

实测生成环境：Python {versions['python'].split()[0]}；PyTorch {versions['torch']}；NumPy {versions['numpy']}；ONNX {versions['onnx']}；ORT {versions['onnxruntime']}；OpenCV {versions['opencv']}。GPU 不是推理/自检依赖；FX QAT 是受 PyTorch 版本影响的恢复格式，优先 ONNX 和标准二进制进行硬件开发。

SHA256SUMS.txt 覆盖包内所有 payload、工具、文档和 SELF_CHECK.json；它不包含自身 hash，也不包含后生成的 FILELIST.json（避免自指）。FILELIST.json 逐项列文件/字节/hash。ZIP/RAR 的 SHA256 由外部 DELIVERY.json 保存。压缩包解压后再次运行完整自检，不能只检查“压缩成功”。

## 9. 当前成果、限制与双方接口

已交付：冻结权重、最终 QDQ/FP32 ONNX、实际完整算子/shape/参数/计算量、权重标准二进制、全部量化参数、INT32 bias 和确定的整数规则、可执行参考、真实向量、逐层比较工具、软件精度/视频问题证据。

待 FPGA 方确定并回传：

| 待协商项 | 本包已固定 | FPGA 端需确认 |
|---|---|---|
| 外部输入/输出链路 | 逻辑 NCHW、形状、数值、标准 BIN | AXI/自定义流/DMA/PCIe 是否使用、物理总线位宽、握手、SOF/EOF、backpressure |
| 地址与控制 | 离线权重 blob offset，不含硬件地址 | 寄存器、descriptor、base address、start/done/error、超时和复位 |
| 存储布局 | 无 padding OIHW/NCHW little endian | 片上 bank、lane/interleave、通道补齐、burst/字节对齐、DDR/BRAM 容量与带宽 |
| 数值内核 | integer-v1 RNE、饱和、zp、INT32/64 | DSP 分解、乘法/移位实现；如不接受规则须版本化并重做向量 |
| 并行与时序 | 精确 MAC 和生命周期账本 | 实际时钟、DSP/BRAM/LUT 预算、tile、并行 lane、流水/复用 |
| 预后处理边界 | host letterbox/LUT；核心量化 logits | RGB/BGR 来源、host/FPGA 哪侧预处理；输出 logits 或 bit mask、bit packing、是否传回几何 |
| 软件接入 | 独立验证工具，不触现有工程 | 帧 ID/模型版本、错误码、缓存一致性、现有工程怎样新增可选接口 |
| 验收目标 | 六向量整数逐层 exact | 目标 FPS、端到端延迟/功耗、迁移精度允许值和场景验收 |

不假设队友第一套模型的协议可复用，不修改其 RTL/权重。模型组负责冻结模型/量化规范/软件 golden；FPGA 组负责转换实现、RTL 对照、综合和板测，返回 mismatch 的 stage、shape、dtype、输出 BIN 和对应 sample ID 即可定位。

RLMD 官方资料未明确许可，公开副本 MIT 声明未获原作者确认；本交付不包含 RLMD 原始训练数据，不能用本包证明数据或训练权重具有任意再分发许可。当前真实向量使用用户提供的视频，仅供项目内部硬件验证。

## 10. 交接包附件目录

- `01_技术文档/`：本文、权重格式和双方待确认接口。
- `02_模型与权重/`：固定最终模型/检查点、标准 INT8 blob 和分层权重。
- `03_网络结构与算子/`：93 ONNX 节点、20 integer stage、MAC/流量/内存、Mermaid 完整图。
- `04_量化参数/`：20 激活域、732 通道参数 CSV、全部 scale/zp/bias/乘子二进制和 integer_contract.json。
- `05_硬件验证向量/`：六个真实场景及两个计算路径的逐层 golden。
- `06_参考推理与转换工具/`：NumPy integer oracle、ORT 跟踪、自检、checkpoint 恢复和生成脚本来源。
- `07_性能与可视化/`：真实验证指标、训练曲线、视频审阅与失败案例；不附巨大原始训练数据/视频缓存。
- `model/config/golden/scripts`：用户附件约定的便捷入口，实际 model_name 为 Candidate-plus；不是 tiny_unet，signed/unsigned/padding 按本模型填写。
'''
    technical=repo/'docs/FPGA_CandidatePlus_技术交接文档.md'
    technical.write_text(document,encoding='utf-8')
    shutil.copy2(technical,root/'01_技术文档'/technical.name)
    (root/'01_技术文档/WEIGHT_FORMAT.md').write_text(document.split('## 6.')[1].split('## 7.')[0],encoding='utf-8')
    (root/'01_技术文档/INTERFACES_TBD.md').write_text(document.split('## 9.')[1].split('## 10.')[0],encoding='utf-8')
    return document
