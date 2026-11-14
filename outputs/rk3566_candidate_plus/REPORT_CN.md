# Candidate-plus 在鲁班猫 RK3566 NPU 的部署与实测

日期：2026-11-14
模型：PR #16 第二阶段 Candidate-plus（32,524 参数，RGB 320×192，输出 4×192×320）
板卡：EmbedFire LubanCat-1 / RK3566，4×Cortex-A55，3.8 GiB RAM，NPU 600 MHz

## 0. 结论

Candidate-plus 已真实部署到 RK3566 NPU，而不是 PC 模拟：

- FP16 RKNN 可运行，单次 NPU 推理平均 **18.096 ms / 55.26 FPS**。
- 最终采用的 QAT INT8 RKNN 单次 NPU 推理平均 **9.228 ms / 108.37 FPS**，P50 9.086 ms，P95 10.964 ms。
- 1080p30 完整视频流水线（解码、预处理、NPU、后处理、H.264 编码）实测 **4.063 FPS**；瓶颈是 CPU 后处理（约 165 ms/帧），不是 NPU。
- QAT INT8 与软件 FP32 的像素级 mask 一致性好坏取决于类别；最大像素分歧约 0.34%（可行驶区域），虚线仅有 0.01% 像素分歧。详细数字见第 4 节。
- 与现有 YOLO P2 的并发测试见第 6 节；单 NPU 上两个模型会串行共享，Candidate-plus 的 9.2 ms 约为 YOLO P2 单帧 41.6 ms 的 22%。

最终模型：`models/candidate_plus_int8_qat.rknn`
最终视频：`candidate_plus_rk3566_overlay.mp4`
同帧对照（最终视频第 674 帧）：`eval/accuracy/figures/same_frame_final_video_f00674.jpg`

精度对照（33 张真实图片）：`eval/accuracy/figures/same_frame_fp32_vs_rk3566_int8.jpg`

## 1. 实验环境与边界

| 项目 | 实测值 |
| --- | --- |
| 板卡 | LubanCat-1，RK3566，aarch64 |
| NPU 驱动 | 0.8.2 |
| 板端 runtime | librknnrt 2.3.2 |
| 板端 Python | 3.8.10 |
| RKNN Lite2 | 2.3.2 |
| NPU 当前频率 | 600 MHz（未改 governor） |
| 转换环境 | WSL Ubuntu，Python 3.10，rknn-toolkit2 2.3.2，onnx 1.16.1 |
| 板端连接 | USB 网络 192.168.137.2 |
| 板端存储 | eMMC 根分区已 100% 满；本次所有模型/视频/中间文件放在 `/tmp`（2 GiB tmpfs），未修改系统分区 |

没有修改正在运行的 YOLO P2 模型、比赛主程序、Clash、防火墙、NPU 驱动或系统服务。测试输入为真实道路视频/图片，不是合成图。

## 2. RKNN 转换

### 2.1 输入图与预处理契约

转换使用 `outputs/fpga_road_seg_stage2_qat/exports/model.preprocessed.onnx`（静态 Resize，FP32，输入 `rgb [1,3,192,320]`，输出 `logits [1,4,192,320]`）。它与原始 `model.fp32.onnx` 数值一致；PyTorch FP32 与 ONNX FP32 的最大 logit 误差为 **3.8e-05**（33 张真实图片）。

预处理严格按训练代码：

1. OpenCV 读取 BGR，先转 RGB；
2. 等比 letterbox 到 320×192，填充 RGB(114,114,114)，无 crop；
3. RGB/255，mean=0，std=1；
4. 输出四个独立 sigmoid，阈值 ≥0.5，不 argmax、不 softmax。

RKNN 配置为 `mean_values=[[0,0,0]]`、`std_values=[[255,255,255]]`、`quant_img_RGB2BGR=False`。板端传入 NHWC uint8 RGB；颜色顺序已用同一帧对比验证：RKNN FP16 与 ONNX RGB 的 logits MAE 为 **0.0081**，与 ONNX BGR 为 **1.135**，确认走 RGB。

### 2.2 FP16 转换

`models/candidate_plus_fp16.rknn`，136 KiB。算子全部被工具链接受，包括 Conv、Depthwise、Pointwise、Add、Nearest Resize、四通道输出。板端输出 shape `[1,4,192,320]`、float32、有限值。

### 2.3 FP32 PTQ INT8

`models/candidate_plus_int8.rknn`，117 KiB。使用 192 张真实道路帧，来自 3 段原始视频；校准数据保存为 **NCHW uint8 RGB** `.npy`（RKNN Toolkit 的校准 loader 要求 NCHW），部署时仍使用 NHWC uint8。量化算法 `normal`，per-channel。

转换日志提示 `onnx::Conv_163` 有 outlier（abs max 10.313），这是后续精度排查的重点之一。

### 2.4 QAT QDQ ONNX 直接转换

`qat.int8.qdq.onnx` 可以被 RKNN-Toolkit2 2.3.2 加载，但工具链会把它识别为 QAT 模型并直接生成 INT8 运行时；`do_quantization=False` 并不等于 FP16。默认 `optimization_level=3` 会提示可能影响精度，改为 **optimization_level=2** 后：

- 模型：`models/candidate_plus_int8_qat.rknn`（117 KiB，opt2）
- 33 张真实图片上，板端输出与 QAT ONNX 的 logits MAE **0.151**，最大绝对误差 1.63
- 同一测试集上，QAT ONNX 与 FP32 的 logits MAE 为 0.331（含 QAT 微调差异）
- RKNN QAT 与 QAT ONNX 的 mask IoU：可行驶 0.909、实线 0.894、虚线 0.582、斑马线 0.830

因此最终采用 QAT 直接转换的 INT8 模型；FP32-PTQ INT8 保留为对照，不作为交付模型。

## 3. 纯 NPU 性能（板端实测）

测试方法：RKNNLite `init_runtime()`，同一张 320×192 NHWC uint8 输入，预热 10 次，连续 100 次纯 `inference()`；延迟为 Python `perf_counter` 包围的单次调用，不含预处理/后处理/视频 I/O。

| 模型 | 平均 ms | P50 ms | P95 ms | P99 ms | FPS（1/平均） | 进程 CPU（单核%） | NPU 负载 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 | 18.096 | 17.664 | 20.448 | 23.770 | 55.26 | 78.83 | 100% @600 MHz |
| INT8（FP32 PTQ） | 9.213 | 9.014 | 11.129 | 11.946 | 108.54 | 61.49 | 100% @600 MHz |
| **INT8（QAT，最终）** | **9.228** | **9.086** | **10.964** | **11.771** | **108.37** | **61.04** | **100% @600 MHz** |

进程 RSS：init 后约 45.5 MiB，100 次后约 85.9 MiB。板端系统内存约 3.8 GiB，测试时可用约 3.2 GiB。

原始日志：`logs/fp16_bench.json`、`logs/int8_bench.json`、`logs/qat_opt2_bench.json`。

## 4. 精度对照

测试集：33 张真实图片（PR #16 `comparisons/` 中的 BDD/RLMD 真实帧 + 另一段视频的 8 帧），全部按同一 letterbox 契约预处理。对比：

- PyTorch FP32 checkpoint
- ONNX FP32
- RKNN FP16
- RKNN INT8（FP32 PTQ）
- RKNN INT8（QAT 直接转换）
- QAT ONNX（QDQ）

### 4.1 logits/概率误差（33 张均值）

| 对比 | logits MAE | logits max abs | 概率 MAE |
| --- | ---: | ---: | ---: |
| PyTorch FP32 vs ONNX FP32 | 3e-06 | 3.8e-05 | — |
| RKNN FP16 vs ONNX FP32 | 0.0081 | 0.086 | 3.4e-05 |
| RKNN INT8 PTQ vs ONNX FP32 | 0.5634 | 5.464 | 0.00285 |
| QAT ONNX vs ONNX FP32 | 0.3310 | 4.055 | 0.00173 |
| RKNN INT8 QAT vs QAT ONNX | 0.1509 | 1.634 | 0.00072 |
| RKNN INT8 QAT vs ONNX FP32 | 0.3321 | 4.092 | 0.00170 |

### 4.2 原生分辨率 mask 一致性

IoU/F1 是“相对参考模型预测”的一致性，不是数据集真值精度。虚线在画面中像素极少，少量边界变化就会显著拉低 IoU，因此同时给出像素分歧比例。

| 对比 | 可行驶 IoU / 分歧 | 实线 IoU / 分歧 | 虚线 IoU / 分歧 | 斑马线 IoU / 分歧 |
| --- | ---: | ---: | ---: | ---: |
| RKNN FP16 vs FP32 | 0.9937 / 0.007% | 0.9975 / 0.001% | 0.9746 / 0.0004% | 0.9923 / 0.0035% |
| RKNN INT8 PTQ vs FP32 | 0.6330 / 0.540% | 0.7682 / 0.090% | 0.3573 / 0.019% | 0.5273 / 0.244% |
| QAT ONNX vs FP32 | 0.7775 / 0.348% | 0.8393 / 0.063% | 0.4628 / 0.0096% | 0.6508 / 0.178% |
| **RKNN INT8 QAT vs QAT ONNX** | **0.9090 / 0.130%** | **0.8936 / 0.025%** | **0.5824 / 0.0048%** | **0.8302 / 0.078%** |
| **RKNN INT8 QAT vs FP32** | **0.7789 / 0.337%** | **0.8266 / 0.058%** | **0.5188 / 0.0099%** | **0.6480 / 0.176%** |

解释：

- FP16 几乎逐像素复现 FP32，说明图结构、颜色顺序、输出排列和阈值语义正确。
- FP32-PTQ INT8 的 logits 误差和 mask 分歧明显大于 QAT；虽然能跑，但不采用。
- QAT 直接转换与 QAT ONNX 的差异很小（像素分歧 ≤0.13%），说明 RKNN 对 QAT 图的处理是可用的。
- “QAT vs FP32”的差异包含 QAT 微调本身，不是纯量化误差；PR #16 在 BDD/RLMD 上的软件指标最大下降为 0.74 个百分点，才是数据集级精度结论。

原始数据：`eval/accuracy/accuracy.json`、`eval/accuracy/accuracy.csv`、`eval/accuracy/figures/`。

## 5. 完整视频流水线（1080p30）

视频：`D:\BaiduNetdiskDownload\紫光同创杯赛题三参考素材\video_20260929_112507.mp4`，1920×1080，30 FPS。

处理链：OpenCV 解码 → RGB letterbox → RKNN QAT INT8 → sigmoid/去 padding/双线性还原/阈值 → 绿/红/青/洋红叠加 → ffmpeg libx264 编码（preset ultrafast，CRF 20，2 threads，保留音频）。

### 5.1 15 秒预览片段（450 帧）

| 阶段 | 平均 ms | P50 | P95 |
| --- | ---: | ---: | ---: |
| 解码 | 20.868 | 18.284 | 29.346 |
| 预处理 | 7.903 | 7.697 | 9.085 |
| NPU 推理 | 9.671 | 8.968 | 13.169 |
| 后处理（还原+叠加+图例） | 165.471 | 163.872 | 194.824 |
| 编码/写入 | 16.653 | 16.317 | 21.420 |
| 单帧合计 | 220.613 | 217.709 | 248.006 |

整段 wall-clock：112.629 s / 450 帧 = **3.995 FPS**。
日志：`logs/video_15s.json`、`logs/video_15s.csv`。
预览：`candidate_plus_rk3566_overlay_15s.mp4`。

### 5.2 完整视频（1349 帧）

| 指标 | 实测 |
| --- | ---: |
| 处理帧数 | 1349 |
| 总 wall time | 332.017 s |
| 完整流水线 FPS | **4.063** |
| 纯 NPU 推理平均 | 9.524 ms |
| 后处理平均 | 163.258 ms |
| 编码/写入平均 | 16.634 ms |

最终视频：`candidate_plus_rk3566_overlay.mp4`。
日志：`logs/video_full2.json`、`logs/video_full2.csv`（最终 1349 帧版本）。

瓶颈明确：NPU 只占约 9.2 ms，后处理占约 165 ms。当前 Python/OpenCV 在 1080p 上做 4 通道概率双线性还原和逐像素叠加，是完整视频 FPS 的主要限制。不是模型或 NPU 算力不足。

## 6. 与 YOLO P2 的对比/并发

现有比赛主模型为 B05N P2 十类 raw-head INT8（512×288），板端路径：
`/home/cat/cityscapes_arch_compare_20261027/models/b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn`。

在同一进程里同时加载两个 RKNNLite runtime，先各自单独跑 50 次，再交替跑 50 对。P2 输入用 512×288 uint8 零张量，只测 NPU 时间，不测检测精度；Candidate-plus 输入用真实 letterbox 帧。

| 场景 | 平均 ms | P50 ms | P95 ms | 吞吐 |
| --- | ---: | ---: | ---: | ---: |
| P2 单独 | 41.573 | 39.659 | 49.497 | 24.05 FPS |
| Candidate-plus 单独 | 8.704 | 8.651 | 9.780 | 114.89 FPS |
| 交替一对（P2→Candidate） | 47.551 | 47.091 | 54.297 | 21.03 对/s（42.06 次推理/s） |
| 交替中的 P2（单次段） | 38.828 | 38.367 | 44.904 | — |
| 交替中的 Candidate-plus（单次段） | 8.722 | 8.637 | 10.081 | — |

进程 CPU 约 97.53%（单核口径），RSS 从 83.1 MiB 增至 94.6 MiB。结论：

- 两个模型可以在同一 NPU 上同时加载、交替运行；RK3566 只有一个 NPU core，推理是串行的，Candidate-plus 的 8.7 ms 基本是叠加在 P2 的 41.6 ms 上。
- 两者都连续满速时，P2 从 24.05 FPS 降到约 21.03 FPS，Candidate-plus 也从 114.9 FPS 降到约 21.0 FPS；总推理吞吐 42.06 次/s，P2 仍是主耗时项。
- 若比赛主程序按现有 10 Hz 跑 P2，再以 10 Hz 跑 Candidate-plus，NPU 占用约 `10×(41.6+8.7)=503 ms/s`，约 50%，可行。
- 若要求 Candidate-plus 30 FPS 且 P2 10 Hz，则 NPU 需求约 `10×41.6 + 30×8.7 = 677 ms/s`，超过实时预算，不可行。

原始日志：`logs/concurrent_p2.json`。

## 7. 问题与排查记录

1. **校准时 npy 形状错误**：RKNN Toolkit 的 `.npy` 校准 loader 要求 NCHW（`[1,3,192,320]`），不是运行时的 NHWC。首次用 NHWC 校准报 `shape (1,192,320,3) is wrong, expect nchw`；改为 NCHW 后成功。部署输入仍是 NHWC uint8。
2. **FP32-PTQ INT8 精度偏低**：logits MAE 0.563，可行驶/虚线 mask 分歧明显。检查了 RGB/BGR（RGB 明显更优）、输出排列（`[1,4,192,320]`，四通道语义正确）、输入 scale（0–255/255）后，确认不是预处理错误，而是 FP32 图直接 PTQ 对 outlier 敏感。
3. **QAT ONNX 直接转换**：RKNN 识别 QAT 后会生成 INT8 运行时，`do_quantization=False` 不是 FP16；默认 optimization_level=3 有精度警告，改为 2 后与 QAT ONNX 的误差下降到 MAE 0.151，最终采用。
4. **中文字体缺失**：板端 OpenCV 只有 Hershey 字体，中文图例会变成 `?`；视频图例改为英文（Drivable/Solid lane/Dashed lane/Crosswalk），颜色语义不变。
5. **板端 eMMC 满**：根分区 100% 满，所有大文件放 `/tmp` tmpfs；没有删除系统文件或比赛程序。
6. **1080p 后处理慢**：初版逐类 float 布尔索引叠加约 504 ms/帧；改为单次 HWC 双线性还原 + bbox 限制的索引叠加后约 165 ms/帧，仍是最慢阶段。

## 8. 交付物

```
outputs/rk3566_candidate_plus/
  models/
    candidate_plus_fp16.rknn          # 非量化 FP16
    candidate_plus_int8.rknn          # FP32-PTQ INT8（对照，不推荐）
    candidate_plus_int8_qat.rknn      # 最终 QAT INT8
  calibration/                        # 192 张真实道路校准 npy + manifest
  logs/                               # 转换、benchmark、视频流水线原始 JSON/CSV
  eval/
    inputs/ originals/                # 33 张精度对照输入
    rknn_fp16/ rknn_int8/ rknn_qat/   # 板端原始输出 npy
    accuracy/                         # accuracy.json/csv、同帧对照图
  candidate_plus_rk3566_overlay.mp4   # 最终完整视频（1349 帧，1920×1080，30 FPS，含音频）
  results_summary.json                # 小型可引用结果汇总
  REPORT_CN.md
```

代码：`deployment/rknn/candidate_plus/`（转换、校准、板端 benchmark、板端视频、PC 精度评估、并发测试）。

## 9. 直接回答

1. **是否成功部署？** 是。FP16 和 QAT INT8 都在真实 RK3566 NPU 上完成推理；最终采用 QAT INT8。
2. **纯 NPU FPS？** 9.228 ms/次，**108.37 FPS**（P50 9.086 ms，P95 10.964 ms，600 MHz，NPU 100%）。
3. **完整视频 FPS？** 1080p30 完整链路 **4.063 FPS**；15 秒预览为 3.995 FPS。瓶颈是 CPU 后处理，不是 NPU。
4. **INT8 相比 FP32 的精度损失？** 相对 FP32 预测：可行驶 mask 分歧 0.337%、实线 0.058%、虚线 0.0099%、斑马线 0.176%；相对 QAT ONNX 的分歧更小（≤0.13%）。数据集级精度以 PR #16 的 BDD/RLMD 软件指标为准，最大下降 0.74 个百分点。
5. **与 YOLO P2 相比？** 实测 B05N P2（512×288）单次 41.573 ms / 24.05 FPS；Candidate-plus QAT INT8 9.228 ms / 108.37 FPS，单次约快 **4.5 倍**。交替运行时 P2 38.828 ms、Candidate 8.722 ms，一对 47.551 ms。
6. **能否与 YOLO P2 同时运行？** 可以。同一进程双 runtime 已实测；单 NPU 串行，Candidate-plus 只增加约 8.7 ms/帧。10 Hz P2 + 10 Hz Candidate 约占用 50% NPU，可行；30 FPS Candidate + 10 Hz P2 超出实时预算。
7. **是否继续优化模型？** 不建议优先优化 NPU 模型：9.2 ms 已很快，NPU 已 100% 负载。优先优化 1080p 后处理（C++/RGA/减少全帧浮点还原）和模型质量（斑马线碎片、可行驶区域误检），而不是继续压缩 CNN。

## 10. 限制

- 精度对照是“模型间一致性”，不是重新在 BDD/RLMD 上跑真值评测；数据集级结论引用 PR #16 已有结果。
- 视频叠加保留原始分辨率/帧率，但为显示做了 0.45 alpha 混合；四通道仍是独立 sigmoid，未做时序平滑、补线或后处理修掩码。
- 未改 NPU governor；600 MHz 是板端当前默认频率，若解锁 900 MHz 可能更快，但本次不做未经验证的频率改动。
- 板端 eMMC 已满，交付视频/模型在仓库输出目录，板端 `/tmp` 副本在重启后消失。
