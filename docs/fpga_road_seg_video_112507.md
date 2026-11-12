# 第二阶段 Candidate-plus 全视频逐帧验证

源视频：D:\BaiduNetdiskDownload\紫光同创杯赛题三参考素材\video_20260929_112507.mp4。完整 1349 帧，1920×1080，30/1 FPS，保持顺序，无跳帧。
输出三段 H.264 / yuv420p MP4，逐段完整解码检查帧数、FPS 和分辨率一致，原音频复制保留。

- road_seg_overlay.mp4：最终 INT8 QAT 四类半透明叠加。
- road_seg_comparison.mp4：原视频、QAT 叠加、四类独立 Mask，共六面板；每面板等比缩放，整体保持原视频分辨率。
- road_seg_fp32_int8_comparison.mp4：原视频、第二阶段 FP32、最终 QAT、阈值差异；差异不是有真值的错误。

绿色=可行驶区域；红色=实线；青色=虚线；洋红=斑马线区域。独立输出可重叠，显示时后者覆盖前者。
RGB、320×192 letterbox（填 114）、/255、sigmoid、去 padding 后双线性还原概率、≥0.5。
无时序平滑、形态学、补线、ROI 屏蔽或额外规则修饰。Mask 边缘放大是训练配套几何还原。

## 模型身份

INT8 ONNX：C:\Users\JingyuWang\adas-edge-issue15\outputs\fpga_road_seg_stage2_qat\exports\model.int8.qdq.onnx

INT8 SHA256：c7a21a9a3f1621ac5cbcbbb9030587251a340cb435413b3c086ff1c1a6d137a6

QAT checkpoint：C:\Users\JingyuWang\adas-edge-issue15\outputs\fpga_road_seg_stage2_qat\checkpoints\candidate-plus-qat\qat.best.pt

QAT checkpoint SHA256：48e6a2dccf1dc98095b41e895cd9c99f7ab044b3c8a343cb4a364275c6fceffc，最佳第 1 轮。

FP32 ONNX：C:\Users\JingyuWang\adas-edge-issue15\outputs\fpga_road_seg_stage2_qat\exports\model.fp32.onnx

FP32 SHA256：4f3999e6f450cac7ac6d7d1c8c9569cc4d1855eebe865f2d9cee219d8ee3923b。

后端：ONNX Runtime CPUExecutionProvider, 4 intra-op/1 inter-op threads for both FP32 and QAT；不是 RK3566/FPGA 延迟。FP32 是第二阶段优化后、QAT 前基线，差异包含 QAT 微调，不是固定相同权重下的纯量化误差。

## 实测耗时

FP32 模型 mean/p50/p95：1.379/1.321/1.842 ms。
INT8 QAT 模型 mean/p50/p95：8.691/8.495/10.528 ms。
预处理 mean：2.537 ms；还原/分析 mean：194.142 ms；
可视化 mean：101.227 ms；三路编码管道写入/背压 mean：7.891 ms。
编码排空时间逐路记录 summary.json。管道时间不是独立测得的编码计算时间，编码器与生产帧重叠执行。
全流程 504.15 s，2.68 源帧/s（含两模型、三路视频、缓存和完整输出解码核验）；渲染/编码阶段 3.21 源帧/s。
逐帧原始耗时见 inference_per_frame.csv、render_per_frame.csv；原始 float32 概率保存在 cache/probabilities.npy，可独立复核。

## 效果检查

frames/ 保存时间均匀样本以及斑马线面积、底部/上部区域、亮色表面重叠、实虚线变化和量化分歧最大的待检查截图。
这些自动触发不能直接认定仪表台/白车误检或车道闪烁，需要结合画面人工核对。frame_analysis.json 保留完整逐帧统计。
视频没有密集真值，不输出 IoU/F1 或把亮色重叠当作误检率。斑马线监督语义为整个过街区域，含条纹间隙。
人工检查结论会另存 VISUAL_REVIEW.md，区分已观察现象与无法确认的情况。

## 画面结论及编码补充

模型能识别部分真实斑马线，但存在白车尾部、道路箭头、普通路面和树木/建筑背景误检，覆盖还存在破碎和漏检。FP32 也有类似问题；详见 本地 outputs/fpga_road_seg_video_test/VISUAL_REVIEW.md 和 本地 outputs/fpga_road_seg_video_test/failures/README.md。人工审阅为 12 张均匀样本、极值候选和 36 张连续帧，不能等同于逐帧人工验收。

CPU 为 Intel Core i9-13980HX（24 核 / 32 逻辑处理器）。两模型均使用 ORT CPUExecutionProvider，QAT 此处比 FP32 慢；不是 RTX 4060、RK3566 或 FPGA 性能结果。

通过 Windows GetProcessTimes 另行记录三个 ffmpeg 进程的累计 CPU 时间：

| 输出 | 累计 CPU 秒（线程总和） | 进程墙钟秒（含等待帧） |
|---|---:|---:|
| road_seg_overlay.mp4 | 85.797 | 420.232 |
| road_seg_comparison.mp4 | 42.844 | 420.277 |
| road_seg_fp32_int8_comparison.mp4 | 81.281 | 420.356 |

这些 CPU 时间包括 H.264 编码、音频解复用/封装及管道处理；多线程 CPU 累加时间不是播放时间，也不是单线程延迟。编码进程约 420 秒墙钟主要受上游还原/分析和可视化供帧限制，不能全部算成编码计算开销。原始计数见 encoding_process_times.json。所有编码器退出码为 0。

FP32 checkpoint：C:\Users\JingyuWang\adas-edge-issue15\outputs\fpga_road_seg_stage2_qat\checkpoints\candidate-plus\best.pt。

FP32 checkpoint SHA256：f12fac988aa29d7ba68d94af2bd0261a6eacf35b111b4c58141119d40bef6498。

13 项训练数据/第二阶段回归检查通过；独立视频脚本已完成本片 1349 帧真实端到端验证。仅新增实验脚本和文档，未修改稳定工程、模型参数或权重。

# 实际画面检查结论

本次模型能响应真实斑马线和车道结构，但仍有明显语义误检、漏检和相邻帧跳变，不能据此称为道路部署已验收。没有添加时序平滑或规则修饰。

## 已观察的现象

- 斑马线有真实响应：0.00s、12.23s、23.80s、24.50s、40.83s 附近的过街区域有洋红输出，近距离跨越区可出现连片响应；部分位置覆盖不足、边缘外扩或区域破碎。RLMD 的斑马线是整个过街区域，包含条纹间隙，不能把填充条纹间隙本身算作错误。
- 误检明确存在：4.07s 普通路面/阴影附近有洋红斑块；8.17s、12.23s 白车尾部出现洋红；20.40s 道路方向箭头被当作斑马线；42.57s 树木/建筑上方也出现洋红和少量绿色响应。FP32 画面中同样可见这些问题。
- 仪表台：所检查截图中未见仪表台大范围被识别成斑马线。自动底部 ROI 极值在 23.80s，对应近处真实斑马线，不能把这个触发器直接计作仪表台误检。不能推断全视频所有仪表台像素均无误报。
- 反光：普通亮色物体及白车存在混淆；当前视频主要为晴天干燥路面，不能给出湿地反光专项结论，也不能根据亮色重叠统计判断因果。
- 车道：直行时近处车道线有响应，但远处和遮挡位置缺失；部分虚线被连成粗的连续青色带，实/虚线局部出现类别混淆和边界断裂。4.00～4.37s、12.00～12.37s、30.00～30.37s 连续帧拼图显示边缘/局部斑块跳变，包含车辆运动，不能用无运动补偿的 XOR 当作纯闪烁率。

## FP32 与 QAT

两者整体形态接近，主要差别是边缘、孔洞和少量孤立区域；未观察到 QAT 独有的整体输出崩溃。上面的白车、方向箭头、背景误检在 FP32 中也存在，不能全部归因于量化。

按全原图像素作为分母，逐帧阈值 Mask 平均分歧分别为：可行驶区域 0.4066%、实线 0.0714%、虚线 0.1017%、斑马线 0.1427%。大面积背景会稀释这个比例，它不是准确率，也不能证明少数类表现一致。各帧详细值见 frame_analysis.json，汇总见 quantization_video_difference.json。

FP32 是第二阶段优化后、QAT 前的基线，比较同时包含 QAT 微调和量化影响，并非相同权重下的纯数值量化误差。当前 CPUExecutionProvider 上 QAT 比 FP32 慢，不能声称 INT8 已带来部署加速。

## 检查范围及文件

已人工查看覆盖首尾的 12 张均匀时间截图、自动极值候选截图，以及三组各 12 张连续帧拼图。全部 1349 帧均已推理和编码，三段输出均已完整解码校验；画面定性审阅是上述采样范围，并未逐帧人工标注或观看验收整个视频。

- failures/：6 张已确认问题的截图、cases.json 和时间/原因。
- frames/temporal_*：36 张连续帧的三张拼图，无重新推理或平滑。
- frames/*encoded_frame00612.jpg：直接从最终 MP4 解码的画面，可核对中文、颜色和六面板比例。
- README_RESULT.md：模型路径/SHA256、预处理、后端和实测耗时；summary.json 与逐帧 CSV 为原始记录。

本次仅验证现有模型。若后续优化，应先处理方向箭头/车辆/背景的斑马线误检及车道连续性；当前结果支持继续研究和 FPGA 算子联调，不支持宣称四类道路分割已稳定可用。
