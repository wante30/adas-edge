# 第二阶段最终 QAT 全视频检查

在仓库根目录运行，使用安装了 PyTorch、OpenCV、Pillow 和 ONNX Runtime 的 Python，PATH 中需要 ffmpeg/ffprobe：

```powershell
python -u -m experiments.training.fpga_road_seg.video_infer --video "D:/BaiduNetdiskDownload/紫光同创杯赛题三参考素材/video_20260929_112507.mp4" --output outputs/fpga_road_seg_video_test
```

默认模型目录为 `outputs/fpga_road_seg_stage2_qat`。脚本检查 `exports/search.json` 最终选择确为 QAT、最终 ONNX 与 QAT 变体 SHA256 相同，以及 QAT checkpoint 来源与第二阶段 FP32 checkpoint 相符。不会自动训练或下载模型。

复用训练适配器的 RGB、320×192 letterbox（填充值 114）、/255、sigmoid、去除 padding、双线性还原概率、独立通道阈值 ≥0.5。每个输入帧同时执行第二阶段优化 FP32 和最终 QAT ONNX；没有时序平滑、形态学处理、人工补线或区域屏蔽。

三段输出均为原视频分辨率、帧率和帧顺序，libx264 H.264 / yuv420p / MP4 faststart，复制可用的原 AAC 音频：

- `road_seg_overlay.mp4`：最终 QAT 四类半透明叠加和中文图例。
- `road_seg_comparison.mp4`：原视频、QAT 叠加和四个独立 Mask，六面板内等比缩放并留黑边。
- `road_seg_fp32_int8_comparison.mp4`：原视频、第二阶段 FP32、最终 QAT、四通道阈值分歧。

FP32 对照是 QAT 前的优化基线，因此差异包含 QAT 微调，不能解释为固定相同权重下的纯量化误差。两个模型使用相同的 CPUExecutionProvider 和线程设置；模型耗时不代表 FPGA 或 RK3566 延迟。

`model_and_video.json`、`summary.json` 保存模型/权重/视频 SHA256、路径、预处理、编码及耗时证据。`inference_per_frame.csv` 记录同步 ORT 调用耗时，`render_per_frame.csv` 分开记录还原/分析、可视化和三路编码管道写入。编码管道耗时包括传输和背压，不是独立编码计算时间；端到端耗时包含两次源视频解码、两模型、三路视频和完整输出解码核验。

约 2.65 GB 的 float32 概率缓存用于复核和避免重复推理。仅完整且 SHA256 身份一致的缓存可以复用；不完整缓存保留并报错。已有 MP4 或未完成的 `.encoding.mp4` 不会被覆盖，失败后应保留文件并选择新的输出目录。当前路径要求输入为 CFR，且宽度可被 6、高度可被 4 整除；不满足时明确报错。

`frames/` 的面积、亮色重叠、底部区域和相邻帧变化极值仅作为人工检查候选，不能据此自动判定车辆/仪表台误检。相邻帧变化没有运动补偿。视频没有分割真值，不能生成有意义的 IoU/F1；应结合原图审阅候选截图并记录真实可见问题。

视频、缓存、权重和逐帧 CSV 均保留在被 Git 忽略的 `outputs/`，不上传普通 Git 仓库。
