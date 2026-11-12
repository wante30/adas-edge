# Stage 3 Autoware CNN 与漏检诊断

本变更为多交通灯离线评估增加三个可复现入口：

- `competition_ai/autoware_classifier.py`：官方 Autoware MobileNetV2 v4 PC ONNX 后端、官方组合标签解析和保守融合后端。
- `tools/stage3_detection_diagnostic.py`：保存 NMS 前/后/抑制候选、阈值阶段、letterbox 参数、源坐标映射和 P2+Specialist 轨迹阶段。
- `tools/stage3_classifier_comparison.py`：在同一检测框上比较传统算法、Autoware CNN 和保守融合，并分别运行 Q4 与源视频原始 ROI。
- `tools/fetch_autoware_models.py`：按固定 Hugging Face v4.0 revision 下载 ONNX、标签和参数，生成 SHA-256 manifest；模型权重不入库。

实现约束：

1. 默认 `OfflineDetector.detect(image)` 接口保持不变；诊断通过 `return_trace=True` 增加旁路信息。
2. `observed_directions` 继续表示视觉观察方向；`directions` 继续遵循现有关联/UNKNOWN 策略。
3. Autoware 官方 ONNX 输出是概率向量，后端只在检测到 logits 时执行 softmax，避免二次 softmax。
4. color-only 官方标签（例如 `red`）不能为小 ROI 提供几何证据；保守融合不会据此把 `UNKNOWN` 形状升级为 `ROUND`。
5. 检测默认阈值没有改成 0.15；该阈值只作为诊断敏感性实验，避免新增候选被误报为召回提升。

复现：

```powershell
python tools\fetch_autoware_models.py --output-dir outputs\multi_signal_stage3\autoware_model --revision v4.0
python tools\stage3_detection_diagnostic.py --output outputs\multi_signal_stage3\detection_diagnostics_final --clips 112454 113421 112507 113452 --stride 5
python tools\stage3_classifier_comparison.py --output outputs\multi_signal_stage3\classifier_comparison_final --cnn-model outputs\multi_signal_stage3\autoware_model\traffic_light_classifier_mobilenetv2_batch_1.onnx --labels outputs\multi_signal_stage3\autoware_model\lamp_labels.txt --clips 112454 113421 112507 113452
python -m pytest -q
```

输出目录中的视频、JSONL、CSV、失败 ROI 和中文报告不属于代码提交；官方模型通过下载脚本获取。
