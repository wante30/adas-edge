# B-03E：BDD YOLOv8n 阈值与RKNN量化优化

任务已批准，直接执行，不重新编写计划。

仓库：`wante30/adas-edge`
分支：`research/rk3566-adas-models`

## 一、目标与限制

复用B-03D的50张BDD100K图片、705个GT框、板端预测缓存及现有BDD YOLOv8n权重。

目标：降低误检，改善INT8行人和交通灯检测质量，同时尽量保持RK3566速度优势。

允许使用Windows/WSL和RK3566。

禁止训练、下载大型数据集、租4090、替换正式模型、修改主线A或删除Traffic-Light Specialist。

## 二、先做阈值分析

直接复用B-03D低阈值预测缓存和GT，不重复板端推理。

比较conf：

`0.25 / 0.35 / 0.45 / 0.55`

输出各阈值下：

- 总体TP、FP、FN、Precision、Recall、F1。
- Car、Person、Traffic Light的Precision/Recall/F1。
- 交通灯误检变化。
- 相比原YOLO11n的实际收益。

可以额外探索少量分类别阈值，但不要无限搜索。

选出一个值得后续复核的工作阈值，同时保留统一conf=0.25的公平对比结果。

这50张属于反复使用的历史诊断样本，阈值结果仅用于筛选，不视为独立最终测试成绩。

## 三、检查INT8退化原因

先检查现有RKNN转换流程：

- RGB/BGR、NHWC、UINT8输入及mean/std。
- ONNX与RKNN预处理是否严格一致。
- 76张校准图片的内容、分布、预处理和代表性。
- 校准集是否与50张评测图片重复。
- 检测框解码、量化输出尺度及NMS是否一致。

重点解释已有结果：

- FP32 AP50 29.59%，INT8 23.34%。
- Person Recall 38.24% → 20.59%。
- Traffic Light Recall 39.87% → 18.30%。
- FP 95 → 198。

注意：先核实FP32缓存的最低conf；若只有0.25以上预测，不将其AP50与INT8完整AP50直接比较。

## 四、有限量化优化

不修改训练权重。

优先复用现有RKNN Toolkit2 2.3.2与项目已有转换脚本。

在不引入GT评测图作为校准数据的前提下：

1. 优先尝试一次校准数据或量化配置改进。
2. 如工具链支持且成本合理，再参考现有Traffic-Light Specialist经验，尝试一次混合精度量化。
3. 最多保留两个新RKNN候选，不做大规模参数搜索。

不要假设RKNN支持某个混合量化API；先确认实际工具链能力。不支持就记录原因并停止该方向。

保存所有候选的配置、模型SHA256和转换日志。

## 五、RK3566实机复测

仅对成功转换的候选做短时验证。

与B-03D原BDD YOLOv8n INT8比较：

- 相同50张GT、512×288及检测协议。
- AP50、Precision、Recall、F1。
- Car、Person、Traffic Light的TP/FP/FN。
- INT8相对FP32的检测退化。
- NPU-only及完整Pipeline FPS。

统一conf=0.25报告公平结果；另外报告第二阶段选出的推荐阈值结果。

如需短时暂停bridge做独占测试，限于项目负责人已授权的测试窗口，保留原状态、设置异常恢复措施，结束后核验服务和实际视频数据流。

不得覆盖正式YOLO11n或Specialist，不修改板端全局驱动/Runtime，不重启设备。

优先复用现有测试脚本，不重复无价值测速。

## 六、最终交付

报告：

`research/rk3566_models/reports/B03E_quantization_optimization.md`

给出简洁对比表：

| 指标 | 原BDD INT8 | 优化后INT8 |
|---|---|---|
| AP50 | 23.34% | 实测 |
| Precision | 60.87% | 实测 |
| Recall | 43.69% | 实测 |
| F1 | 50.87% | 实测 |
| Person Recall | 20.59% | 实测 |
| Traffic Light Recall | 18.30% | 实测 |
| 总FP | 198 | 实测 |
| Pipeline FPS | 26.317 | 实测 |

必须区分conf=0.25和推荐阈值，不混合比较。

最终回答：

1. 调整阈值能减少多少FP，损失多少TP？
2. INT8退化主要来自预处理、校准还是量化本身？无法归因时明确说明。
3. 新RKNN是否明显改善行人和交通灯？
4. 新模型是否仍比原YOLO11n的21.616 Pipeline FPS快？
5. 推荐保留原INT8、采用改进INT8，还是暂时停止优化？

输出PASS/PARTIAL/FAIL。

提交必要代码、配置和Markdown报告到GitHub研究分支，push并提供真实Commit SHA。

不提交模型权重、图片或大型数据。

**最多两款新量化候选，完成后停止，不自动替换正式模型、不开始训练。**