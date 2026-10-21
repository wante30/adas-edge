# B-03B：BDD100K 现成模型快速验证

**建议 A：先对 BDD YOLOv8n 做离线 RKNN 转换验证，暂缓 YOLOv6n 微调。** 两个现成 BDD 模型在这 50 张图上明显提高了召回、F1 和 AP50；YOLOv8n 整体最好且 PC 推理最快，但误检约为原始 COCO YOLO11n 的两倍。结果只用于快速筛选。

## 数据与模型

使用 B-01R-Lite 同一批 BDD100K validation 图片：验证目录中按文件名排序的前 50 张，753 个评测目标。输入固定为 512×288、batch=1、conf=0.25、NMS IoU=0.45、max_det=100；预测以 0.001 分数下限收集，再统一计算阈值指标和 AP50。GT 的 `pedestrian` 映射为 `person`；第三方模型按名称映射 `motor→motorcycle`、`bike→bicycle`，`rider` 独立保留并排除，不按 class ID 对齐。GT 中的 rider、traffic sign、train 等非共同类不计分。

两个 [HF 仓库](https://huggingface.co/dronefreak/bdd100k-yolo11n) / [YOLOv8n 仓库](https://huggingface.co/dronefreak/bdd100k-yolov8n) 在固定 revision 下公开可访问，模型卡标注 AGPL-3.0、BDD100K 数据、Ultralytics YOLO 和 DetectionBench 训练来源；args.yaml 显示 50 epochs、960 输入。仓库没有固定 Ultralytics 版本，本机使用 Ultralytics 8.4.68。文件大小及 SHA256：YOLO11n `5,490,522` bytes / `c120d075bcc6aaf94523f9a2c9d681819ebd692f36eb5ea64656d1c5f1278a70`（revision `2b5554133522f96f50ce87a4df0a2de556e4b7ad`）；YOLOv8n `6,268,010` bytes / `acd17780b331c3970e22faf3c6f123eae8b8c4abcc736ad5ef38ff28ecd72fb2`（revision `2c24162504d096e7bb0a12ec0af6bfeb9d5d697a`）。原始 COCO YOLO11n checkpoint SHA256 为 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`。静态 pickle 检查只见 PyTorch/Ultralytics 模型类型，并通过 Ultralytics `weights_only` 安全加载；三款模型均成功加载，两个 BDD 模型为 10 类检测头。

第三方模型卡称其评测 test split 是 BDD100K 官方 validation 的 10,000 张图；本轮 50 张属于同一评测池。因此这是已知的**评测集重叠/模型选择偏差风险**，不能视作独立外部测试。卡片未证明这 50 张曾参与权重更新，args.yaml 声明训练数据来自 BDD100K，但仅凭仓库材料无法审计训练文件清单。模型卡自报指标标为未验证。权重与 BDD 图片均只保存在本机，未提交 Git；商业分发或部署前还需核对 AGPL 与 BDD100K 数据许可义务。

## 表 1：50 张图片效果

总体 P/R/F1 为七类微平均；TP/FP/FN 按 IoU≥0.5 一对一匹配。类别列为 Precision / Recall，百分数。AP50 使用现有 COCOeval。

| 模型 | 7类 TP / FP / FN | 总体 P / R / F1 | AP50 | Car P / R | Person P / R | Traffic Light P / R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 COCO YOLO11n | 225 / 48 / 528 | 82.4 / 29.9 / 43.9 | 25.28% | 87.2 / 38.3 | 85.0 / 22.4 | 33.3 / 1.6 |
| BDD YOLO11n | 356 / 98 / 397 | 78.4 / 47.3 / 59.0 | 37.07% | 79.6 / 55.9 | 82.1 / 30.3 | 70.6 / 29.3 |
| BDD YOLOv8n | 381 / 100 / 372 | 79.2 / 50.6 / 61.8 | 37.33% | 79.8 / 59.2 | 100.0 / 34.2 | 66.7 / 32.5 |

BDD YOLOv8n 多检出 156 个 GT 目标，AP50 比原始 YOLO11n 高 12.05 个百分点；代价是多 52 个 FP，整体精度下降 3.2 个百分点。两个 BDD 模型都明显改善交通灯召回，但 YOLOv8n 仍漏掉 83/123 个灯，不能视为交通灯问题已解决。抽查叠框可见 BDD 模型多找回了道路边缘行人；密集车辆和截断车辆仍有框偏大、重叠或 IoU 不足导致的误检。50 张样本中的摩托车和自行车 GT 分别只有 3 和 7 个，相关结果不稳定。

## 表 2：RTX 4060 测速

Windows 本机 NVIDIA GeForce RTX 4060 Laptop GPU，Ultralytics PyTorch CUDA FP32；batch=1、512×288，同一批图。预热 10 次、计时 50 次，每次计时前后同步 CUDA。Pipeline 包含预处理、推理、解码/NMS 和坐标映射，不计磁盘读取。

| 模型 | 纯推理 ms / FPS | 完整 Pipeline ms / FPS | 后端 |
| --- | ---: | ---: | --- |
| 原始 COCO YOLO11n | 8.48 / 117.94 | 10.53 / 94.93 | PyTorch CUDA FP32 |
| BDD YOLO11n | 8.52 / 117.41 | 10.94 / 91.43 | PyTorch CUDA FP32 |
| BDD YOLOv8n | 6.37 / 157.04 | 9.31 / 107.44 | PyTorch CUDA FP32 |

该短测中 YOLOv8n 最快；PC 数据不代表 RK3566 性能，单次短测也不代表长期稳定速度。

## 表 3：选择结论

| 问题 | 结论 |
| --- | --- |
| 两个 BDD 模型是否加载、类别映射是否正确 | 是。两者均安全加载；按实际类别名映射，未将 rider 当 person，也未按类别 ID 硬对齐。 |
| 误检和漏检的主要变化 | 相比 COCO YOLO11n，BDD YOLOv8n 的 TP 从 225 升至 381、Recall 从 29.9% 升至 50.6%、F1 从 43.9% 升至 61.8%；FP 从 48 升至 100。交通灯召回升至 32.5%，仍漏检多数 GT。 |
| 是否值得继续 | **A。** 推荐获批后优先离线转换 BDD YOLOv8n，BDD YOLO11n 保留为备选；继续转换前先核对 RKNN 工具链对 10 类输出的处理，并在未参与该模型卡评测的新样本上复核 FP。 |
| 是否现在训练 YOLOv6n | 否。现有 BDD YOLOv8n 已展现足够收益，先验证离线导出/转换成本；本轮未训练、未导出 ONNX/RKNN，也未接触 RK3566。 |

三款模型的代表性叠框图保存在本机：`D:\Datasets\adas_edge_training\b03b_eval_20261021\comparison_grid_final.jpg`。该图和预测缓存未提交，报告中不内嵌或上传 BDD 原图。
