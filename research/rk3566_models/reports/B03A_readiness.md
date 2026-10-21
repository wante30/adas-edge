# B-03A: ADAS 训练前置检查

**结论：PARTIAL。现在不建议开始 B-03B 微调。** YOLOv6n 的权重加载和 Rockchip 专用 ONNX 导出路径已在 PC 冒烟通过；本机没有可用于七类训练的完整 BDD 图像和原始检测标注。

## 数据盘点

| 项目 | 结果 |
| --- | --- |
| 本机 BDD 衍生图片 | 9,000 张可读的 512×192 红绿灯 ROI 图；另有 100 张可读的 1280×720 冻结 holdout 原图。共 9,100 张不同文件，后 100 张均属于冻结的 1,000 张评测名单，不能训练。 |
| 原图/标注文件 | 完整 BDD 图像树、`bdd100k.csv` 和原始 `det_train.json`/`det_val.json` 均未找到；项目内 `data/competition_eval/bdd100k_validation` 也不存在。9,000 张图已裁成 ROI，不能恢复全帧。 |
| 数据来源 | 9,000 张来自 BDD100K **official validation**，其中 8,513/487 是历史 traffic-light 专项的内部 train/val 划分，不是官方 train。此前已用于 YOLO11n traffic-light 专项训练。 |

现存 ROI YOLO 标注的可统计数量如下。`无法统计`表示本机没有对应类别的原始标注，不能解读成该类目标数为零。

| 类别 | 实例数 | 含该类图片数 | 说明 |
| --- | ---: | ---: | --- |
| person | 无法统计 | 无法统计 | 原始 `pedestrian` 标注文件缺失 |
| bicycle | 无法统计 | 无法统计 | 类别标注文件缺失 |
| motorcycle | 无法统计 | 无法统计 | 类别标注文件缺失 |
| car | 无法统计 | 无法统计 | 类别标注文件缺失 |
| bus | 无法统计 | 无法统计 | 类别标注文件缺失 |
| truck | 无法统计 | 无法统计 | 类别标注文件缺失 |
| traffic light | 23,468 | 4,968 | 仅保留 ROI 裁剪后的框；数据卡记录另有 682 个框被裁剪过滤 |

抽查了 16 张复杂道路 ROI 图。已有框大体落在可见信号灯上，但缩图中远处小灯难以逐一确认；数据只标红绿灯，不能检查行人和两轮车是否漏标，也不能证明全图标注完整。审计脚本在现有 9,000 张上发现 0 个缺失标签文件、0 条格式错误和 0 个完全重复框。由于文件名哈希被用来切分，同一 BDD 视频名前缀仍有 284 个同时出现在 train/val，当前内部验证存在视频序列泄漏。

BDD 官方格式说明定义了灯色属性 `trafficLightColor: red|green|yellow|none`，见[官方标注格式](https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst)。但本机没有原始 JSON，ROI YOLO `.txt` 不保存属性，因此无法统计本地红/黄/绿标签数量或确认覆盖率。

官方数据许可允许教育、研究和非营利用途；按用途判断，校内非商业竞赛训练看起来在许可范围内，但应保留[官方许可](https://github.com/bdd100k/bdd100k/blob/master/doc/source/license.rst)和镜像来源说明。镜像文件自带的 BSD 许可不取代 BDD 数据条款；商业用途另需授权。

**不能安全组成 1,000–3,000 张七类训练集。** 排除冻结 holdout 后虽然有 9,000 张 validation 衍生图，但这些图只有 traffic-light ROI 框、已经用于历史专项目标训练，不能补出其余六类标签。最小解决办法是取得有许可的 BDD100K 官方 train 全帧及检测 JSON（或团队自采并完整标注的七类数据），按视频序列重做 train/val；保留冻结 1,000 张只作最终评测。

审计命令：

```powershell
python tools/audit_b03a_readiness.py `
  --dataset-root D:/Datasets/adas_edge_training/tl_specialist/data_512x192 `
  --dataset-yaml D:/Datasets/adas_edge_training/tl_specialist/data_512x192/dataset.yaml
```

已修复待用的 BDD 子集脚本类别映射：`pedestrian` 映射为 `person`，`rider` 保持独立并排除；内部验证改为按 BDD 视频名前缀切分，不再把相邻同视频帧拆到 train/val。没有实际生成新训练数据。

## YOLOv6n 导出路径

| 检查 | 结果 |
| --- | --- |
| 预训练权重 | 官方 `yolov6n.pt`（10,511,334 bytes，SHA256 `3E2944E1C2D6A0DC4EED72E1493E3AFE1C052373588B33ABFECF0BFB4212B1A4`）在 Meituan YOLOv6 `0.3.0` tag / commit `7123c541e7d5c3e168ea535376e4f2fabfd6e66f` 成功加载：80 类、约 5.07M 参数、stride 8/16/32。使用 `0.4.0` 代码会因 checkpoint 所需 `SimConv` 符号不匹配而失败，因此后续须固定对应版本。配置为 `configs/yolov6n.py`。 |
| 普通 ONNX | `deploy/ONNX/export_onnx.py` 在 512×288 可导出并通过 ONNX checker，输出 `[1,3024,85]`。这是标准解码输出，不符合 Rockchip Model Zoo YOLOv6 优化图的9路输出接口。 |
| RKNN 优化 ONNX | Rockchip YOLOv6 fork commit `0e7c2d5a93f6d49ed5ab6f005ccdd9d9bbd3db9b` 的 `deploy/RKNN/export_onnx_for_rknn.py` 使用同一官方 checkpoint 成功导出并通过 checker。输入 `[1,3,288,512]`，9个输出按三个stride分别为 `[1,4,H,W]`、`[1,80,H,W]`、`[1,1,H,W]`，与[Model Zoo v2.5.0 YOLOv6后处理](https://github.com/airockchip/rknn_model_zoo/tree/v2.5.0/examples/yolov6)的布局一致。输出ONNX SHA256 `1A40F4C75F555839C368A497A7CA1D8D4F92A943236232A2553AC1D57627AFF7`。 |
| 七类 RKNN | 可信路径存在：按 Meituan YOLOv6 训练文档微调，再用 Rockchip fork 专用 exporter，最后按 Model Zoo `python/convert.py ... rk3566` 转换。当前仅验证了80类原始权重的加载和优化 ONNX 导出；**没有训练七类模型、执行 RKNN 转换或连接 RK3566**，七类输出仍需后续获批后验证。 |

参考：[Meituan YOLOv6 0.3.0](https://github.com/meituan/YOLOv6/tree/0.3.0)、[Rockchip YOLOv6 fork](https://github.com/airockchip/YOLOv6/tree/0e7c2d5a93f6d49ed5ab6f005ccdd9d9bbd3db9b/deploy/RKNN)。权重与导出文件仅存在本机临时目录，没有加入 Git。

## 决策

**PARTIAL。不要立即开始 B-03B。** 先取得合规的官方 BDD train 全帧和原始七类检测 JSON（或完整标注的团队数据），重新按视频序列划分，审查标注覆盖并排除冻结 holdout；数据 gate 通过后再审批短程微调。模型路径不构成当前训练数据不足的替代。
