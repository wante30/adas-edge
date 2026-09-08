# BDD100K 外部检测预验收

日期：2026-09-07

## 数据来源

- 数据集：BDD100K validation-only 镜像（Hirundo）
- 官方项目入口：[BDD100K GitHub](https://github.com/bdd100k/bdd100k)
- 实际下载地址：[Hugging Face 数据集](https://huggingface.co/datasets/hirundo-io/bdd100k-validation-only)
- 本地压缩包：`data/competition_eval/raw/bdd100k_validation_hirundo.zip`
- 压缩包大小：约 575 MB（下载完成后 SHA256 应另行记录）
- 镜像许可证文件：BSD 3-Clause；BDD100K 原始数据的使用条款仍需遵守，镜像许可证不自动扩大原始数据权利。
- 内容：10,000 张 validation 图像、`bdd100k.csv` 逐框标注；不是视频，未伪装成视频 benchmark。

## 方法

从 validation 图像按文件名排序取前 1,000 张，使用本地 `yolo11n.pt` COCO 预训练权重，在 RTX 4060 的 `yolo` 环境进行 Ultralytics FP32 推理。匹配类别为 car、truck、bus、person（BDD pedestrian 映射）、motorcycle、bicycle、traffic light；IoU >= 0.5 视为 TP，置信度阈值 0.25。该实验是图像级外部预验收，不代表 RK3566 INT8 结果，也不是对完整 validation 集的官方 benchmark。

## 结果

| 类别 | 320 precision | 320 recall | 320 F1 | 416 precision | 416 recall | 416 F1 |
|---|---:|---:|---:|---:|---:|---:|
| car | 0.765 | 0.230 | 0.354 | 0.765 | 0.296 | 0.427 |
| truck | 0.309 | 0.148 | 0.200 | 0.348 | 0.199 | 0.253 |
| bus | 0.341 | 0.166 | 0.223 | 0.322 | 0.204 | 0.250 |
| person | 0.659 | 0.094 | 0.165 | 0.653 | 0.161 | 0.258 |
| motorcycle | 0.286 | 0.037 | 0.066 | 0.625 | 0.093 | 0.161 |
| bicycle | 1.000 | 0.011 | 0.021 | 0.833 | 0.053 | 0.099 |
| traffic light | 0.205 | 0.010 | 0.018 | 0.359 | 0.033 | 0.060 |

完整原始结果：

- `outputs/bdd100k_external/320/report.json`
- `outputs/bdd100k_external/416/report.json`

带框预览（由抽样图像按 10 fps 合成，仅用于查看，不是原始视频）：

- `outputs/bdd100k_external/320/annotated_preview.mp4`
- `outputs/bdd100k_external/416/annotated_preview.mp4`

## 判断

1. 416 在 car、truck、bus、person、motorcycle、bicycle、traffic light 的 recall 均高于 320；traffic light recall 从 0.95% 提升到 3.29%，person recall 从 9.41% 提升到 16.09%。
2. 绝对召回仍然偏低，尤其 traffic light、bicycle、motorcycle。这不是“416 已经够用”的证据，而是当前 COCO 预训练模型与 BDD 场景存在域差异/小目标问题的证据。
3. 由于该样本是按文件名取前 1,000 张，不能外推完整 validation 集；下一轮应做分层抽样（按类别、昼夜、场景）或直接使用官方视频/团队视频。
4. 训练决策：可以进入一个小规模、可复现的 fine-tune 试验，但不应直接宣称训练有效。先固定 train/val 按视频或场景切分，保留本轮 1,000 图像作为外部 holdout，比较训练前后的同一指标。

## 当前状态

- 外部图像验证：COMPLETED（PC FP32）
- 视频验证：BDD100K 原始视频尚未取得，状态 UNKNOWN
- RK3566 INT8 验证：本轮未执行
- 论文/MOT17 实验：未修改

下载文件 SHA256：B57FAF25AC614BFC5CF508BAD8E2AF18534B846F230EF2222A2A69EC7A678122
下载文件字节数：574910510
