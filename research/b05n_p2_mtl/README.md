# B05N P2 多任务支线（阶段 A）

在 B05N P2 十类检测模型上增加共享分割解码器，一次前向输出：

| 输出 | 类别 |
|---|---|
| detection | 原 10 类，顺序不变 |
| lane_cat | crosswalk / double other / double white / double yellow / road curb / single other / single white / single yellow / background |
| lane_style | solid / dashed / background |
| drivable | direct / alternative / background |

不改动比赛主线、RK3566 推理和 Qt 车机；本目录独立运行。

## 模型

- 源权重：`weights/b05n_p2_multiclass.pt`（FP32，Ultralytics `yolov8-p2.yaml`，输入 512×288，BDD holdout mAP50=0.3711）。
- 保留 Backbone / Neck / P2-P5 / 十类检测头，阶段 A 全部冻结（`requires_grad=False`，BN 保持 eval）。
- 新增 `SegDecoder`：P4→P3→P2 自上而下融合，最近邻上采样到 1/1，三个 1×1 头。
  只使用 Conv/BatchNorm/SiLU/Upsample，便于后续单图 ONNX 导出。
- 分割头参数量：0.22M；总模型 ~3.1M。

## 数据

使用官方 BDD100K：

- 图像：`bdd100k_images_100k.zip`（100K，1280×720），只解压 train/val。
- 标注：`bdd100k_labels.zip` 中每图 JSON，含 `lane/*`、`area/drivable`、`area/alternative` 的 `poly2d`。
  该 JSON 与官方 lane mask 使用同一套标注，直接按官方编码渲染，不依赖额外 434MB 车道包。

车道 mask 编码（`bdd100k/label/to_mask.py`）：

```
value = category_id + (direction << 5) + (style << 4)
category 0..7，style 0=solid 1=dashed，direction 0=parallel 1=vertical
background = 255
```

渲染时把贝塞尔控制点采样为折线，线宽 4px（512×288 画布，约等于原图 10px，用于缓解极细前景）。
drivable mask 为 `0=direct, 1=alternative, 2=background`。
缺失 mask 的样本会置 `has_lane/has_drivable=False`，对应任务损失被跳过，不会当成背景。

## 阶段 A 训练

```bash
python scripts/prepare_data.py --root /root/autodl-tmp/bdd100k --workers 48 --splits train,val

python scripts/train_stage_a.py \
  --root /root/autodl-tmp/bdd100k \
  --ckpt weights/b05n_p2_multiclass.pt \
  --out runs/stageA \
  --epochs 10 --batch 16 --limit-train 45000 --limit-val 600 \
  --workers 12 --lr 1.2e-3
```

- RTX 3090 24GB，AMP fp16，imgsz 512×288，batch 16。
- 损失：lane 加权 CE + 8 类 Dice + 前景 Dice；style 只在车道像素上算 CE/Dice；drivable CE/Dice。
- 检测头冻结且无检测损失，仅保留前向，阶段 B 再解冻并加入检测监督。

## 推理与可视化

```bash
python scripts/predict_visualize.py \
  --root /root/autodl-tmp/bdd100k \
  --ckpt runs/stageA/mtl_best.pt \
  --out runs/stageA/viz --num 8
```

每张图输出 2×2 面板：检测框、可行驶区域、车道类别、实线/虚线。

## 目录

```
mtl/model.py           多任务模型 + 分割解码器
mtl/label_render.py    官方 lane/drivable mask 渲染
mtl/data.py            Dataset（检测框 + 双 mask）
mtl/losses.py          CE / Dice / 前景 Dice / 指标
scripts/prepare_data.py
scripts/train_stage_a.py
scripts/predict_visualize.py
weights/               源 FP32 权重
runs/stageA/           阶段 A checkpoint、日志、预测图
```

## 阶段 B（已完成）

B-1 全量分割头 → B-2a 选择性解冻 + 检测损失 → B-2b 对照，均已完成。
B-2a 作为部署版本（检测/TL 最好），B-2b 作为分割最佳对照。
完整数据、训练代码和证据见本目录 `scripts/`、`mtl/`、`runs/` 与
`deployment/stageB1/`、`deployment/stageB2a/`。

> Git 归档说明：`weights/`、`runs/*/mtl_best.pt`、`*.onnx`、`*.rknn`、
> 原始帧和演示 MP4 不进入普通 Git；SHA256、大小和板端路径见
> `artifacts/manifests/model_manifest.json`。`runs/stageB2b/contact_sheet.jpg`
> 是训练可视化缩略图。

## 阶段 A 实测结果

- 数据：BDD100K train 中 45,000 张（每 epoch 2,812 step），batch 16，AMP，512×288。
- 训练：先 5 epoch（lr 1.2e-3）跑通，再用 Tversky 微调 4 epoch（lr 3e-4）；合计 9 epoch / 约 25,300 step，3090 上约 29 分钟。
- 验证（600 张 val，均为模型真实输出）：

| 指标 | 值 |
|---|---|
| 车道前景 IoU | 0.369 |
| 斑马线 IoU | 0.242 |
| 线型准确率（solid/dashed） | 0.835 |
| 可行驶区域 mIoU | 0.584 |

主要 loss（最后 val）：lane_ce 0.735、lane_dice 0.488、lane_fg_dice 0.538、lane_tversky 0.492、style_ce 0.312、drv_ce 0.343。

这是短程验证，不是最终精度：线条仍偏粗，斑马线与白线会混淆，夜间/逆光有漏检。要继续阶段 A：

```bash
python scripts/train_stage_a.py \
  --root /root/autodl-tmp/bdd100k \
  --ckpt weights/b05n_p2_multiclass.pt \
  --resume runs/stageA_ft/mtl_best.pt \
  --out runs/stageA_full \
  --epochs 20 --batch 16 --limit-train 0 --limit-val 1000 \
  --workers 12 --lr 5e-4
```

阶段 B 的解冻共享特征 + 检测损失已在 B-2a/B-2b 完成；结果见下文与 `scripts/train_stage_b2a.py`。

## B-1：全量分割头训练（已完成）

```bash
python scripts/train_stage_a.py \
  --root /root/autodl-tmp/bdd100k \
  --ckpt weights/b05n_p2_multiclass.pt \
  --resume runs/stageA_ft/mtl_best.pt \
  --out runs/stageB1 \
  --epochs 14 --batch 16 --limit-train 0 --limit-val 1000 \
  --workers 16 --lr 5e-4
```

- 全量 70k train，14 epoch，每 epoch 4,375 step；检测分支完全冻结，网络结构未改，分割头未扩大。
- 3090 上 80.6 分钟；best 按 `lane_fg_iou + crosswalk_iou + 0.5*style_acc + 0.5*drivable_miou` 选择（epoch 13）。

同一 1000 张 val、同一指标口径对比：

| 指标 | 阶段 A | B-1 | 变化 |
|---|---|---|---|
| 车道前景 IoU | 0.3726 | 0.3868 | +0.0142 |
| 斑马线 IoU | 0.2440 | 0.2646 | +0.0206 |
| 线型准确率（前景内） | 0.8378 | 0.8468 | +0.0090 |
| 实线准确率 | 0.7798 | 0.7955 | +0.0157 |
| 虚线准确率 | 0.9092 | 0.9103 | +0.0011 |
| 可行驶区域 mIoU | 0.5828 | 0.5885 | +0.0057 |

结论：全量数据有提升，但幅度不大，瓶颈在 0.22M 解码器容量与 P2 步长，而不是数据量。B-2 解冻共享特征 + 检测损失前，需要先确认检测 mAP 的回退幅度。

## B-2b：第二轮 3 epoch 联合微调（已完成）

从 B-2a best 接续，同样解冻 P2/P3 相关层、同样损失和分层学习率，3 epoch / 70k，24.8 分钟。

| 指标 | B-1 | B-2a | B-2b |
|---|---|---|---|
| 车道前景 IoU | 0.3868 | 0.4281 | **0.4396** |
| 斑马线 IoU | 0.2646 | 0.3079 | **0.3236** |
| 前景线型准确率 | 0.8468 | 0.8756 | **0.8830** |
| 实线准确率 | 0.7955 | 0.8387 | **0.8470** |
| 虚线准确率 | 0.9103 | 0.9211 | **0.9275** |
| 可行驶区域 mIoU | 0.5885 | 0.6480 | **0.6683** |
| 检测 mAP50 | 0.3700 | **0.3707** | 0.3667 |
| car AP50 | 0.7086 | 0.7122 | **0.7123** |
| person AP50 | **0.4564** | 0.4441 | 0.4449 |
| traffic light AP50 | 0.4949 | **0.4997** | 0.4962 |
| BDD50 TL AP50 | 0.5226 | **0.5312** | 0.5221 |
| 8–16px TL recall | **0.4524** | 0.4286 | 0.4167 |

判断：分割仍在涨，但增幅已从 B-2a 的 +0.043（斑马线）降到 B-2b 的 +0.016；检测总 mAP 小幅回落 0.003，关键类别都在 ±0.02 内；小交通灯 recall 继续下降。按停止规则，B-2b 后停止继续训练，保留 B-2b 作为分割最佳、B-2a 作为检测/TL 最佳。
