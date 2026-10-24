# B-05J — Specialist 精度差异审计、训练泄漏修复与板端对照

日期：2026-10-24  
分支：`research/rk3566-adas-models`  
基线：`187e027ca39cb0d5f6dbd1fb6379fa7c4367d4da`

## 0. 结论

已修复/确认：

- B-05I 的尺寸口径和 conf 口径问题；
- B-05I 域训练数据泄漏（val/test 被汇入 domain）；
- `best.pt`、ONNX、RKNN 的 SHA 与 manifest 一致；
- B-05G 零有效颜色的根因是 **AI Bridge quad vs Qt front**，B-05G-R 已修复并得到 31 条有效 GREEN；
- 修正后的 candidate_clean 仍不如 source，停止继续训练。

未完成：

- ONNX FP32 的直接公平评估因固定 `1×3×192×512` 输入与当前 Ultralytics square wrapper 不兼容，未能完成；
- 因此不能把 PyTorch→ONNX→RKNN 的损失精确拆分到每一段。

## 1. P0：评估口径修复

### GT 数量

BDD50 原图 human GT：

```text
154 traffic lights
```

顶部 60% ROI（Specialist 实际输入）内：

```text
150 GT
```

ROI 外：

```text
4 GT
```

这 4 个目标无法被当前 Specialist 输入覆盖，不能计入 ROI Recall，也不能忽略。

### 尺寸口径

旧 B-05I 用 512×192 网络像素宽度分箱，B-05D 用 1280×720 原图宽度。  
已改为：

```text
原图宽度 ×2.5 = 1280 空间宽度
```

后再分 `<8 / 8–16 / 16–32 / >32px`。

### conf

评估固定：

```text
conf=0.25
IoU=0.5
```

## 2. 修正后的指标

### top60 ROI / 150 GT

| 模型 | Precision | Recall | AP50 |
|---|---:|---:|---:|
| PyTorch source `best.pt` | 0.6262 | **0.4467** | **0.3681** |
| candidate_clean | 0.6238 | 0.4200 | 0.3612 |
| RKNN hybrid2 | 0.3657 | 0.3267 | 0.2404 |

注意：

- B-05I 之前报告的 source AP50 0.402 使用的是非统一阈值；此处 0.3681 才是固定 conf=.25 的结果。
- candidate_clean 是修复域泄漏后重新训练的 1 epoch 模型，仍低于 source，未部署。

### 原图宽度尺寸 Recall（conf=.25）

RKNN 与 PyTorch 分开统计，不混表。

RKNN：

| 尺寸 | GT | Recall |
|---|---:|---:|
| <8px | 17 | 0.059 |
| 8–16px | 80 | 0.325 |
| 16–32px | 47 | 0.596 |
| >32px | 6 | 0.833 |

PyTorch source 在 150 ROI GT 上的 8–16px Recall 为 0.325；candidate_clean 为 0.35。  
两者均来自同一 conf=.25、原图宽度口径，但 RKNN 与 PyTorch 仍是两个推理实现，不能把差异全部归因于量化。

### 端到端覆盖率

```text
ROI 覆盖：150 / 154 = 97.4%
未覆盖：4 / 154 = 2.6%
```

## 3. P1：模型来源核对

manifest 声明并实测 SHA：

```text
best.pt     F5FC9C2B1220D55C4A49C09EC929B0C977B680ED8A84FAF64D471D15FEB6CE2C
ONNX        7188A00D543781F63C79DD0119EF04CB7848D0FCAF653EB63A03391E63DA4CA4
RKNN hybrid BECFAADA5DA49C4DD6C143E815F8F957FB7B7ED759040D4E18C685A9C434E6F8
```

manifest：

```text
weights_best_pt = f5fc9c2b...
onnx_static     = 7188a00d...
rknn_hybrid2    = becfaada...
```

结论：

**`tl_specialist_512x192_hybrid2.rknn` 的 manifest 标注来源与 `best.pt/ONNX` 一致。**

ONNX：

```text
input:  images [1,3,192,512]
output: output0 [1,5,2016]
```

训练配置：

```text
yolo11n.pt, imgsz=512, rect=True, mosaic=0
epochs=60, batch=16, seed=0
dataset: top-60% ROI 512×192
```

## 4. P2：PyTorch/ONNX/RKNN 差异

已完成：

- PyTorch source 固定 conf=.25 评估：AP50 0.3681；
- RKNN hybrid2 同一 150 GT：AP50 0.2404；
- 两者差距明显。

未完成：

- ONNX FP32 直接评估。当前 Ultralytics ONNX 后端按 square 512×512 warmup，而模型固定 192×512，出现：

```text
Got invalid dimensions index 2: Got 512 Expected 192
```

因此不能把 PyTorch→ONNX 与 ONNX→RKNN 的损失拆开。  
当前只能确认：

1. 权重 SHA 同源；
2. RKNN 与 PyTorch 的最终 AP 差异显著；
3. 差异位于 ONNX 转换/预处理/后处理/RKNN 量化中的某一段，尚未定位到单一步骤。

不继续盲目量化，也不把差距全归因于 INT8。

## 5. P3：B-05G 零有效颜色

根因已在 B-05G-R 定位并修复：

```text
AI Bridge: --layout quad
Qt:        --pgv-layout front
```

Specialist 收到的是 Q1 而不是完整前视，已知绿灯被裁掉。  
改为 `--layout front` 后：

```text
78 Track 记录
31 条 color_valid=True
effective_color=GREEN 31
```

B-05G 的 604 条日志 6 条 Track、0 有效颜色是布局错误的结果，不是 Tracker/HSV 单独失效。

## 6. 训练泄漏修复

B-05I `b05i_build_train_data.py` 曾把：

```text
pseudo_A/images/{train,val,test}
```

全部汇入 domain，导致比赛视频 val/test 泄漏到训练。

已修复：

```text
domain 只读取 pseudo_A/images/train
domain images：7
BDD50 始终不进入训练
```

重新训练：

```text
candidate_clean
```

结果：

```text
AP50 0.3612
Recall 0.4200
```

低于 source 的 AP50 0.3681 / Recall 0.4467，未部署。

## 7. P4：是否继续训练

不继续。

理由：

- 修正泄漏后，域辅助候选没有优于 source；
- RKNN 与 PyTorch 的显著差距尚未拆解；
- 先修复部署链路比继续微调更优先；
- 现有证据不支持新增训练。

## 8. P5：回答

1. **PyTorch 与 RKNN 是否同源**：manifest/SHA 表明是同一来源链路；实际精度不等价。
2. **统一口径 AP50/Recall**：
   - PyTorch source：AP50 0.3681，Recall 0.4467；
   - RKNN hybrid2：AP50 0.2404，Recall 0.3267；
   - candidate_clean：AP50 0.3612，Recall 0.4200。
3. **原图 8–16px Recall**：RKNN 0.325；PyTorch source 0.325；candidate_clean 0.35（不同实现，分开看）。
4. **量化/后处理/输入链路差异**：已确认最终差异，尚未拆分到 ONNX/后处理/量化具体一段。
5. **训练泄漏**：已修复，domain 只取 train，BDD50 独立。
6. **B-05G 零颜色根因**：layout quad vs front，已定位并修复；31 条有效 GREEN。
7. **是否实际修复并上板复测**：B-05G-R 已上板修复并产生有效 GREEN；B-05J 未再改板端。
8. **现有模型能否靠修复部署链路提升**：B-05G-R 证明布局修复能恢复颜色；但检测 AP 的 RKNN/PyTorch 差距仍需修 ONNX/RKNN 链路。
9. **是否还需要额外训练**：当前不需要；先完成 ONNX 直评和 RKNN 后处理 parity。
10. **是否有明确优于现有 RKNN 的可部署候选**：**没有**。candidate_clean 不如 source，未部署。

## 9. 交付

```text
tools/b05i_eval.py                 # conf=.25 修复
tools/b05i_size_eval.py            # 原图宽度口径
tools/b05i_build_train_data.py     # train-only domain
research/rk3566_models/reports/B05J_model_parity_and_fix.md
research/rk3566_models/reports/assets/B05J_metrics.json
```

旧 `tl_specialist_512x192_hybrid2.rknn` 保留；没有覆盖正式权重。

本轮未修改 A 线 Qt、FPGA、PCIe 或驾驶决策。
