# B05N P2 / B-2a MTL 研究记录

更新日期：2026-10-31
范围：B05N P2 十类检测、B-2a 多任务分割、RKNN INT8 / Seg-256、RK3566 C API、AI Bridge 多模型集成、Qt 分割显示。
用途：比赛文档、论文写作、复现实验和后续开发的统一研究记录。数值均来自仓库内 JSON / 日志；测试条件逐项标注。

---

## 1. 研究问题与目标

### 1.1 问题

1. 单帧 1280×720 道路画面中，同时检测 10 类交通目标，并输出车道类别、实线/虚线、斑马线和可行驶区域。
2. 在 RK3566（4×A55，NPU 0.8 TOPS 级）上保持可用的交互帧率；不依赖额外 GPU/服务器。
3. 保留 B05N P2 纯检测模型作为正式方案，多任务模型作为可切换扩展，不因新增任务削弱原检测。
4. 把 C API 的 INT8 原生输出、共享内存分割通道和多模型热切换整合进现有 Qt 车机。

### 1.2 约束

- 不重新训练检测 backbone；B-2a 在 B05N P2 权重上增加共享分割解码器。
- 不改变原 10 类检测输出顺序和 BDD10 类别定义。
- 不修改 FPGA/PCIe 视频链路、HUD、内核、RKNN Runtime。
- 模型权重、ONNX、RKNN、数据集不进入普通 Git 历史。

### 1.3 最终系统形态

```text
B05N P2（B）          纯检测，正式默认
B-CAPI                B05N P2 + RKNN C API
MTL-FULL              B-2a + 原版 512×288 分割
MTL-FAST              B-2a + Seg-256 144×256 分割
A/C/D + 历史方案       保留，可切换
```

---

## 2. B05N P2 模型设计

### 2.1 结构

- Ultralytics `yolov8-p2.yaml`，在标准 YOLOv8n 的 P3/P4/P5 之外增加 P2/4 分支。
- 输入 512×288（rect），4 个检测尺度：P2/P3/P4/P5。
- 10 类，顺序：

```text
0 person, 1 rider, 2 car, 3 truck, 4 bus,
5 train, 6 motor, 7 bike, 8 traffic light, 9 traffic sign
```

- 输出采用 Raw-head：每尺度 3 路（reg 64 通道 DFL logits、cls 10 通道、cls_sum 1 通道），共 12 路；host 端做 DFL softmax/期望、ltrb→xyxy、NMS。
- RKNN 输入 RGB uint8 NHWC，`/255`；不把 DFL/坐标解码放进模型，避免 INT8 输出范围被坐标主导。

### 2.2 训练数据与初始化

- 训练集：BDD100K validation 10k 镜像中的 8,513 张（复用 B05L 名单），val 482 张，test 1,000 张 holdout。
- BDD50：50 张人工精修 GT，只做交通灯专项；不进入训练/验证。
- 初始化：正式 BDD-YOLOv8n；P2 结构导致 neck/head 索引后移，精确匹配参数约 43.5%，neck/head 大量重初始化。
- 训练：60 epoch，batch 32，seed 0，AMP；最终交付 checkpoint 在 RTX 4060 Laptop 上训练。

### 2.3 检测结果

同一 BDD holdout-1000、conf 0.25、IoU 0.5：

| 模型 | mAP50 | mAP50-95 | Precision | Recall |
|---|---:|---:|---:|---:|
| 正式 BDD-YOLOv8n | **0.3831** | **0.2102** | 0.6754 | 0.3569 |
| 标准 YOLOv8n 同数据 60ep | 0.3660 | 0.1939 | — | — |
| B05N P2（4060 最终） | 0.3711 | 0.1944 | 0.6035 | 0.3580 |

分车型 AP50（BDD vs P2）：

| 类别 | BDD | P2 | 差值 |
|---|---:|---:|---:|
| person | 0.4389 | **0.4528** | +1.39 |
| rider | **0.3207** | 0.2636 | −5.71 |
| car | 0.6632 | **0.7117** | +4.85 |
| truck | **0.4960** | 0.4686 | −2.74 |
| bus | **0.4443** | 0.4059 | −3.84 |
| motor | **0.3705** | 0.2718 | −9.87 |
| bike | **0.2031** | 0.1585 | −4.46 |
| traffic light | 0.4283 | **0.4834** | +5.51 |
| traffic sign | 0.4665 | **0.4948** | +2.83 |

结论：P2 在 car/person/交通灯/交通标志上更好，rider/motor/bike/truck/bus 退化；总体 mAP50 比正式 BDD 低约 1.2 点。它是“检测结构 + 交通灯”导向的选择，不是全面替代 BDD。

### 2.4 交通灯专项

BDD50、150 ROI GT、conf 0.25、IoU 0.5、一对一匹配：

| 模型 | AP50 | Recall | 8–16px Recall |
|---|---:|---:|---:|
| 正式 BDD 全画面 | 0.4622 | 0.4333 | 0.3500 |
| B05N P2 全画面 | **0.5205** | 0.4533 | 0.4125 |
| B05L P2 Specialist（top60 ROI） | **0.6107** | **0.5800** | **0.5750** |

P2 全画面比 BDD 的 AP 高约 5.8 点、小灯 recall 高约 6.25 点；但仍低于单类 Specialist。B05L 继续作为 C/D 双模型方案的交通灯专模保留。

### 2.5 量化结果

800 张 BDD test 子集，Raw-head INT8：

| 格式 | mAP50 | TL AP50 |
|---|---:|---:|
| ONNX FP32 | 0.3646 | 0.4766 |
| RKNN FP16 | 0.3644 | 0.4778 |
| RKNN INT8 Raw-head | **0.3651** | **0.4733** |
| RKNN INT8 post-DFL split | 0.3031 | 0.2596 |

Raw-head 导出是最终采用的量化解决方法：把 DFL/坐标解码移出模型，避免 concat 后坐标主导量化范围；分类分数保留在更合适的域；每尺度独立输出。详见 `docs/deployment/RKNN_QUANTIZATION_REPORT.md`。

### 2.6 证据

- `docs/models/evidence/B05N/*.json`
- `research/b05n_p2/reports/B05N_multiclass_P2_comparison.md`
- `research/b05n_p2/reports/B05P_quantization_investigation.md`
- `research/b05n_p2/tools/`

---

## 3. B-2a 多任务网络结构

### 3.1 总体

```text
B05N P2 Detector（Backbone/Neck/P2-P5/10-class head）
        │
        ├── 原检测头 → 12 路 Raw-head
        │
        └── P2/P3/P4 neck taps
                │
        SegDecoder（P4→P3→P2 自上而下融合）
                │
        ┌───────┼────────┐
     lane_cat  lane_style  drivable
      9 类       3 类        3 类
```

- 检测 backbone/neck/head 全部保留；分割解码器只消费 neck 特征，无第二 backbone。
- 解码器仅使用 Conv/BatchNorm/SiLU/Upsample，便于 ONNX→RKNN。
- 分割头参数量 0.219M；总模型约 3.1M。
- 输出 stride 1/2（先 1/2，再 1/4，再最近邻上采样到 1/1），三头 1×1 卷积。

### 3.2 输出协议

严格按 `research/b05n_p2_mtl/deployment/stageB2a/output_contract.json`：

| 输出 | shape | 类别 |
|---|---|---|
| `lane_cat` | 1×9×288×512（Seg-256：1×9×144×256） | crosswalk, double other, double white, double yellow, road curb, single other, single white, single yellow, background |
| `lane_style` | 1×3×H×W | solid, dashed, background |
| `drivable` | 1×3×H×W | direct, alternative, background |

- 全部为 logits，per-tensor affine INT8。
- 检测 12 路：`reg_s4/s8/s16/s32`、`cls_s4/s8/s16/s32`、`cls_sum_s4/s8/s16/s32`。
- ONNX 输入 1×3×288×512，NCHW，RGB，0–1。

### 3.3 损失

- lane：加权 CE + 8 类 Dice + 前景 Dice + 前景 Tversky（β>α 抑制粗线/假阳性）。
- style：仅在车道像素上算 CE/Dice；权重 `[1.0, 3.0, 0.2]`。
- drivable：CE/Dice；权重 `[1.0, 2.0, 0.5]`。
- B-2a/B-2b 联合训练时检测损失权重 0.5，检测头与 P2/P3 相关层选择性解冻。

代码：

- `research/b05n_p2_mtl/mtl/model.py`
- `research/b05n_p2_mtl/mtl/losses.py`
- `research/b05n_p2_mtl/mtl/data.py`
- `research/b05n_p2_mtl/mtl/label_render.py`

---

## 4. 分阶段联合训练

### 4.1 阶段 A / B-1：冻结检测，只训分割头

- 阶段 A：45,000 张，9 epoch，仅分割损失；用于跑通数据/损失/导出。
- B-1：全量 70k，14 epoch，检测分支完全冻结；best 按 `lane_fg_iou + crosswalk_iou + 0.5*style_acc + 0.5*drivable_miou` 选择。

| 指标 | 阶段 A | B-1 |
|---|---:|---:|
| 车道前景 IoU | 0.3726 | 0.3868 |
| 斑马线 IoU | 0.2440 | 0.2646 |
| 前景线型准确率 | 0.8378 | 0.8468 |
| 实线准确率 | 0.7798 | 0.7955 |
| 虚线准确率 | 0.9092 | 0.9103 |
| 可行驶区域 mIoU | 0.5828 | 0.5885 |

### 4.2 B-2a：选择性解冻 + 检测损失

- 从 B-1 best 接续，3 epoch / 70k，24.6 分钟。
- 解冻层：`[0,1,2,3,4,5,6,12,15,18,19,20,21]`，检测头参与训练，分层学习率（shared 2e-5 / head 5e-5 / seg 3e-4）。
- 检测损失 0.5，分割损失同 B-1。

### 4.3 B-2b：第二轮对照

- 从 B-2a best 接续，同样 3 epoch / 70k，24.8 分钟。
- 目的：确认分割是否继续上涨、检测/小交通灯是否出现不可接受回退。

### 4.4 三阶段对照

同一 1,000 张 val、同一指标：

| 指标 | B-1 | B-2a | B-2b |
|---|---:|---:|---:|
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

### 4.5 为什么最终选 B-2a

- B-2a 的检测 mAP50、traffic light AP50、BDD50 TL AP50 三项最好；分割相对 B-1 有明显提升。
- B-2b 的分割继续提升，但增幅减半（斑马线 +0.043 → +0.016），检测 mAP 回落 0.003，小交通灯 recall 继续下降（0.4524 → 0.4286 → 0.4167）。
- 按停止规则，B-2a 作为检测/TL 优先的部署版本；B-2b 保留为“分割最佳”对照，不替换正式模型。
- B-2b 的作用是证明继续联合训练会进入收益递减，并量化小目标回退，而不是“失败版本”。

### 4.6 证据

- `research/b05n_p2_mtl/runs/stageB1/`
- `research/b05n_p2_mtl/runs/stageB2a/`
- `research/b05n_p2_mtl/runs/stageB2b/`
- `research/b05n_p2_mtl/scripts/train_stage_a.py`
- `research/b05n_p2_mtl/scripts/train_stage_b2a.py`
- `research/b05n_p2_mtl/scripts/eval_mtl.py`
- `research/b05n_p2_mtl/scripts/eval_det_map.py`
- `research/b05n_p2_mtl/scripts/eval_tl_50.py`

---

## 5. 量化问题与最终方法

### 5.1 旧 post-DFL split 的问题

- 输出 `boxes[1,4,N]` / `scores[1,C,N]` 的坐标范围被少量离群 anchor 拉大：P2 box 量化步长 3.73 px，logit 步长 0.339。
- 8–16 px 的交通灯在 512 宽输入下只有 3–6 px，3.73 px 的步长超过目标本身。
- `accuracy_analysis` 显示 single-layer cosine 多数 ≥0.999，但 entire cosine 累积下降；最差在 neck/检测头 P2 分支。
- 只把输出/检测头改 FP16 无效；输出裁剪只恢复约 1/4 的损失。

### 5.2 最终方法：Raw-head 导出

- DFL/坐标解码移到 host；每尺度独立输出，避免 concat 后坐标主导量化范围。
- 分类分数保留在更合适的域；12 路输出分别量化。
- 结果：P2 INT8 0.3651 / TL 0.4733，接近 ONNX/FP16；BDD 仍有约 0.8/1.3 点差距，不能宣称所有模型无损。
- 代价：host 端 DFL/NMS 更重；C API 把这段后处理放到 C，降低 Python 开销。

### 5.3 B-2a INT8 对照

Toolkit2 模拟器、10 张、与 PyTorch FP32 的一致性（不是 GT 精度）：

| 指标 | FP16 | INT8 |
|---|---:|---:|
| 车道前景 IoU | 0.9992 | 0.9302 |
| 斑马线 IoU | 0.9990 | 0.8875 |
| 前景线型准确率 | 0.9994 | 0.9163 |
| 实线准确率 | 0.9990 | 0.9659 |
| 虚线准确率 | 0.9996 | 0.8929 |
| 可行驶区域 mIoU | 0.9994 | 0.9264 |
| car 框匹配率 | 1.000 | 0.985 |
| person 框匹配率 | 1.000 | 1.000 |
| traffic light 框匹配率 | 1.000 | 0.967 |
| traffic light 分数变化 | ~0 | −0.0088 |

结论：INT8 检测端可用，交通灯没有明显退化；分割端可用，**lane_style 的虚线**是弱项（0.893）。B-2a INT8 的分割一致性高于 B-1 INT8（B-1：0.888/0.824/0.919/0.890），说明联合微调后的特征对量化更稳。

### 5.4 证据

- `research/b05n_p2_mtl/deployment/stageB2a/conversion_report.md`
- `research/b05n_p2_mtl/deployment/stageB2a/int8_analysis.json`
- `research/b05n_p2_mtl/deployment/stageB2a/fp16_analysis.json`
- `research/b05n_p2_mtl/deployment/stageB2a/output_contract.json`

---

## 6. Seg-256 输出优化

### 6.1 方法

- 在 ONNX 图内对三个分割输出各加 2×2 AveragePool（stride 2），输出 144×256；12 个检测输出不变。
- 权重不重新初始化；同一份 500 张校准。
- ONNX 校验：新分割输出与原 logits 的 2×2 均值池化最大差 3.8e-6；12 个检测输出逐元素一致。
- RKNN INT8 大小 4,933,694 B；分割张量 2,211,840 → 552,960 个值（4× 减少），float32 8.85 MB → 2.21 MB。

### 6.2 速度

| 指标 | B-2a 原版 | Seg-256 |
|---|---:|---:|
| RKNN 调用 p50 / p95 | 158.8 / 411.9 ms | **117.8 / 340.2 ms** |
| 分割后处理 p50 / p95 | 48.8 / 63.9 ms | **14.7 / 23.2 ms** |
| 完整单帧 p50 / p95 | 244.6 / 512.2 ms | **161.1 / 436.2 ms** |
| 连续 FPS | 3.55 | **4.79** |

相对原版：FPS +34.9%，单帧 p50 −34.1%，RKNN 调用 −41.0 ms（−25.8%）。

### 6.3 一致性

60 帧，恢复到 512×288 后对比原 B-2a INT8：

| 指标 | 值 |
|---|---:|
| 车道前景 IoU | 0.794 |
| 斑马线 IoU | 0.768 |
| 车道内线型一致率 | 0.963 |
| 实线一致率 | 0.920 |
| 虚线一致率 | **0.978** |
| 可行驶区域 mIoU | 0.872 |

结论：Seg-256 把分割后处理降到 1/3，但边界一致性约 0.77–0.79；线型（尤其虚线）保持较好。精度优先时不要用 Seg-256 替换原分割；速度优先时作为 MTL-FAST 可选方案。

### 6.4 证据

- `research/b05n_p2_mtl/deployment/stageB2a/seg256/seg256_report.md`
- `research/b05n_p2_mtl/deployment/stageB2a/seg256/output_contract_seg256.json`
- `research/b05n_p2_mtl/deployment/stageB2a/seg256/make_seg256_onnx.py`
- `research/b05n_p2_mtl/deployment/stageB2a/seg256/bench_seg_experiments.json`
- `research/b05n_p2_mtl/deployment/stageB2a/seg256/consistency_extra.json`

---

## 7. RKNN C API 部署方法

### 7.1 为什么用 C API

Python `RKNNLite.inference()` 在 B-2a 上约 161 ms RKNN 调用；其中大量时间在输出取回/反量化。C API 直接 `rknn_run` + `rknn_outputs_get(want_float=0)`，把 Raw-head 解码、NMS、分割 argmax 放在 C。

### 7.2 库接口

- `p2_create` / `p2_destroy`：持久化 context。
- `p2_detect`：12 输出检测模型。
- `p2_detect_mtl` / `p2_seg_size`：15 输出 MTL 模型。
- `P2Result`：最多 1000 框，`cls/score/x1/y1/x2/y2`。
- `masks`：3×H×W uint8 argmax。

### 7.3 INT8 原生输出

- `want_float=0`，输出 dtype INT8 NCHW per-tensor affine。
- 同一像素的 9/3/3 个通道共享 scale/zp，直接跨通道 int8 argmax 与反量化后 argmax 等价。
- 60 帧、8,847,360 像素：lane_cat / lane_style / drivable 一致率均为 1.000000。
- 检测框数量与 Python 差异约 0.4%，来自 C 双线性预处理与 OpenCV 的细微差别，不是解码/量化布局错误。

### 7.4 性能

独立 C API，5 轮 ×60 帧 = 300 次：

| 模型 | want_float | RKNN p50 | 单帧 p50 | 连续 FPS |
|---|---|---:|---:|---:|
| B05N P2 | 0 | 23.9 ms | 49.0 ms | **14.33** |
| B-2a 原版 | 0 | 87.6 ms | 161.2 ms | **4.95** |
| B-2a Seg-256 | 0 | 80.5 ms | 122.3 ms | **5.97** |

对比 Python：B05N 46.0 ms → 23.9 ms；B-2a 161.2 ms → 87.6 ms。`want_float=0` 把 B-2a 输出获取从 39.0 ms 降到 11.2 ms（−71.3%）。

### 7.5 剩余瓶颈

- B-2a 原版：`rknn_run` 87.6 ms（最大），分割 argmax 34.1 ms，预处理 16.4 ms，输出获取 11.2 ms。
- Seg-256：`rknn_run` 80.5 ms（66%），预处理 17.7 ms，输出获取 8.0 ms，分割 argmax 7.4 ms。
- 继续优化应优先模型执行/预处理，而不是继续降分割分辨率。

### 7.6 证据

- `deployment/lubancat/capi/p2capi.c`
- `deployment/lubancat/capi/README.md`
- `research/b05n_p2_mtl/deployment/stageB2a/capi/capi_report.md`
- `research/b05n_p2_mtl/deployment/stageB2a/capi/capi_bc_report.md`
- `research/b05n_p2_mtl/deployment/stageB2a/capi/bench_*.log`

---

## 8. 消融与对照数据

### 8.1 检测 / 分割阶段

| 实验 | 目的 | 关键结果 |
|---|---|---|
| 阶段 A | 冻结检测，跑通分割 | 车道 0.3726 / 斑马线 0.2440 / 可行驶 0.5828 |
| B-1 | 全量数据只训分割 | 0.3868 / 0.2646 / 0.8468 / 0.5885 |
| B-2a | 选择性解冻 + 检测损失 | 0.4281 / 0.3079 / 0.8756 / 0.6480；mAP 0.3707；TL 0.4997 |
| B-2b | 再 3 epoch 对照 | 0.4396 / 0.3236 / 0.8830 / 0.6683；mAP 0.3667；TL 0.4962 |

### 8.2 量化 / 输出结构

| 实验 | 结果 |
|---|---|
| post-DFL split INT8 | P2 mAP 0.3031 / TL 0.2596 |
| Raw-head INT8 | P2 mAP 0.3651 / TL 0.4733 |
| B-2a FP16 模拟器 | 分割 0.9992–0.9996，检测匹配 1.0 |
| B-2a INT8 模拟器 | 分割 0.8875–0.9659，TL 匹配 0.967 |
| 原生 int8 argmax | 与 float 100% 一致 |

### 8.3 分割输出 / 后处理

| 实验 | FPS | 单帧 p50 |
|---|---:|---:|
| B-2a 原版 Python | 3.41 | 232.3 ms |
| B-2a 跳过分割后处理 | 4.55 | 156.4 ms |
| B-2a Seg-256 Python | 4.79 | 161.1 ms |
| B-2a C API 原版 | 4.95 | 161.2 ms |
| B-2a Seg-256 C API | **5.97** | **122.3 ms** |

### 8.4 完整 AI Bridge 回放

本地 60 帧、10 Hz 请求、正式 bridge + Qt/HUD 负载：

| 模型 | 结果 FPS | 主模型 p50 / p95 | RKNN p50 / p95 | 解码 p50 / p95 |
|---|---:|---:|---:|---:|
| B | 8.56 | 66.2 / 209.3 ms | 50.2 / 167.4 ms | 9.5 / 16.2 ms |
| B-CAPI | **9.39** | **44.0 / 138.7 ms** | **23.0 / 53.2 ms** | **2.3 / 3.1 ms** |
| MTL-FULL | 5.00 | 152.9 / 412.4 ms | 78.3 / 170.5 ms | 23.5 / 59.5 ms |
| MTL-FAST | 6.78 | 104.4 / 248.5 ms | 72.3 / 173.1 ms | 6.3 / 7.2 ms |

**不要把这组数字写成 PCIe 实时整机性能**：它是本地 60 帧回放 + 10 Hz 请求 + 同时运行 Qt/HUD 的条件。独立 C API 的 14.33/4.95/5.97 FPS 也不能直接与 bridge FPS 混用。

**MTL-FAST 的 6.78 FPS 是全窗口平均，不是峰值。** 同一套正式代码在 live Qt 中
显示 AI FPS **8.3**，用户观测峰值 **9.9**；20 s 日志结果率 7.51 FPS，
result delta p50/p95 130.7/155.0 ms，pipeline p50/p95 118.7/133.9 ms。
基准的 6.78 受 64 次 latest-only 替换和 p95 248.5 ms 抖动影响；模型 p50
104–119 ms 对应约 8.4–9.6 FPS，短窗口 9.9 合理。原始记录：
`docs/deployment/evidence/MTL/live_mtl_fast_20261031.json`。

### 8.5 消融结论

1. Raw-head 是 INT8 精度的关键，不是后处理技巧。
2. B-2a 的检测/交通灯最好，B-2b 的分割最好但收益递减。
3. Seg-256 是“速度优先”选项，精度代价主要在边界；虚线一致性反而最好。
4. C API + `want_float=0` 是 B-2a 上板可用的必要条件；Python 3.4 FPS 无法满足交互。
5. 完整 bridge 的瓶颈是模型执行本身，不是分割传输；Qt 叠加层开销在噪声范围内。

---

## 9. 已完成的系统集成

### 9.1 AI Bridge

- `model_profiles.py`：21 个方案；`backend`（`rknn_lite` / `rknn_capi`）、`task`（`detect` / `mtl`）。
- `ai_bridge.py`：初始化和热切换传递 backend/task；`same_main` 同时比较路径、backend、task；profile 匹配用 `absolute()`，避免符号链接被误判。
- `b05p_competition_bridge.py`：按 backend/task 选择 `CapiDetector` / `CapiMtlDetector`；类别归一化（pedestrian→person 等）。
- MVA2 状态包：模型选项改为紧凑 `wire()` 格式 + 脏标记/每 50 包重发，避免 4096 字节上限；超限时回退到无 options 状态，保证推理不中断。
- 切换失败保留原检测器；成功后释放旧 RKNN context。

### 9.2 热切换实测

| 切换 | 结果 | 耗时 |
|---|---|---:|
| B → B-CAPI | APPLIED | 204.7 ms |
| B-CAPI → MTL-FULL | APPLIED | 296.0 ms |
| MTL-FULL → MTL-FAST | APPLIED | 288.0 ms |
| MTL-FAST → B | APPLIED | 1028.3 ms |
| B → MTL-FAST（Qt 直播回放） | APPLIED | 158.9 ms |
| MTL-FAST → MTL-FULL（Qt） | APPLIED | 290.6 ms |
| MTL-FULL → B（Qt） | APPLIED | 1103.6 ms |

### 9.3 分割通道

- C 检测器把 3 张 mask 写入 `/dev/shm/adas_mtl_overlay.png`（RGBA PNG）和 `.json` 元数据；`os.replace` 原子替换。
- PNG 编码在后台线程完成，合并最新帧；检测主路径 `overlay_ms=0`。
- Qt QML `Image` 以 160 ms 周期、URL fragment `#g=` 刷新；query string 在 Qt file URL 下不渲染，必须用 fragment。
- 颜色：斑马线洋红、direct 绿、alternative 蓝、实线红、虚线青。
- 覆盖层跟随道路象限；viewMode=1 时只占对应象限；切回非 MTL 立即隐藏并删除 shm 文件。
- 截图：`docs/hmi/assets/mtl/screenshot_mtl_fast.png`、`screenshot_mtl_full.png`、`screenshot_b_overlay_cleared.png`。

### 9.4 Qt 性能

本地 PGV 回放 + llvmpipe：

| 指标 | B（无覆盖层） | MTL-FAST（有覆盖层） |
|---|---:|---:|
| scene_graph_fps p50 / p95 | 4.2 / 6.7 | 3.9 / 6.2 |
| source_fps p50 | 16.2 | 15.2 |
| CPU（全核）p50 | 52.0% | 46.9% |
| AI 结果更新 p50 | 17.5 ms | 15.7 ms |

覆盖层开销在噪声范围内。PNG 编码：FULL 36.9 ms，FAST 9.9 ms（后台线程）。

---

## 10. 局限与后续方向

### 10.1 模型局限

- B-2a 小交通灯召回低于 B05N P2（60 帧：含灯帧 44 vs 57，框数 63 vs 122；FP16 也只有 47/60）。这是模型级退化，不是单纯 INT8。
- 分割边界仍偏粗，斑马线与白线有混淆，夜间/逆光漏检；Seg-256 边界一致性约 0.77–0.79。
- B-2b 分割继续提升但检测/小灯回退，说明联合训练已进入收益递减。
- 训练数据不是完整 BDD 70k 检测集（8.5k P2 训练），P2 与正式 BDD 不是数据公平对比。

### 10.2 系统局限

- 当前无 PCIe 输入，Qt/AI 的端到端验证使用本地 PGV 回放；真实 PCIe 恢复后需复测。
- Qt 使用 llvmpipe 软件 OpenGL，scene_graph_fps 约 4 FPS；不能代表 Mali 正常显示上限。
- 分割覆盖层是 2D 叠加，未转换为 Tesla 3D 道路几何。
- MTL-FULL 在完整 bridge 下 5.0 FPS，MTL-FAST 6.78 FPS；10 Hz 请求下有大量 latest-only 替换，交互帧率需按实际场景验收。

### 10.3 后续方向

1. 补 rider/motor/bike 与小交通灯数据，提升 B-2a 的弱势类别。
2. 在不增加分割分辨率的前提下改 decoder 容量/特征融合，改善斑马线边界。
3. 把分割结果用于 3D 道路边界/Tesla BEV，而不是只做 2D 叠加。
4. 优化 C 预处理（RGA）和 `rknn_run`，目标稳定 ≥6 FPS。
5. 用真实 PCIe 四路视频做长时间稳定性、温度和丢帧测试。
6. 评估 GitHub Release / 对象存储保存权重与演示视频。

---

## 11. 复现入口

| 内容 | 路径 |
|---|---|
| B05N P2 训练/评估 | `research/b05n_p2/tools/`、`research/b05n_p2/reports/` |
| B-2a 训练/评估 | `research/b05n_p2_mtl/scripts/`、`mtl/` |
| B-2a 阶段结果 | `research/b05n_p2_mtl/runs/stageB{1,2a,2b}/` |
| RKNN 转换 | `research/b05n_p2_mtl/deployment/stageB1/`、`stageB2a/` |
| Seg-256 | `research/b05n_p2_mtl/deployment/stageB2a/seg256/` |
| C API | `deployment/lubancat/capi/`、`research/b05n_p2_mtl/deployment/stageB2a/capi/` |
| AI Bridge | `deployment/lubancat/qt_gpu_probe/` |
| Qt 叠加 | `deployment/lubancat/qt_gpu_probe/main.qml`、`docs/hmi/MTL_SEGMENTATION_OVERLAY.md` |
| 模型清单 | `artifacts/manifests/model_manifest.json` |
| 部署与性能 | `docs/deployment/MTL_CAPI_DEPLOYMENT.md` |
| 多模型架构 | `docs/architecture/MULTIMODEL_AI_BRIDGE.md` |
