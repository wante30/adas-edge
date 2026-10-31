# 模型训练报告（B01 → B05P）

> 2026-10-31 更新：正式默认模型已切为 **B（B05N P2）**；第 1–5 节为 2026-10-28 的 A 方案选型记录，B-2a/MTL 见第 6 节。


更新日期：2026-10-28  
范围：BDD100K 道路检测、YOLOv8n/P2 对比、交通灯 Specialist、RKNN Raw-head 量化前后精度。  
原则：只记录能从权重、训练元数据、评估 JSON 或报告中追溯的数字；PC、RKNN 仿真、RK3566 实板分开标注。

---

## 0. 结论

1. **正式主模型仍是 BDD-YOLOv8n 10 类**（`bdd100k-yolov8n-best.pt`，SHA256 `acd17780...72fb2`）。它在 1,000 张 BDD holdout 上 mAP50 `0.3831`，高于 B05N P2 的 `0.3711`。
2. **B05N P2 10 类没有整体超过 BDD 主模型**：P2 在 car/person/traffic light/traffic sign 上更好，但 rider/motor/bike/truck/bus 退化，总体 mAP50 低约 1.2 点。
3. **B05L P2 Specialist 在交通灯小目标上明显更强**：BDD50 + Resize 达到 AP50 `0.6107`、Recall `0.5800`、8–16px Recall `0.5750`；但它在 Cityscapes 独立集上带来大量交通灯 FP。
4. **2026-10-27 的四方案独立评测最终选择 A（BDD 单模型 Raw-head INT8）作为默认**，原因是 Cityscapes Fine Val 500 上 A 的 mAP50 最高（`0.2822`）、总 FP 最低（`1,417`），C/D 的交通灯 FP 分别达到 `1,554/1,668`。
5. **Raw-head 导出显著修复了 INT8 精度**：P2、Specialist 的 Raw-head INT8 已接近 ONNX/FP16；BDD 仍有约 0.8 点 mAP50 / 1.3 点 TL AP50 的差距。
6. **量化精度与板端运行是两件事**：当前板端运行的是 BDD Raw-head INT8（A），实时 10 Hz；这不等于 B05N/B05L 双模型已在板端默认运行。

---

## 1. 数据与任务

### 1.1 主检测任务（BDD 10 类）

类别顺序固定为：

```text
0 person
1 rider
2 car
3 truck
4 bus
5 train
6 motor
7 bike
8 traffic light
9 traffic sign
```

BDD 镜像映射：`pedestrian → person`、`motorcycle → motor`、`bicycle → bike`；`other vehicle / trailer / other person` 忽略。

### 1.2 数据划分

B05N/B05L 主要使用 BDD100K **validation 10k 全画面镜像**的派生 split：

| 划分 | 图像 | 用途 | 说明 |
|---|---:|---|---|
| train | 8,513 | 训练 | 复用 B05L 训练名单 |
| val | 482 | 验证/早停 | B05L clean val，排除近重复 |
| test | 1,000 | 独立 holdout | B03D holdout-1000 |
| BDD50 | 50 | 交通灯专项 | 人工精修 GT；154 个 TL 框，ROI 内 150 |

B05N 数据统计：

| 划分 | 图像 | 框数 | 交通灯框 | 8–16px 交通灯 |
|---|---:|---:|---:|---:|
| train | 8,513 | 103,133 | 22,935 | 11,374 |
| val | 482 | 9,041 | 1,208 | 620 |
| test | 1,000 | 18,450 | 2,734 | 1,345 |

B05L Specialist 数据：

| 项目 | train | val（原始） | clean val |
|---|---:|---:|---:|
| 图像 | 8,513 | 487 | 482 |
| 有标签图像 | 4,719 | 249 | 244 |
| 空标签图像 | 3,794 | 238 | 238 |
| 标注框 | 22,283 | 1,185 | — |

Specialist 输入是原图顶部 60% ROI（1280×432）直接缩放到 `512×192`，**不是 Letterbox**。训练/部署预处理必须一致；B05L 报告显示 P2 在 Resize 下 AP50 `0.6107`，Letterbox 只有 `0.4853`。

### 1.3 独立方向性验证集

B05P 使用 Cityscapes Fine Val 500 张作为独立方向性验证：

- revision：`b7e3f28b5aba95b55de608c7e94c952ea3966d64`
- 清单 SHA256：`19a30f0bcf80bbd65230b65467c07452b0bb12b4169ea407e4e42d50b4491514`
- 欧洲城市域，**不等同 BDD 或比赛视频**，不能替代比赛域最终验证。

---

## 2. 模型与训练配置

### 2.1 BDD 主模型

| 项目 | 值 |
|---|---|
| 权重 | `bdd100k-yolov8n-best.pt` |
| SHA256 | `acd17780b331c3970e22faf3c6f123eae8b8c4abcc736ad5ef38ff28ecd72fb2` |
| 架构 | YOLOv8n，10 类 |
| 评估输入 | 512×288 |
| 训练数据 | 比 B05N 更大的 BDD 训练集（具体配置未在本机完整留存） |
| 已知指标 | BDD holdout-1000 mAP50 `0.3831`，mAP50-95 `0.2102` |
| 当前角色 | 正式主模型；Raw-head INT8 后作为板端 A 方案 |

> BDD 主模型的完整训练日志/超参不在当前可访问目录中，本文不补写未知配置。可追溯的是权重 hash、评估 JSON 和 B05N 报告中的对比结果。

### 2.2 B05N P2 10 类

| 项目 | 值 |
|---|---|
| 架构 | YOLOv8n-P2（Ultralytics `yolov8-p2.yaml`） |
| 初始化 | BDD 主模型 `bdd100k-yolov8n-best.pt` |
| 类别 | 10 |
| 输入 | 512（rect=True → 512×288） |
| epochs | 60 |
| batch | 32 |
| seed | 0 |
| patience | 15（P2）/ 100（公平对照） |
| close_mosaic | 10 |
| AMP | on |
| GPU | RTX 3080 Ti 12GB（租用）+ RTX 4060 Laptop 8GB |
| PyTorch/CUDA | 2.12.1+cu126 / 12.6（4060） |
| Ultralytics | 8.4.136 |

初始化审计：P2 与标准 YOLOv8n 的 neck/head 索引不同，`YOLO.load()` 日志为 219/437、389/437 items；训练前精确同值检查只有 **43.5% 参数**。P2 不能视为完整预训练迁移。

训练结果：

| 模型 | 设备 | epochs | 训练时间 | val mAP50 | val mAP50-95 |
|---|---|---:|---:|---:|---:|
| P2 10 类 | RTX 3080 Ti | 60 | 1,624 s | 0.3512 | 0.1887 |
| **P2 10 类（最终）** | **RTX 4060 Laptop** | **60** | **4,926 s** | **0.3690** | **0.1992** |
| 标准 YOLOv8n 对照（16 epoch 早停） | RTX 3080 Ti | 16 | 336 s | 0.3471 | 0.1918 |
| 标准 YOLOv8n 对照（60 epoch 公平版） | RTX 3080 Ti | 60 | 1,243 s | 0.3668 | 0.2025 |

> 两台机器同 seed/同配置仍有约 1.8 点 val mAP50 差异；最终以 test 指标选择，不按 val 挑模型。  
> 训练元数据 `b05n_run_meta.json` 的 `results_dict` 记录为 val mAP50 `0.37137`、mAP50-95 `0.20045`；上表 `0.3690/0.1992` 来自 B05N 报告的评估轮次。两者可能对应不同 val 评估/快照，本文保留原报告值并在此标注差异，不把 val 当作最终选型依据。

### 2.3 B05L P2 Specialist

| 项目 | 值 |
|---|---|
| 架构 | YOLOv8n-P2，单类交通灯 |
| 初始化 | `yolov8n.pt` + P2 结构 |
| 输入 | 顶部 60% ROI → 512×192 |
| epochs | 60 |
| batch | 32 |
| seed | 0 |
| GPU | RTX 4060 Laptop 8GB |
| PyTorch/CUDA | 2.12.1+cu126 / 12.6 |
| 参数量 | 3,366,022（训练时） |
| best epoch | 55 |
| val mAP50 | 0.5250 |
| val mAP50-95 | 0.1740 |
| weights SHA256 | `c2d8dd9250393af9eae838f147e9fcaff8bae5c37b05252ea1c1286e9c8776a7` |

P2 初始化同样只匹配 219/437 层、43.5% 参数；Specialist 的收益不能简单归因于“完整预训练”。

### 2.4 标准 YOLOv8n 对照

同数据、同机器、同 Ultralytics/PyTorch 版本的严格对照只有一组：RTX 3080 Ti / torch 2.5.1+cu124 / Ultralytics 8.4.136，标准 YOLOv8n 60ep vs P2 10 类 60ep。

| 指标 | 标准 YOLOv8n 60ep | P2 10 类（3080） | P2 − 标准 |
|---|---:|---:|---:|
| mAP50 | 0.3660 | 0.3692 | +0.32 |
| mAP50-95 | 0.1939 | 0.1931 | −0.08 |
| car AP50 | 0.6685 | 0.7157 | +4.72 |
| person AP50 | 0.4298 | 0.4554 | +2.56 |
| rider AP50 | 0.2967 | 0.2680 | −2.87 |
| TL AP50 | 0.4795 | 0.4955 | +1.60 |
| 8–16px TL Recall | 0.4000 | 0.3875 | −1.25 |

结论：P2 架构相对同数据标准 YOLOv8n 总体基本持平；交通灯 AP 略高，但 8–16px Recall 没有提高。

---

## 3. 评估结果

### 3.1 BDD holdout-1000（10 类）

固定 conf=0.25、IoU=0.5，Ultralytics 标准验证：

| 模型 | mAP50 | mAP50-95 | Precision | Recall |
|---|---:|---:|---:|---:|
| 正式 BDD-YOLOv8n | **0.3831** | **0.2102** | 0.6754 | 0.3569 |
| 同数据标准 YOLOv8n（16ep） | 0.3662 | 0.1994 | 0.6464 | 0.3385 |
| 同数据标准 YOLOv8n（60ep） | 0.3660 | 0.1939 | — | — |
| P2 10 类（3080） | 0.3692 | 0.1931 | 0.5118 | 0.3497 |
| **P2 10 类（4060，最终）** | **0.3711** | **0.1944** | 0.6035 | 0.3580 |

分车型 AP50：

| 类别 | 正式 BDD | 标准同数据（60ep） | P2（4060） | P2 − BDD |
|---|---:|---:|---:|---:|
| person | 0.4389 | 0.4298 | 0.4528 | **+1.39** |
| rider | **0.3207** | 0.2967 | 0.2636 | **−5.71** |
| car | 0.6632 | 0.6685 | **0.7117** | **+4.85** |
| truck | **0.4960** | 0.4648 | 0.4686 | −2.74 |
| bus | **0.4443** | 0.4053 | 0.4059 | −3.84 |
| train | 0 | 0 | 0 | 0（test 只有 1 个 GT） |
| motor | **0.3705** | 0.2676 | 0.2718 | **−9.87** |
| bike | 0.2031 | 0.2323 | 0.1585 | −4.46 |
| traffic light | 0.4283 | 0.4354 | **0.4834** | **+5.51** |
| traffic sign | 0.4665 | 0.4600 | **0.4948** | **+2.83** |

### 3.2 BDD50 交通灯专项

conf=0.25、IoU=0.5、一对一匹配，ROI 内 150 GT：

| 模型 | TL AP50 | TP/FP/FN | Recall | 8–16px Recall |
|---|---:|---:|---:|---:|
| 正式 BDD（全画面） | 0.4622 | 65/17/85 | 0.4333 | 0.3500 |
| 标准 YOLOv8n 同数据（16ep） | 0.4598 | 62/24/88 | 0.4133 | 0.3000 |
| 标准 YOLOv8n 同数据（60ep） | 0.4795 | 68/24/82 | 0.4533 | 0.4000 |
| P2（3080） | 0.4955 | 68/24/82 | 0.4533 | 0.3875 |
| **P2（4060，最终）** | **0.5205** | **68/23/82** | **0.4533** | **0.4125** |
| **B05L 单类 P2 Specialist（top60 ROI）** | **0.6107** | **87/26/63** | **0.5800** | **0.5750** |

### 3.3 B05P Raw-head INT8（800 张 test 子集）

| 模型 / 格式 | mAP50 | TL AP50 | 备注 |
|---|---:|---:|---|
| BDD post-DFL split INT8，KL 500 | 0.3184 | 0.2406 | 旧基线 |
| BDD Raw-head INT8，KL 500 | 0.3606 | 0.3752 | 当前 A 模型 |
| BDD ONNX FP32 | 0.3688 | 0.3880 | 参考 |
| P2 post-DFL split INT8，normal 500 + 裁剪 | 0.3031 | 0.2596 | 旧基线 |
| P2 Raw-head INT8，normal 500 | 0.3651 | 0.4733 | B 方案 |
| P2 Raw-head ONNX FP32 | 0.3646 | 0.4766 | 参考 |
| Specialist post-DFL split INT8 | 0.2421 | — | 旧基线 |
| Specialist Raw-head INT8 | **0.6119** | — | BDD50 |
| Specialist Raw-head FP16 | 0.6109 | — | BDD50 |
| Specialist Raw-head ONNX | 0.6099 | — | BDD50 |

### 3.4 Cityscapes Fine Val 500（独立方向性）

| 方案 | mAP50 | mAP50-95 | TL AP50 | TL TP/FP/FN | TL Precision / Recall |
|---|---:|---:|---:|---:|---:|
| A BDD | **0.2822** | 0.1564 | 0.1486 | 200 / **123** / 1461 | **0.619 / 0.120** |
| B P2 | 0.2761 | 0.1490 | 0.1567 | 260 / 285 / 1401 | 0.477 / 0.157 |
| C BDD+Spec | **0.2852** | **0.1568** | 0.1790 | 409 / 1,554 / 1252 | 0.208 / 0.246 |
| D P2+Spec | 0.2790 | 0.1495 | **0.1860** | 419 / 1,668 / **1242** | 0.201 / **0.252** |

总 TP/FP/FN：

| 方案 | 总 TP | 总 FP | 总 FN |
|---|---:|---:|---:|
| A | 4,566 | **1,417** | 11,383 |
| B | 5,167 | 2,683 | 10,782 |
| C | 4,775 | 2,848 | 11,174 |
| D | **5,326** | 4,066 | **10,623** |

结论：C/D 的交通灯召回更高，但 FP 数量级更高；A 仍是综合最稳的默认。

---

## 4. 训练结论与选型

### 4.1 为什么不是“P2 全面替代 BDD”

- P2 的 car/person/TL/sign 更好，但 rider/motor/bike/truck/bus 退化。
- P2 训练数据只有 8,513 张，BDD 主模型训练集更大；不是公平数据对比。
- P2 初始化只继承 43.5% 参数，neck/head 大量重初始化。
- P2 的 4 尺度输出对 RKNN 转换/解码更复杂。

### 4.2 为什么 Specialist 没有直接默认开启

- B05L P2 + Resize 在 BDD50 上 AP50 `0.6107`、8–16px Recall `0.5750`，确实强。
- 但在 Cityscapes 独立集，C/D 的交通灯 FP 分别 `1,554/1,668`，整体 mAP 只比 A 高 0.003/0。
- 比赛如果更看重交通灯召回，可把 C/D 作为验证后模式；默认仍用 A。

### 4.3 当前正式选型

```text
默认（2026-10-31）：B = B05N P2 10 类 Raw-head INT8
保留：A = BDD-YOLOv8n Raw-head INT8
保留：C = BDD + B05L Specialist
保留：D = B05N P2 + B05L Specialist
```

模型切换代码：`deployment/lubancat/qt_gpu_probe/model_profiles.py`、`ai_bridge.py`、`video_item.cpp`。  
2026-10-31 板端默认已切为 B；A/C/D 与历史方案保留。UI 状态显示不一致的问题在本次模型注册/状态包更新中处理。

---

## 5. 证据索引

| 证据 | 路径 |
|---|---|
| B05N 对比报告 | `docs/models/evidence/B05N/summary.json`；原始报告在 B05N worktree `research/rk3566_models/reports/B05N_multiclass_P2_comparison.md` |
| B05L 选型报告 | `docs/models/evidence/B05L/BDD50_summary_resize.json`；原始报告 `B05L_specialist_model_selection.md` |
| B05P 架构选型 | `docs/models/evidence/B05P/B05P_bridge_runtime_20261027_{A,B,C,D}.json`；原始报告 `B05P_architecture_selection_20261027.md` |
| 训练元数据 | `D:\Datasets\adas_edge_training\b05n_runs\b05n_p2_bdd_full\b05n_run_meta.json` |
| 训练元数据 | `D:\Datasets\adas_edge_training\b05l_runs\v8n_p2\b05l_run_meta.json` |
| 模型 hash | `artifacts/manifests/model_manifest.json` |

> 训练权重、完整数据集、中间 checkpoint 不上传普通 Git；需要时按 manifest 路径从训练机取回并校验 SHA256。

---

## 6. 2026-10-31 更新：B-2a MTL 训练

> 第 1–5 节是 2026-10-28 的检测选型记录。此后新增了 B-2a 多任务分支；
> 正式默认模型在 2026-10-31 为 **B（B05N P2）**，A/C/D 保留。

### 6.1 分阶段

| 阶段 | 训练内容 | 关键结果 |
|---|---|---|
| 阶段 A | 冻结检测，只训分割头，45k 子集 | 车道 0.3726 / 斑马线 0.2440 |
| B-1 | 全量 70k，检测冻结，14 epoch | 0.3868 / 0.2646 / 线型 0.8468 / 可行驶 0.5885 |
| B-2a | 选择性解冻 P2/P3 + 检测头 + 检测损失，3 epoch | 0.4281 / 0.3079 / 0.8756 / 0.6480；mAP 0.3707 |
| B-2b | 从 B-2a 再接 3 epoch（对照） | 0.4396 / 0.3236 / 0.8830 / 0.6683；mAP 0.3667 |

B-2a 最终被选为部署版本：检测 mAP、traffic light AP、BDD50 TL AP 最好；B-2b 分割更好，但检测和小交通灯回退，作为“分割最佳”对照保留。

### 6.2 训练代码

- `research/b05n_p2_mtl/mtl/model.py`：B05NMTL + SegDecoder。
- `research/b05n_p2_mtl/mtl/losses.py`：CE / Dice / 前景 Dice / Tversky。
- `research/b05n_p2_mtl/mtl/data.py`、`label_render.py`：BDD lane/drivable mask。
- `research/b05n_p2_mtl/scripts/train_stage_a.py`：阶段 A / B-1。
- `research/b05n_p2_mtl/scripts/train_stage_b2a.py`：B-2a / B-2b 联合微调。
- `research/b05n_p2_mtl/scripts/eval_mtl.py`、`eval_det_map.py`、`eval_tl_50.py`。

### 6.3 证据

- `research/b05n_p2_mtl/runs/stageB1/`
- `research/b05n_p2_mtl/runs/stageB2a/`
- `research/b05n_p2_mtl/runs/stageB2b/`
- `docs/models/evidence/MTL/`
- `docs/research/MTL_RESEARCH_SUMMARY.md`
