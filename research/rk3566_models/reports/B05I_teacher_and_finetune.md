# B-05I — 比赛视频自动标注升级与 Specialist 首轮微调

日期：2026-10-24  
分支：`research/rk3566-adas-models`  
基线：`3a497b7dd34a85791ee846857128f416dc259a12`

## 0. 结论

已完成：

- 修复 B-05H 的待审核/空标签污染；
- 43 个 B 候选逐一审阅：26 ACCEPT、17 REJECT、C=0；
- 得到 30 个 A 框（15 张干净图），来自多段比赛视频；
- 用原 Specialist 的 `best.pt` 源 checkpoint 实际完成首轮 1 epoch 微调；
- 固定 BDD50/154 GT 独立评估。

未完成/未通过：

- Grounding DINO、Florence-2 未实际运行；
- 新模型相对旧模型只有 **AP50 +1.7 点**，Recall **-0.7 点**，Precision **-6.8 点**；
- 8–16px Recall 没有改善；
- 新模型不值得直接 RKNN 部署。

## 1. P0：修复数据集错误

新增干净数据集：

```text
D:\Datasets\adas_edge_training\b05i_clean_20261024\pseudo_A
```

规则：

- 只有被确认的 A 框所在图片才进入主训练集；
- 不再生成空标签图片；
- B 组不再作为空负样本；
- 17 个明确非灯候选只保留在 review JSON，不写空标签；
- 未独立确认的图片不做负样本；
- 小灯/远灯不因旧模型漏检自动当背景。

审计：

```text
主标签图片：15
主标签框：30
空标签：0
越界框：0
```

## 2. P1：独立教师

尝试环境：

```text
GPU：RTX 4060 Laptop 8GB
CUDA torch：2.7.1+cu126
transformers：4.57.3
```

实际结果：

- Grounding DINO：**未运行**；
- Florence-2：**未运行**；
- 没有可用预训练权重下载/加载记录，未伪造教师输出。

替代审核：

- 用图像理解模型（本 Codex 视觉能力）对 43 个 B 裁剪图逐张审阅；
- 结果保存在：

```text
tools/b05i_clean_dataset.py
tools/b05i_review_sheets.py
D:\Datasets\adas_edge_training\b05h_auto_annotation_20261024\meta\B_review_1..3.jpg
```

这不是 Grounding DINO/Florence 的替代精度声明。

## 3. P2：43 个 B 候选

```text
B total：43
ACCEPT：26
REJECT：17
UNCERTAIN：0
```

ACCEPT 包括明显的绿灯、红灯、黄灯灯体；REJECT 包括路牌、车身/车辆灯、杆件和不可确认的小区域。  
保存了修正前后坐标/审核依据：

```text
D:\Datasets\adas_edge_training\b05h_auto_annotation_20261024\meta\pseudo_labels.json
```

## 4. P3/P4：数据集

### 新增可信标签

```text
B accepted：26
原 A：4
最终 A：30
```

A 组：

```text
图片：15
train 11 / val 10 / test 9 框
colors：GREEN 25, RED 4, UNKNOWN 1（HSV_UNVERIFIED）
sizes：8–16px 5, 16–32px 14, >32px 11, <8px 0
```

### 原有人工数据

```text
tl_specialist/data_512x192
train images：8513
val images：487
labels：traffic light 单类
```

### 独立 BDD50

```text
50 images / 154 human traffic-light GT
只用于独立评估，不进入训练
```

## 5. P5：实际训练

源 checkpoint：

```text
D:\Datasets\adas_edge_training\tl_specialist\tl-512x192\weights\best.pt
```

训练配置：

```text
imgsz=512, rect=True, batch=16, epochs=1, seed=0
device=RTX 4060
baseline: 8513 human images only
candidate: 8513 human + 15 domain images / 30 A boxes
```

实际输出：

```text
D:\Datasets\adas_edge_training\b05i_runs\baseline\weights\best.pt
D:\Datasets\adas_edge_training\b05i_runs\candidate\weights\best.pt
```

训练日志：

```text
research/rk3566_models/reports/assets/B05I_baseline_train.log
research/rk3566_models/reports/assets/B05I_candidate_train.log
```

## 6. P6：独立 BDD50 评估

固定 50 图/154 GT，conf 0.25，IoU 0.5：

| 模型 | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|
| source best.pt | **0.644** | **0.433** | 0.402 | 0.126 |
| baseline (1 epoch) | 0.480 | 0.387 | 0.368 | 0.115 |
| candidate (1 epoch) | 0.576 | 0.427 | **0.419** | **0.139** |

解释：

- candidate AP50 比 source 高 1.7 点；
- 但 Recall 低 0.7 点、Precision 低 6.8 点；
- baseline 因为只训 1 epoch 明显退化；
- 不能把 AP 小幅上升包装成全面提升。

### 尺寸 Recall

| 模型 | <8px | 8–16px | 16–32px |
|---|---:|---:|---:|
| source | 0.342 | 0.630 | 0.667 |
| candidate | **0.417** | 0.630 | 0.333 |

- <8px 提升 7.5 点；
- 8–16px 没有改善；
- 16–32px 下降。

### 速度

CUDA 测试：

```text
source inference：2.28 ms
candidate inference：0.75 ms
```

这是 4060 GPU 的 PyTorch 数字，不是 RK3566 RKNN；不能直接作为部署延迟。

## 7. P9：RKNN

未执行 RKNN 转换。  
原因：候选模型没有显示明确 Recall/Precision 收益，按任务约束不进入转换/部署；原 `tl_specialist_512x192_hybrid2.rknn` 未覆盖。

## 8. 回答交付问题

1. **Grounding DINO/Florence-2**：都没有成功运行；没有可用权重/运行记录。
2. **43 B 候选**：26 ACCEPT，17 REJECT，0 UNCERTAIN，C=0。
3. **新增高可信框**：26 个，A 总数 30；仍不足 50–100 目标。
4. **补小灯**：新增 8–16px 5 个，没有新增 <8px；不能声称远距离小灯覆盖充分。
5. **空标签污染**：已消除；主数据集 15 张图、30 框，无空标签。
6. **原始训练数据**：8513 train/487 val 可正常训练；本轮实际完成。
7. **首轮训练**：完成 1 epoch baseline/candidate。
8. **新旧指标**：candidate AP50 0.419 vs source 0.402；Recall 0.427 vs 0.433；Precision 0.576 vs 0.644。
9. **8–16px**：没有改善（0.630 vs 0.630）；<8px 改善但 16–32px 退化。
10. **是否值得 RKNN 部署**：不值得；没有明确综合收益，保持原模型。

## 9. 交付

```text
tools/b05i_clean_dataset.py
tools/b05i_review_sheets.py
tools/b05i_build_train_data.py
tools/b05i_train.py
tools/b05i_eval.py
tools/b05i_size_eval.py
research/rk3566_models/reports/B05I_teacher_and_finetune.md
research/rk3566_models/reports/assets/B05I_eval.json
research/rk3566_models/reports/assets/B05I_size_eval.json
research/rk3566_models/reports/assets/B05I_dataset_index.json
research/rk3566_models/reports/assets/B05I_*_train.log
```

数据集和候选权重保留本地训练目录，不上传 GitHub。

本轮未修改 RK3566 正式模型、Qt、FPGA 或 PCIe。
