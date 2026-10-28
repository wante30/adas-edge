# 数据集与评估口径

更新日期：2026-10-28  
目的：让每个模型指标都能追溯到数据集、划分、预处理、评估命令和硬件；避免把 PC、RKNN 仿真、RK3566 实板混为一谈。

---

## 1. 数据集总览

| 数据集/素材 | 用途 | 规模 | 位置 | 备注 |
|---|---|---|---:|---|
| BDD100K validation 10k 镜像 | 主训练/验证/测试 | 10,000 张 1280×720 | `D:\Datasets\bdd100k_val_hf` / `bdd100k_val_hirundo` | 本机没有完整 70k train |
| B05N/B05L 派生 split | 训练/验证/测试 | 8,513 / 482 / 1,000 | `D:\Datasets\adas_edge_training\b05n_bdd_multiclass`、`b05l_data` | 名称不重叠 |
| BDD50 | 交通灯专项 | 50 张 / 154 GT | `D:\Datasets\adas_edge_training\b05l_data` | 人工精修，ROI 内 150 |
| Cityscapes Fine Val 500 | 独立方向性验证 | 500 张 | `D:\Datasets\external_cityscapes_val_20261026` | 欧洲城市域 |
| 官方比赛视频 | 视频验证/演示 | 4 段 1080p | `D:\BaiduNetdiskDownload\紫光同创杯赛题三参考素材` | 113421/111440/112739/112454 等 |
| MOT17 | 跟踪/多流研究 | 独立 | `D:\Datasets\MOT17` | 不用于比赛检测精度结论 |

> 完整 BDD100K 70k train、原始标注、训练缓存不上传 GitHub；只保留 split 名称、统计、评估 JSON 和权重 hash。

---

## 2. BDD 10 类映射

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

映射规则：

- `pedestrian → person`
- `motorcycle → motor`
- `bicycle → bike`
- `other vehicle / trailer / other person` 忽略
- `traffic light` 保留；颜色不进入检测类别，颜色由 HSV/多帧规则单独判断

---

## 3. 划分与防泄漏

### 3.1 B05N/B05L 主 split

| 划分 | 图像 | 说明 |
|---|---:|---|
| train | 8,513 | 复用 B05L 训练名单 |
| val | 482 | clean val，排除近重复 |
| test | 1,000 | B03D holdout-1000 |
| BDD50 | 50 | 人工精修 TL，不进入 train/val |

B05N 统计：

| 划分 | 图像 | 框数 | TL 框 | 8–16px TL |
|---|---:|---:|---:|---:|
| train | 8,513 | 103,133 | 22,935 | 11,374 |
| val | 482 | 9,041 | 1,208 | 620 |
| test | 1,000 | 18,450 | 2,734 | 1,345 |

### 3.2 泄漏检查

- BDD50 的 50 个文件名不在 train，也不在 val；50 个都在 `bdd_holdout_1000_names.txt` 中。
- train/val 名称无重叠。
- 用 dHash 检查到 5 张 val 与 train 近重复（距离 ≤3），已从验证集排除，不移入训练集；最终 clean val 482 张。
- 训练集内仍有 2 组完全相同的图像哈希，属于同场景重复帧；记录为数据噪声，未为此改动训练集。
- BDD50 的 ROI 内 150 GT、ROI 外 4 GT；Specialist 只能覆盖 ROI 内，不把 ROI 外 GT 混入 ROI Recall。

`bdd_holdout_1000_names.txt` SHA256：

```text
6ab68ae10000afb9c7d25c756ba2c99ce7ca826a18c6bb9a3598dc3e368a2623
```

### 3.3 Cityscapes 500

- 来源 revision：`b7e3f28b5aba95b55de608c7e94c952ea3966d64`
- 清单 SHA256：`19a30f0bcf80bbd65230b65467c07452b0bb12b4169ea407e4e42d50b4491514`
- 只用于独立方向性验收；不能替代比赛域/BDD 测试。
- 同类 group 区域在未匹配预测中按 IoU 规则忽略（537 个区域）。

---

## 4. 评估口径

### 4.1 全画面主模型

| 项目 | 值 |
|---|---|
| 输入 | 整帧直接 resize 到 512×288（RGB） |
| 置信度下限 | AP 排名 `0.001`；固定工作点 `0.25` |
| 匹配 IoU | `0.50` |
| NMS IoU | `0.70`（评测）；正式 bridge 使用 `0.45` |
| 类别 | 10 类 |
| 硬件 | PC FP32 / RKNN simulator / RK3566 分开标注 |

### 4.2 Specialist

| 项目 | 值 |
|---|---|
| 输入 | 原图顶部 60% ROI → 直接 resize 到 512×192 |
| 类别 | 单类 `traffic light` |
| AP 下限 | `0.001` |
| 工作点 | `conf=0.25`，NMS IoU `0.50` |
| 融合 | C/D 用 Specialist 替换主模型 TL；B05O 的 union+NMS 仅用于调度研究 |
| 颜色 | HSV 后处理，不训练颜色头 |

### 4.3 指标含义

| 指标 | 含义 | 不能说明 |
|---|---|---|
| mAP50 | IoU=0.5 的平均精度 | 不是比赛域准确率；不是 INT8 实板精度 |
| mAP50-95 | IoU 0.5:0.95 平均 | 对小目标更敏感，但仍受数据集影响 |
| Precision/Recall | 固定工作点 | 不是全阈值曲线 |
| 8–16px Recall | 缩放到模型输入后的框宽分组 | 不是原始像素尺寸；不是夜间/箭头灯结论 |
| AI 结果率 | bridge 每秒发布结果数 | 不是 NPU 原始 FPS；不是显示 FPS |
| NPU 调用 p50 | 单次 `rknn.inference()` 时间 | 不含预处理/解码/NMS |
| 主模型总耗时 | 预处理+RKNN+DFL/NMS | 不含队列等待/显示 |

---

## 5. 各评测集能回答什么

| 评测集 | 能回答 | 不能回答 |
|---|---|---|
| BDD holdout-1000 | 同数据、同类别下模型相对强弱 | 比赛道路/夜间/中国路况 |
| BDD50 | 交通灯小目标/ROI 专模相对强弱 | 全画面交通灯召回；颜色准确率 |
| Cityscapes 500 | 跨域方向性泛化、FP 代价 | 比赛域最终精度 |
| 官方视频 | 候选/漏检/误检样本、时序行为 | 无逐帧 GT，不能算 Recall/AP |
| RK3566 实板 | 真实延迟/吞吐/资源 | 不是 PC FP32 精度 |
| 四路文件回放 | 调度/队列行为 | 不是四路 PCIe 实时输入 |

---

## 6. 复现命令（示例）

### 6.1 PC 评估 B05N P2

```powershell
cd C:\Users\JingyuWang\adas-edge-b05n
.venv\Scripts\python.exe tools\b05n_eval_multiclass.py `
  --model D:\Datasets\adas_edge_training\b05n_final\b05n_p2_multiclass.pt `
  --data D:\Datasets\adas_edge_training\b05n_bdd_multiclass\data.yaml `
  --split test --imgsz 512 --conf 0.25 --iou 0.5
```

> 具体参数以 `tools/b05n_eval_multiclass.py --help` 为准；不同轮次的 JSON 可能使用不同 `--ap-floor`，报告中已标注。

### 6.2 BDD50 交通灯

```powershell
cd C:\Users\JingyuWang\adas-edge-b05n
.venv\Scripts\python.exe tools\b05n_tl_eval.py `
  --model D:\Datasets\adas_edge_training\b05n_final\b05n_p2_multiclass.pt `
  --images D:\Datasets\adas_edge_training\b05l_data\bdd50\images `
  --labels D:\Datasets\adas_edge_training\b05l_data\bdd50\labels
```

### 6.3 RKNN 精度

RKNN Toolkit2 2.3.2 的 PC simulator 评估脚本：

```text
deployment/rknn_work/rknn_eval.py
deployment/rknn_work/rknn_raw_heads.py
deployment/rknn_work/rknn_accuracy_analysis.py
```

### 6.4 板端 bridge 回放

```text
deployment/rknn_work/b05p_competition_bridge.py
deployment/rknn_work/run_qat_smoke_control.py
```

原始 JSON 已归档在 `docs/models/evidence/B05P/`。

---

## 7. 数据限制与风险

1. **不是完整 BDD 70k**：B05N/B05L 主要用 8,513 张 train，结论不能外推到完整 BDD 训练。
2. **BDD50 已用于多轮调参**：它适合筛选，不是最终独立测试集。
3. **Cityscapes 是欧洲城市域**：与比赛视频/中国道路有域差异。
4. **官方视频没有逐帧 GT**：只能做候选/漏检/误检样本审查，不能算 AP/Recall。
5. **交通灯颜色没有完整 GT**：HSV 结果不能宣称数据集级颜色准确率。
6. **夜间/雨天/隧道样本不足**：不能宣称复杂场景已通过。
7. **四路数据是文件回放**：不是 PCIe 四路实时输入。
8. **板端时钟未同步**：时间戳只能用于相对顺序，不是真实日历时间。

---

## 8. 证据

- `docs/models/evidence/B05N/summary.json`
- `docs/models/evidence/B05L/BDD50_summary_resize.json`
- `docs/models/evidence/B05P/B05P_bridge_runtime_20261027_*.json`
- `docs/models/MODEL_MANIFEST.md`
- 原始报告：B05N/B05L/B05P Markdown（Windows 训练 worktree）
