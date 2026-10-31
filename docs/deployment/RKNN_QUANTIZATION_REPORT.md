# RKNN 量化与精度报告

更新日期：2026-10-28  
硬件：LubanCat-1 RK3566（4×A55，Mali-G52，3.8 GiB）  
运行时：RKNN Toolkit2 / Lite2 2.3.2，RKNPU driver 0.8.2  
范围：BDD 主模型、B05N P2、B05L Specialist 的 ONNX → FP16/INT8 RKNN 转换、精度对照和板端状态。

---

## 0. 结论

1. **FP16 与 ONNX 等价**：BDD 800 张 mAP50 `0.3686` vs ONNX `0.3688`；P2 `0.3644` vs `0.3646`。
2. **传统 post-DFL split INT8 损失严重**：BDD `0.2995`、P2 `0.3031`、Specialist `0.2421`，交通灯尤其明显。
3. **Raw-head 导出显著修复 INT8**：P2 `0.3651 / TL 0.4733`，Specialist `0.6119`，已接近 ONNX/FP16；BDD 仍为 `0.3606 / 0.3752`，比 ONNX 低 0.8/1.3 点。
4. **当前默认是 BDD Raw-head INT8（A）**，已在 RK3566 实际加载并运行 10 Hz；B/C/D 也已板端验证，但默认不启用。
5. **INT8 精度不能只用“能加载”证明**：必须区分 PC simulator、板端单图 smoke、板端连续视频；本报告分别标注。
6. **QAT 只做了 smoke/探索，没有可引用的最终精度**；不要把 `bdd_qat_full_10ep_20261026` 当作已验证的正式模型。

---

## 1. 工具链与转换链路

```text
PyTorch .pt
  → Ultralytics export ONNX（静态 1×3×H×W）
  → （可选）split/raw-head 输出改造
  → RKNN Toolkit2 2.3.2（WSL/Linux）
  → FP16 或 INT8 .rknn
  → RKNN Lite2 2.3.2 on RK3566
```

| 项目 | 值 |
|---|---|
| Toolkit | `rknn-toolkit2 2.3.2` |
| Runtime | `rknn-toolkit-lite2 2.3.2` |
| Driver | `0.8.2` |
| 目标平台 | `rk3566` |
| 输入 | RGB uint8 NHWC，`mean=0`，`std=255` |
| BDD 输入 | `[1,288,512,3]` |
| P2 输入 | `[1,288,512,3]` |
| Specialist 输入 | `[1,192,512,3]` |
| 量化 | INT8 `do_quantization=True`；校准图来自 train split |

### 1.1 两种输出结构

| 结构 | 输出 | 特点 |
|---|---|---|
| post-DFL split | boxes `[1,4,N]` + scores `[1,C,N]` | 旧方案；DFL/坐标已在模型内完成，INT8 对输出范围敏感 |
| raw-head | 每尺度 3 个张量：回归 logits、分类分数、分数和；4 尺度共 12 路 | DFL/坐标解码移到 host；INT8 精度显著更好 |

Raw-head 的代价：host 端要做 DFL softmax、坐标解码和 NMS；板端 Python 后处理可能成为瓶颈。当前正式 bridge 已实现 Raw-head adapter。

---

## 2. 精度对照

### 2.1 BDD 主模型（800 张 BDD test 200–999）

| 格式 | mAP50 | TL AP50 | 相对 ONNX |
|---|---:|---:|---:|
| ONNX FP32 | 0.3688 | 0.3880 | 基准 |
| RKNN FP16 | 0.3686 | 0.3874 | −0.0002 / −0.0006 |
| RKNN INT8 Raw-head | 0.3606 | 0.3752 | −0.0082 / −0.0129 |
| RKNN INT8 post-DFL split（KL 500） | 0.3184 | 0.2406 | −0.0504 / −0.1474 |
| RKNN INT8 post-DFL split（normal 500） | 0.2995 | 0.2116 | −0.0693 / −0.1764 |

### 2.2 B05N P2（800 张 BDD test 200–999）

| 格式 | mAP50 | TL AP50 |
|---|---:|---:|
| ONNX FP32 Raw-head | 0.3646 | 0.4766 |
| RKNN FP16 | 0.3644 | 0.4778 |
| RKNN INT8 Raw-head | **0.3651** | **0.4733** |
| RKNN INT8 post-DFL split | 0.3031 | 0.2596 |

### 2.3 B05L Specialist（BDD50）

| 格式 | AP50 | Recall | TP/FP/FN |
|---|---:|---:|---:|
| ONNX FP32 Raw-head | 0.6099 | 0.580 | 87/26/63 |
| RKNN FP16 Raw-head | 0.6109 | 0.580 | 87/28/63 |
| RKNN INT8 Raw-head | **0.6119** | **0.580** | 87/24/63 |
| RKNN INT8 post-DFL split | 0.2421 | 0.340 | 51/92/99 |

> BDD50 已用于多轮调参，以上是筛选结果，不是独立最终测试集。

---

## 3. 为什么旧 INT8 损失大

### 3.1 输出离群值

P2 post-DFL split 的量化配置：

| 张量 | dtype | min | max | scale | zero_point |
|---|---|---:|---:|---:|---:|
| `boxes_int8` | int8 | −98.7 | 852.6 | **3.73 px** | −102 |
| `scores_int8` | int8 | −83.14 | 3.41 | **0.339** | 117 |

- 坐标步长 3.73 px；8–16 px 的交通灯在 512 宽输入下只剩 3–6 px，误差超过目标本身。
- logits 步长 0.339；在 conf=0.25 附近会把概率推到 0.16–0.36。
- 输出裁剪到 `[-32,544]` / `[-12,6]` 只能把 mAP 从 0.2747 提到 0.2976，恢复约 1/4。

### 3.2 逐层累积

`rknn.accuracy_analysis`（2 张图，223 层）：

- 大多数层 single-layer cosine ≥ 0.999；
- entire cosine 随层数下降，最差集中在 neck/检测头：
  - `/model.28/cv2.3/cv2.3.1/act` entire 0.966
  - `/model.8/Split` entire 0.945–0.982
  - `/model.24/Concat` entire 0.968–0.987
- 只把输出/检测头改 FP16 没有效果（mAP 0.2744 / 0.2729）。

结论：**不是单层/单输出问题，而是全网络累积量化误差 + 输出离群值。**

### 3.3 Raw-head 为什么有效

- 把 DFL/坐标解码移出模型，减少输出端的极端范围；
- 每尺度独立输出，避免 concat 后坐标主导量化范围；
- 分类分数保留在更合适的域；
- 代价是 host 后处理更重。

> 这是“输出结构 + 分数域 + 分尺度量化”的组合收益，不能单独归因于 DFL 外移。

---

## 4. 板端验证状态

### 4.1 当前实际运行

| 项目 | 值 |
|---|---|
| 方案 | A（BDD Raw-head INT8） |
| 路径 | `/home/cat/cityscapes_arch_compare_20261027/models/bdd_yolov8n_512x288_raw_heads_int8.rknn` |
| SHA256 | `a86ab053...041afb` |
| 输入 | `[1,288,512,3]` uint8 NHWC |
| 配置 | `--main-hz 10 --hazard-hz 5` |
| 实测 | 主 RKNN p50 29.15 ms；主模型总耗时 p50 38.77 ms；AI 结果率 9.98 Hz |
| 资源 | NPU 100% @ 600 MHz；RSS 144 MB |

### 4.2 已板端验证的候选

| 方案 | 模型 | 板端状态 | 证据 |
|---|---|---|---|
| A | BDD Raw-head INT8 | 当前运行 | `current_live_summary_20261028.json` |
| B | P2 Raw-head INT8 | bridge 回放/切换验证 | `B05P_bridge_runtime_20261027_B.json` |
| C | BDD + B05L Specialist | bridge 回放验证 | `B05P_bridge_runtime_20261027_C.json` |
| D | P2 + B05L Specialist | bridge 回放验证 | `B05P_bridge_runtime_20261027_D.json` |

### 4.3 未验证/不能声称

- 未在板端做完整 800 张/1,000 张精度复测；板端精度来自 PC simulator。
- 未做长时间（>10 分钟）四路 PCIe + AI + HMI 稳定性测试。
- 未做 INT8 vs FP16 的板端同帧精度对照。
- 未做 QAT 最终模型验证。
- 未证明四路独立 PCIe 流的吞吐。

---

## 5. 转换/兼容性问题记录

| 问题 | 现象 | 处理 |
|---|---|---|
| `load_rknn()` 不能 simulator 推理 | Toolkit2 模拟器要求 `load_onnx + build + init_runtime` | 用 ONNX 重建后评估 |
| `accuracy_analysis` 输入格式 | 需要 NCHW float，不是 NHWC uint8 | 转换脚本显式 NCHW |
| hybrid step2 | 不允许修改 scale/zero_point；内部 operand 名不能加入 `custom_quantize_layers` | 放弃该路径 |
| concat 输出 `[1,14,N]` | INT8 分数被坐标范围主导 | 改 split boxes/scores |
| logits vs prob | logits 低置信度保真更好，但有 −83 极值 | raw-head + host sigmoid |
| Specialist 旧评估脚本 | ONNX 参考也 AP=0 | 用 B05L `b05l_eval.py` 的 Resize 路径，以 0.6107 为准 |
| MMSE | 跑到 70/142 层、39 分钟后中止 | 无结果，未采用 |
| QAT | 有 smoke/10ep 运行目录 | 无最终精度报告，不采用 |

---

## 6. 复现

### 6.1 转换（WSL）

```bash
wsl -d Ubuntu
source /home/jingyuwang/.venvs/rknn232/bin/activate
cd /mnt/c/Users/JingyuWang/adas-edge-b05o/deployment/rknn_work

python rknn_build.py \
  --onnx /mnt/d/Datasets/adas_edge_training/rknn_b05o/bdd_yolov8n_512x288_raw_heads.onnx \
  --output /mnt/d/Datasets/adas_edge_training/rknn_b05o/bdd_yolov8n_512x288_raw_heads_int8.rknn \
  --calib-dir /mnt/d/Datasets/adas_edge_training/rknn_b05o/calib_train_full_500 \
  --precision int8
```

### 6.2 精度评估

```bash
python rknn_eval.py --kind multiclass --precision int8 \
  --onnx .../bdd_yolov8n_512x288_raw_heads.onnx \
  --rknn .../bdd_yolov8n_512x288_raw_heads_int8.rknn \
  --images .../images/test --labels .../labels/test \
  --offset 200 --limit 800 --conf 0.001 --iou 0.7 --mode map
```

### 6.3 板端加载

```sh
python3 -c "from rknnlite.api import RKNNLite; r=RKNNLite(); print(r.load_rknn('...rknn')); print(r.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO))"
```

> 板端加载成功不等于精度通过；必须跑同一帧/同一后处理的输出对照。

---

## 7. 证据

- `docs/models/evidence/B05P/B05P_bridge_runtime_20261027_A.json`
- `docs/models/evidence/B05P/B05P_bridge_runtime_20261027_B.json`
- `docs/models/evidence/B05P/B05P_bridge_runtime_20261027_C.json`
- `docs/models/evidence/B05P/B05P_bridge_runtime_20261027_D.json`
- `docs/deployment/evidence/current_live_summary_20261028.json`
- `artifacts/manifests/model_manifest.json`
- 原始报告：`B05P_quantization_investigation.md`、`B05P_INT8_accuracy_followup.md`、`B05P_multi_model_raw_heads_20261026.md`、`B05P_architecture_selection_20261027.md`

---

## 8. 2026-10-31 更新：B-2a 多任务 / Seg-256

### 8.1 B-2a 输出与量化

B-2a 在 B05N P2 的 12 路 Raw-head 之外增加 3 路分割：

| 输出 | shape | 类别 |
|---|---|---|
| `lane_cat` | 1×9×288×512 | crosswalk / double other / double white / double yellow / road curb / single other / single white / single yellow / background |
| `lane_style` | 1×3×288×512 | solid / dashed / background |
| `drivable` | 1×3×288×512 | direct / alternative / background |

全部为 logits + per-tensor affine INT8。与 PyTorch FP32 的一致性（Toolkit2 模拟器，10 张）：

| 指标 | FP16 | INT8 |
|---|---:|---:|
| 车道前景 IoU | 0.9992 | 0.9302 |
| 斑马线 IoU | 0.9990 | 0.8875 |
| 线型准确率 | 0.9994 | 0.9163 |
| 实线准确率 | 0.9990 | 0.9659 |
| 虚线准确率 | 0.9996 | 0.8929 |
| 可行驶 mIoU | 0.9994 | 0.9264 |
| traffic light 框匹配率 | 1.000 | 0.967 |
| traffic light 分数变化 | ~0 | −0.0088 |

检测端可用；分割端可用，但虚线（lane_style dashed）是弱项。B-2a INT8 的分割一致性高于 B-1 INT8，说明联合微调特征对量化更稳。

### 8.2 Seg-256

在 ONNX 图内对三个分割输出加 2×2 AveragePool（stride 2），输出 144×256；权重不重新初始化，校准集不变。

- ONNX 校验：新分割输出与原 logits 的 2×2 均值池化最大差 3.8e-6；12 个检测输出逐元素一致。
- RKNN 文件 4,933,694 B；分割张量 4× 减少。
- 60 帧一致性（恢复到 512×288）：车道 IoU 0.794、斑马线 0.768、线型一致率 0.963、可行驶 mIoU 0.872。
- 速度：RKNN 调用 158.8 → 117.8 ms，完整单帧 244.6 → 161.1 ms，Python 连续 3.55 → 4.79 FPS；C API 5.97 FPS。

### 8.3 原生 INT8 argmax

三个分割头均为 per-tensor affine；同一像素 9/3/3 通道共享 scale/zp，因此直接对 int8 取 argmax 与反量化后取 argmax 等价。C API `want_float=0` 在 60 帧、8,847,360 像素上与 `want_float=1` 一致率 **1.000000**。

### 8.4 证据

- `research/b05n_p2_mtl/deployment/stageB2a/conversion_report.md`
- `research/b05n_p2_mtl/deployment/stageB2a/seg256/seg256_report.md`
- `research/b05n_p2_mtl/deployment/stageB2a/capi/capi_report.md`
- `research/b05n_p2_mtl/deployment/stageB2a/capi/capi_bc_report.md`
- `docs/models/evidence/MTL/`
- `docs/research/MTL_RESEARCH_SUMMARY.md`
