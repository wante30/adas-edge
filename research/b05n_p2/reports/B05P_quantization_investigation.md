# B-05P — RK3566 模型量化故障定位与部署验证

日期：2026-10-25<br>
分支：`research/rk3566-b05o-video-scheduling`<br>
范围：不训练新模型；只做 RKNN 转换、量化诊断、精度对照和板端验证准备。

## 0. 结论

1. **P2 INT8 的损失不是单一层或单一输出造成的，而是全网络累积的量化误差。** 逐层 single-layer cosine 多在 0.999 以上，但 entire cosine 累积降到 0.945–0.966；最差集中在检测头 P2 分支（`/model.28/cv2.3.1`）和 neck（`/model.8`、`/model.24`）。
2. **输出层确实有离群值问题，但不是主因。** box 量化步长 3.73 px、logit 步长 0.339；拆输出 + 裁剪只把 mAP 从 0.2747 提到 0.2976–0.3031，仍比 FP16 低 6 个点以上。
3. **FP16 与 ONNX 等价**（mAP 0.3644 vs 0.3646，TL 0.4778 vs 0.4766），是当前唯一可直接上板验证的格式。
4. **当前 INT8 不能作为正式部署权重。** 需要 hybrid（敏感层 FP16）或更大的结构调整；本轮没有拿到有效的 hybrid 结果。
5. **RK3566 板端未验证。** 主机没有 adb/hdc 设备，P50/P95、NPU 占用、10Hz/15Hz 可行性均缺失；不能把 PC 数字当板端结果。
6. **正式 BDD 模型同样受 INT8 影响**：mAP 0.3688 → 0.2995，TL 0.3880 → 0.2116；FP16 无损。
7. **交接文档**：`research/rk3566_models/reports/B05P_handoff.md`，包含板端步骤、复现命令、已知坑和文件索引。

## 0.1 两个问题的直接回答

**第一：P2 INT8 为什么损失严重，能否修复？**

损失来自 200+ 层累积量化误差，叠加输出离群值；交通灯这类小目标最敏感。可以通过裁剪输出、扩大校准集小幅恢复（约 +3 mAP），但要接近 FP16 需要 hybrid 保留敏感层，当前尚未完成。

**第二：RK3566 能否稳定运行正式双模型并达到实时性？**

未知。板子当前不可访问，没有任何板端延迟或占用数据；本轮不能给出“能/不能”的结论。

## 1. 模型与口径

| 角色 | 权重 | 输入 | 输出 |
|---|---|---|---|
| BDD 主模型 | `bdd100k-yolov8n-best.pt` | 512×288 | `[1,14,3024]`（3 尺度） |
| B05L P2 Specialist | `b05l_runs/v8n_p2/weights/best.pt` | 512×192 | `[1,5,8160]`（4 尺度） |
| B05N P2 10 类 | `b05n_final/b05n_p2_multiclass.pt` | 512×288 | `[1,14,12240]`（4 尺度） |

统一：RGB、`1/255`、`conf=0.25`（AP 用 0.001）、NMS IoU 0.5（AP 0.7）。

评估集：1000 张 BDD holdout 中 **第 200–999 张**（校准不使用这 800 张）。

## 2. P0 板端验证

**状态：阻塞。** 当前 Windows 主机没有可用的 RK3566 连接：

```text
adb devices        -> 无设备
hdc list targets   -> 无设备
```

因此本轮没有板端延迟、CPU/NPU 占用或实际推理频率数据。所有性能数字均标明为 PC 模拟，不能当作 RK3566 结果。

板端可用后需要执行：

- BDD 单模型 NPU-only P50/P95；
- Specialist 单模型 P50/P95；
- 双模型顺序执行 P50/P95；
- 10 Hz + 5 Hz 调度下的端到端延迟、掉帧、内存；
- INT8 / FP16 / Hybrid 三种格式对比。

## 3. P1：P2 INT8 误差定位

### 3.1 量化配置

从 `b05n_p2_split_logits.quantization.cfg` 直接读取（不按文件名推断）：

| 张量 | dtype | min | max | scale | zero_point |
|---|---|---:|---:|---:|---:|
| `boxes_int8` | int8 | −98.7 | 852.6 | **3.73 px** | −102 |
| `scores_int8` | int8 | −83.14 | 3.41 | **0.339** | 117 |

- 坐标量化步长 3.73 px（网络坐标，512 宽）；8–16 px 的灯在原图缩放 0.4 倍后只剩 3–6 px，误差超过目标本身。
- logits 量化步长 0.339；在 conf=0.25（logit≈−1.1）附近，±0.34 会把概率推到 0.16–0.36。
- 两个范围的极值都来自少量离群 anchor：box 最大 852（超出画面），logit 最小 −83。

### 3.2 输出级与检测级误差

同一批 50 张图（test 第 200–249 张），`conf=0.25`、`NMS IoU=0.7`，RKNN INT8 与 ONNX 逐类匹配（IoU≥0.5）：

| 类别 | ONNX 框 | RKNN 框 | 匹配 | RKNN 多出 | ONNX 未匹配 | 匹配 IoU | 置信度差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| car | 427 | 613 | 389 | **224** | 38 | 0.772 | 0.028 |
| person | 29 | 41 | 22 | **19** | 7 | 0.816 | 0.031 |
| traffic light | 121 | 168 | 84 | **77** | 29 | 0.665 | 0.050 |
| traffic sign | 225 | 323 | 128 | **135** | 30 | 0.696 | 0.042 |

结论：RKNN INT8 不是简单“漏检”，而是**同时产生大量低质量重复框和位置偏移**；交通灯的匹配 IoU 只有 0.665，明显低于 car（0.772）。

原始输出平均绝对差 `0.67`（主要来自坐标通道）；FP16 的同类差值为 `7e-4` 量级。

### 3.3 逐层分析

`rknn.accuracy_analysis`（2 张图，223 层）结果：

- 绝大多数层的 **single-layer cosine ≥ 0.999**，量化本身没有“某一层彻底坏掉”；
- **entire cosine 随层数累积下降**，最差层集中在检测头 P2 分支和 neck：
  - `/model.28/cv2.3/cv2.3.1/act` entire `0.966`，single `0.993`
  - `/model.28/cv2.3/cv2.3.1/conv` entire `0.973`，single `0.995`
  - `/model.8/Split` entire `0.945–0.982`（两张图）
  - `/model.24/Concat` entire `0.968–0.987`
- DFL 分支：`Softmax` entire `0.987–0.992`，single `0.999`；
- 最终 `boxes`/`scores` 输出没有出现单层崩塌。

结论：**误差是 200 多层累积的，不是单一输出张量造成的**。这解释了为什么“只把 boxes/scores 输出改成 FP16”没有效果（mAP 仍为 0.2744）。

### 3.3.1 输出范围裁剪实验

把 `boxes` 裁剪到 `[-32, 544]`、`scores` logits 裁剪到 `[-12, 6]`，再做 INT8（train 200 校准）：

| 配置 | mAP50 | TL AP50 |
|---|---:|---:|
| ONNX | 0.3646 | 0.4766 |
| INT8 原始 | 0.2747 | 0.2010 |
| INT8 + 输出裁剪 | **0.2976** | **0.2640** |

输出范围确实有影响（+2.3 mAP / +6.3 TL），但只恢复了约 1/4 的损失，**主要损失仍在网络内部**。

### 3.4 结论

1. FP16 与 ONNX 等价（mAP 差 0.0002，TL 差 0.0012）；
2. INT8 的损失不是单一层、也不是单一输出张量，而是**全网络累积量化误差**；
3. 输出离群值（box max 852、logit min −83）让输出 scale 过大（3.73 px / 0.339），裁剪后能恢复一部分，但不是主因；
4. 交通灯受害最重：框小、置信度低、匹配 IoU 只有 0.665；
5. 目前没有证据支持“只修输出就能救回 INT8”。

## 4. P2：量化恢复实验

### 4.1 MMSE

**未完成。** 第一次 MMSE 运行到 70/142 层、耗时 39 分钟后中止（ETA 还需约 71 分钟），没有生成结果 JSON；当前没有可复用结果。

由于 MMSE 单次约 1.5–2 小时，本轮先完成其他实验；MMSE 结果待补。

### 4.2 校准集规模

校准图全部来自 **train split**，评估固定为 test 第 200–999 张（800 张），无重叠。

| 校准集 | 输出裁剪 | mAP50 | TL AP50 | 说明 |
|---|---|---:|---:|---|
| test 200 | 否 | 0.2747 | 0.2010 | 初始基线 |
| train 200 | 是 | 0.2976 | 0.2640 | 与上一行同时改了校准集和输出裁剪，**不能单变量归因** |
| train 500 | 是 | **0.3031** | 0.2596 | 校准集扩大后小幅提升 |
| train 1000 | 是 | — | — | 运行超过 25 分钟未完成，已中止；500 已显示收益饱和 |

观察：

- 校准集从 200 扩到 500，mAP +0.55 点，TL −0.44 点，属于噪声量级；
- 扩大校准集没有改变“INT8 明显低于 FP16”的结论；
- 尚未做“train 200 + 不裁剪”的单变量对照，因此不能把 0.2747→0.2976 全部归因于裁剪。

### 4.3 Hybrid INT8/FP16

已尝试：

| 方案 | 保留 FP16 的范围 | mAP50 | TL AP50 |
|---|---|---:|---:|
| 纯 INT8 | 无 | 0.2747 | 0.2010 |
| 输出 FP16 | `boxes_int8` / `scores_int8` | 0.2744 | 0.2002 |
| 检测头 FP16 | `/model.28/` 51 个有效层 | 0.2729 | 0.1980 |

结论：

- **输出和检测头改 FP16 都没有恢复精度**，说明主要误差在 backbone/neck，而不是检测头；
- 继续做 hybrid 只能把大量 backbone 层也保留 FP16，速度收益会迅速消失；
- 因此“hybrid 救 INT8”在当前结构下不成立，除非接受接近 FP16 的算力开销。

### 4.4 正式模型：BDD 与 Specialist

BDD 主模型（同样的 800 张 test 200–999）：

| 格式 | mAP50 | TL AP50 |
|---|---:|---:|
| ONNX | 0.3688 | 0.3880 |
| RKNN FP16 | **0.3686** | **0.3874** |
| RKNN INT8（裁剪 + 500 校准） | 0.2995 | 0.2116 |

Specialist：本轮 `tlmap` 评估脚本有 bug（ONNX 参考同样 AP 0、TP 0、FP 114），**结果无效，不引用**。以 B05L 既有评估为准：AP50 0.6107、8–16px Recall 0.5750。修正方向见交接文档。

### 4.5 完整 FP16

| 格式 | mAP50 | TL AP50 | 大小 |
|---|---:|---:|---:|
| ONNX | 0.3646 | 0.4766 | 12.0 MB |
| RKNN FP16 | **0.3644** | **0.4778** | 6.3 MB |
| RKNN INT8 | 0.2747 | 0.2010 | 4.3 MB |

FP16 精度与 ONNX 等价，是当前唯一可以直接上板验证的候选格式；速度待板端实测。

## 5. P3：10 Hz 官方视频对比

复用 B05O 已完成的调度视频；本轮不重新推理，不修改评估参数：

```text
D:\Datasets\adas_edge_training\b05o_video_scheduling\
  video_20260929_112739\scheduled\compare_tl_3schemes.mp4
  video_20260929_112739\upper_bound\compare_tl_3schemes.mp4
```

`112739` 81.23 s 两个红灯：三方案与 30 Hz 上限均为 0 框；BDD 在 81.3 s 检出右侧红灯，P2 在 78.7–83.5 s 完全没有 TL 框。详见 B05O 报告。

## 6. 推荐方案

### 6.1 当前可交付

| 用途 | 格式 | 文件 |
|---|---|---|
| 板端精度基准 | FP16 | `b05n_p2_multiclass_512x288_split_fp16.rknn` |
| 板端速度基准 | INT8（裁剪输出） | `b05n_p2_clip_int8_500.rknn` |
| 正式双模型主模型 | FP16/INT8 待板端 | `bdd_yolov8n_512x288_split_fp16.rknn` / `..._clip_int8` |
| 正式 Specialist | FP16/INT8 待板端 | `b05l_p2_specialist_512x192_split_fp16.rknn` / `..._clip_int8` |

### 6.2 选择依据

- **精度优先：FP16。** 与 ONNX 等价，不需要重新训练，风险最低；RK3566 NPU 支持 FP16，但速度待测。
- **速度优先：INT8 暂不可用。** 当前最好 0.3031 mAP，交通灯 0.2596，比 FP16 低 6–22 个点；除非板端实测证明 FP16 达不到 10Hz，否则不应为了体积/速度牺牲这些精度。
- **双模型 vs 单模型：** 仍应按 B05O 的结论，在板端实测 15 次/秒能否稳定；如果 FP16 双模型达不到实时性，再比较“FP16 单 P2”与“INT8 双模型”的精度/速度折中。
- **不建议**直接把当前 INT8 权重上板做效果对比，否则结论会被量化损失污染。

## 7. 交付物

### 7.1 RKNN 模型

```text
D:\Datasetsdas_edge_training
knn_b05o  b05n_p2_multiclass_512x288_split_fp16.rknn
  b05n_p2_multiclass_512x288_split_int8.rknn
  b05n_p2_clip_int8.rknn
  b05n_p2_clip_int8_500.rknn
  bdd_yolov8n_512x288_split_fp16.rknn
  bdd_yolov8n_512x288_split_int8.rknn
  b05l_p2_specialist_512x192_split_fp16.rknn
  b05l_p2_specialist_512x192_split_int8.rknn
  b05n_p2_multiclass_512x288_hybrid_out.rknn
  b05n_p2_multiclass_512x288_hybrid_head.rknn   （未成功生成）
```

所有文件均为新增，未覆盖 `b05n_final` 或正式部署权重。

### 7.2 配置与日志

```text
D:\Datasets\adas_edge_training\rknn_b05o\
  calib_train_full_{200,500,1000}.txt
  calib_train_spec_{200,500,1000}.txt
  hybrid_p2\b05n_p2_split_logits.quantization.cfg
  hybrid_p2\output_fp16.cfg
  accuracy_analysis_p2\image_{0,1}\error_analysis.txt
```

### 7.3 脚本

```text
deployment/rknn_work/
  rknn_build.py                  ONNX -> RKNN（FP16/INT8）
  rknn_eval.py                   ONNX vs RKNN 精度/一致性
  rknn_make_calib_train.py       训练集校准图
  rknn_accuracy_analysis.py      逐层误差
  rknn_hybrid_step1.py           hybrid 配置生成
  rknn_hybrid_cfg.py             hybrid 层选择
  rknn_hybrid_eval.py            hybrid 构建 + mAP
  clip_onnx_outputs.py           输出范围裁剪
  split_yolo_multiscale.py       多尺度输出拆分（备用）
  run_bdd_spec_rknn.sh           正式模型批量评估
```

## 8. 限制

- RK3566 不在线，所有板端指标缺失。
- PC 推理耗时只能说明转换/评估流程可用。
- INT8 结论基于 RKNN Toolkit2 2.3.2 模拟器，板端可能略有差异。
