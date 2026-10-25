# B-05P 交接文档 — RKNN 量化与板端验证

最后更新：2026-10-25<br>
分支：`research/rk3566-b05o-video-scheduling`<br>
接手目标：完成 RK3566 板端 FP16 / INT8 对比，回答“正式双模型能否满足实时性”。

## 1. 一句话现状

**FP16 已可用且精度无损；INT8 精度损失严重，暂不可部署；板端完全没测，因为当前主机没有 RK3566 连接。**

## 2. 已经确定的事实

### 2.1 精度（800 张 BDD holdout，test 第 200–999 张）

| 模型 | 格式 | mAP50 | TL AP50 |
|---|---|---:|---:|
| P2 10 类 | ONNX | 0.3646 | 0.4766 |
| P2 10 类 | **RKNN FP16** | **0.3644** | **0.4778** |
| P2 10 类 | RKNN INT8（原始） | 0.2747 | 0.2010 |
| P2 10 类 | RKNN INT8（输出裁剪 + 500 校准） | 0.3031 | 0.2596 |
| BDD 主模型 | ONNX | 0.3688 | 0.3880 |
| BDD 主模型 | **RKNN FP16** | **0.3686** | **0.3874** |
| BDD 主模型 | RKNN INT8（裁剪 + 500 校准） | 0.2995 | 0.2116 |

结论：

- FP16 与 ONNX 等价，误差 ≤0.0006；
- INT8 总体掉 6–7 个点，交通灯掉 13–18 个点；
- 输出裁剪 + 扩大校准只能恢复约 3 点；
- 把输出层或整个检测头改 FP16 都没有提升（0.2744 / 0.2729），说明误差在 backbone/neck；
- 继续 hybrid 需要把大量 backbone 层保 FP16，速度优势会消失。

### 2.2 原版 BDD RKNN 的真相

B03C 当时：

- 用 `bdd_yolov8n_512x288_split.onnx`（**Sigmoid 后的概率**）+ **6 张**校准图；
- 板端只测了 50 张图的框一致性：**48/83 达到 IoU 0.5**；
- **没有测 mAP**。

所以“原来的 INT8 没问题”是视觉印象，不是精度结论。

### 2.3 为什么 INT8 损失大

- 逐层 single-layer cosine 多在 0.999 以上，但 entire cosine 累积降到 0.945–0.966；
- 最差层：`/model.8`、`/model.24`、`/model.28/cv2.3.1`；
- 输出离群值放大问题：box 范围 −99–853 → 量化步长 3.73 px；logits 范围 −83–3.4 → 步长 0.339；
- 输出裁剪后 mAP 0.2747 → 0.2976，但仅恢复约 1/4 损失。

## 3. 可直接使用的模型

目录：`D:\Datasets\adas_edge_training\rknn_b05o\`

| 文件 | 用途 | 精度 | 备注 |
|---|---|---|---|
| `b05n_p2_multiclass_512x288_split_fp16.rknn` | P2 板端基准 | 0.3644 | **推荐先测** |
| `b05n_p2_clip_int8_500.rknn` | P2 速度参考 | 0.3031 | 不可用于效果对比 |
| `bdd_yolov8n_512x288_split_fp16.rknn` | BDD 板端基准 | 0.3686 | **推荐先测** |
| `bdd_yolov8n_512x288_split_int8.rknn` | BDD 速度参考 | 0.2995 | 不可用于效果对比 |
| `b05l_p2_specialist_512x192_split_fp16.rknn` | Specialist 板端基准 | 未验证 | 见第 6 节 |
| `b05l_p2_specialist_512x192_split_int8.rknn` | Specialist 速度参考 | 未验证 | 见第 6 节 |

输入输出约定：

```text
输入：RGB uint8 NHWC，mean=0，std=255
BDD：  [1,288,512,3]  输出 boxes [1,4,3024]  + scores [1,10,3024]
P2：   [1,288,512,3]  输出 boxes [1,4,12240] + scores [1,10,12240]
Spec： [1,192,512,3]  输出 boxes [1,4,8160]  + scores [1,1,8160]
```

**注意**：`*_split_fp16/int8` 是从 `*_split_logits.onnx` 转的，scores 是 **logits**，解码前需要 `sigmoid`。  
原始 `bdd_yolov8n_512x288_split.onnx` 是概率，不需要 sigmoid。不要混用。

## 4. 板端待办（P0）

当前主机没有设备：

```text
adb devices        -> 空
hdc list targets   -> 空
```

接板后按顺序：

1. 确认板端 runtime 能加载 `.rknn`，输入 `RGB uint8 NHWC`；
2. 单模型 NPU-only：预热 20 次，计时 200 次，输出 P50/P95；
3. 单模型端到端：预处理 + 推理 + 解码 + NMS；
4. BDD 10 Hz、Specialist 5 Hz、双模型顺序执行的端到端延迟；
5. 连续跑 10 分钟，记录 CPU/NPU 占用、内存、丢帧；
6. 先测 **FP16**，再测 INT8；
7. 用同一段视频、同一阈值（建议 conf=0.25、NMS IoU=0.45，与 B03C 一致）。

需要回答：

```text
FP16 双模型能否稳定 15 次前向/秒？
FP16 单 P2 能否稳定 10 Hz？
如果 FP16 不行，INT8 的 6–7 点 mAP 损失是否可接受？
```

## 5. 复现命令

WSL 环境：

```bash
wsl -d Ubuntu
source /home/jingyuwang/.venvs/rknn232/bin/activate
cd /mnt/c/Users/JingyuWang/adas-edge-b05o/deployment/rknn_work
```

转换 FP16：

```bash
python rknn_build.py \
  --onnx /mnt/d/Datasets/adas_edge_training/rknn_b05o/b05n_p2_split_logits.onnx \
  --output /mnt/d/Datasets/adas_edge_training/rknn_b05o/p2_fp16.rknn \
  --precision fp16
```

转换 INT8：

```bash
python rknn_build.py \
  --onnx /mnt/d/Datasets/adas_edge_training/rknn_b05o/b05n_p2_split_logits_clip.onnx \
  --output /mnt/d/Datasets/adas_edge_training/rknn_b05o/p2_int8.rknn \
  --calib-dir /mnt/d/Datasets/adas_edge_training/rknn_b05o/calib_train_full_500 \
  --precision int8
```

精度评估：

```bash
python rknn_eval.py --kind multiclass --precision fp16 \
  --split-onnx /mnt/d/.../b05n_p2_split_logits.onnx \
  --rknn /mnt/d/.../p2_fp16.rknn \
  --onnx /mnt/d/Datasets/adas_edge_training/b05n_final/b05n_p2_multiclass.onnx \
  --images /mnt/d/Datasets/adas_edge_training/b05n_bdd_multiclass/images/test \
  --labels /mnt/d/Datasets/adas_edge_training/b05n_bdd_multiclass/labels/test \
  --offset 200 --limit 800 --conf 0.001 --iou 0.7 --mode map --output /mnt/d/.../result.json
```

> 模拟器很慢：INT8 约 170 ms/图，FP16 约 1.5 s/图。800 张 FP16 要 20 分钟，别用全量做快速对比。

## 6. 已知坑

1. **`load_rknn()` 不能在模拟器推理**，必须 `load_onnx` + `build` + `init_runtime()`。
2. **`accuracy_analysis` 输入是 NCHW float**，不是 NHWC uint8。
3. **hybrid step2 不允许修改 scale/zero_point**；内部 operand 名称（`Constant`、`_i1`、`-rs`）也不能加入 `custom_quantize_layers`。
4. **校准图必须来自 train split**，不能与 test 200–999 重叠。
5. **combined 输出 `[1,14,N]` 的 INT8 会丢掉全部分数**：concat 的量化范围被坐标主导。必须用 split boxes/scores 输出。
6. **logits vs prob 未定论**：logits 低置信度保真更好，但本模型 logits 有 −83 极值；概率模式测试因模拟器过慢未完成。换人后可先跑 100 张小集对比。
7. **Specialist 评估脚本有 bug**：`rknn_eval --kind single --mode tlmap` 对 ONNX 参考也给出 AP 0（FP 114、TP 0），说明预处理/坐标映射不对，不是模型问题。B05L 既有结果是 AP50 0.6107、8–16px Recall 0.5750，以那个为准。修正方向：用 `b05l_eval.py --backend onnx` 的 letterbox 路径，不要用当前 resize 路径。
8. **MMSE 未完成**：跑到 70/142 层、39 分钟后中止，ETA 还需约 71 分钟；没有结果 JSON。需要时让它整夜跑。
9. **不要覆盖** `D:\Datasets\adas_edge_training\b05n_final\` 和正式部署权重。

## 7. 文件索引

```text
D:\Datasets\adas_edge_training\rknn_b05o\
  *.rknn                         转换结果
  *_map.json                     精度结果
  calib_train_full_{200,500,1000}\  校准图
  calib_train_full_{200,500,1000}.txt
  calib_train_spec_{200,500,1000}\
  hybrid_p2\                      hybrid step1 产物与 cfg
  accuracy_analysis_p2\image_{0,1}\error_analysis.txt
```

脚本：

```text
deployment/rknn_work/
  rknn_build.py
  rknn_eval.py
  rknn_make_calib_train.py
  rknn_accuracy_analysis.py
  rknn_hybrid_step1.py
  rknn_hybrid_cfg.py
  rknn_hybrid_eval.py
  clip_onnx_outputs.py
  run_bdd_spec_rknn.sh
```

完整报告：`research/rk3566_models/reports/B05P_quantization_investigation.md`  
视频对比：见 B05O 报告与 `D:\Datasets\adas_edge_training\b05o_video_scheduling\`

## 8. Git

```text
分支：research/rk3566-b05o-video-scheduling
B05O 提交：b3299f1（远程）
B05P：本文与报告提交后更新
```

## 9. 建议的下一步顺序

1. **接板，先测 FP16**：BDD 单模型、Specialist 单模型、双模型顺序执行；
2. 用板端数据回答“15 次/秒是否可行”；
3. 若 FP16 可行 → 直接用 FP16 做双模型/单模型效果对比；
4. 若 FP16 不可行 → 再评估 INT8（接受 6–7 点损失）或 QAT；
5. Specialist 评估修好后再补 AP；当前不要引用 `spec_*_tlmap.json` 的数字。
