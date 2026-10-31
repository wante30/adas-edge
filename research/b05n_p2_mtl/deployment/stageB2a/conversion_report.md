# B05N P2 MTL B-2a → RK3566 转换报告

源 checkpoint：`runs/stageB2a/mtl_best.pt`（只读，未修改）
转换目录：`deployment/stageB2a/`
工具链：RKNN Toolkit2 2.3.2，target_platform `rk3566`，WSL Ubuntu

## 1. 产物

| 文件 | 大小 |
|---|---:|
| `b05n_p2_mtl_b2a.onnx` | 12,630,960 B |
| `b05n_p2_mtl_b2a_fp16.rknn` | 8,360,418 B |
| `b05n_p2_mtl_b2a_int8.rknn` | 4,954,226 B |

结构、输入、15 个输出与 B-1 完全一致；检测仍为 4 尺度 × 3 路 Raw-head，分割仍为 3 个头。
INT8 使用同一份 500 张 BDD 校准清单，没有重新选图。

## 2. ONNX 输出

输入：`1×3×288×512`，NCHW，RGB，float32，0–1。

检测 12 路：
`reg_s4/s8/s16/s32`（1×64×H×W，DFL logits）、
`cls_s4/s8/s16/s32`（1×10×H×W，sigmoid 分数）、
`cls_sum_s4/s8/s16/s32`（1×1×H×W）。

分割 3 路：
`lane_cat`（1×9×288×512）、`lane_style`（1×3×288×512）、`drivable`（1×3×288×512），均为 logits。

ONNX Runtime 对 PyTorch FP32（10 张）：
Raw-head 最大差 `3.1e-5`，解码框最大差 `4.8e-4`，分割最大差 `9.5e-5`。

## 3. FP16 / INT8 对照（Toolkit2 模拟器，10 张）

| 指标 | FP16 | INT8 |
|---|---|---|
| 车道前景 IoU | 0.9992 | 0.9302 |
| 斑马线 IoU | 0.9990 | 0.8875 |
| 前景线型准确率 | 0.9994 | 0.9163 |
| 实线准确率 | 0.9990 | **0.9659** |
| 虚线准确率 | 0.9996 | **0.8929** |
| 可行驶区域 mIoU | 0.9994 | 0.9264 |
| car 框匹配率 | 1.000 | 0.985 |
| person 框匹配率 | 1.000 | 1.000 |
| traffic light 框匹配率 | 1.000 | **0.967** |
| traffic light 分数变化 | ~0 | −0.0088 |

说明：这是与 PyTorch FP32 的一致性，不是 GT 精度。

## 4. 重点检查

**Traffic light**：10 张里 GT/FP32 共 30 个交通灯框，INT8 保留 29 个，匹配率 96.7%，置信度平均只降 0.0088；没有出现分数塌缩。FP16 为 30/30。

**lane_style**：INT8 实线 96.6%，虚线 89.3%。虚线仍是弱项，和 B-1 INT8 的结论一致；量化对细、断开的虚线更敏感。车道类别与可行驶区域在 INT8 下仍可用（0.930 / 0.926）。

B-2a INT8 的分割指标比 B-1 INT8 更好（B-1：车道 0.888、斑马线 0.824、线型 0.919、可行驶 0.890），说明联合微调后的特征对量化更稳。

## 5. 结论

- ONNX / FP16 / INT8 全部转换成功，无 RKNN 不支持算子，无全零或全背景输出。
- INT8 检测端可用，交通灯没有明显退化；分割端可用，但 **lane_style 的虚线**需要板端重点确认。
- 未做 RK3566 板端实测；本轮只完成模拟器验证。板端 Lite 用 `want_float=1`，输出按 float32 解释。
- 未替换正式模型、未修改 Qt、未继续训练。

## 6. 接入 AI Bridge 的差异

与 B-1 相同：输入 RGB uint8 NHWC、`/255`；检测 Raw-head 12 路；新增 `lane_cat/lane_style/drivable` 三路反量化与 argmax。
输出顺序必须按 `output_contract.json` 名称绑定，不能依赖索引猜测。
