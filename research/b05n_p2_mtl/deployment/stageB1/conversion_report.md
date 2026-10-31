# B05N P2 MTL B-1 → RK3566 转换报告

源 checkpoint：`runs/stageB1/mtl_best.pt`（只读，未修改）
转换目录：`deployment/stageB1/`
工具链：RKNN Toolkit2 2.3.2，target_platform `rk3566`，WSL Ubuntu

## 1. 产物

| 文件 | 大小 |
|---|---:|
| `b05n_p2_mtl_b1.onnx` | 12,630,960 B |
| `b05n_p2_mtl_b1_fp16.rknn` | 8,360,418 B |
| `b05n_p2_mtl_b1_int8.rknn` | 4,954,290 B |

INT8 使用原 B05N P2 Raw-head 的 500 张校准清单（`b05n_p2_multiclass_raw_heads_int8_500_20261026_calib.txt`）。

## 2. ONNX 输出契约

输入：`1×3×288×512`，NCHW，RGB，float32，0–1（`raw/255`）。

检测沿用 Raw-head，4 个尺度 × 3 路 = 12 个输出：

| 输出 | shape | 含义 |
|---|---|---|
| `reg_s4/s8/s16/s32` | `1×64×H×W` | DFL logits，4 边 × 16 bin |
| `cls_s4/s8/s16/s32` | `1×10×H×W` | sigmoid 后类别分数 |
| `cls_sum_s4/s8/s16/s32` | `1×1×H×W` | 类别分数和 |

分割 3 个输出：

| 输出 | shape | 类别 |
|---|---|---|
| `lane_cat` | `1×9×288×512` | 8 类标线 + background（logits） |
| `lane_style` | `1×3×288×512` | solid / dashed / background（logits） |
| `drivable` | `1×3×288×512` | direct / alternative / background（logits） |

检测后处理仍在 host：DFL softmax/期望 → ltrb → xywh → NMS；**没有**把框和分数重新合并成 post-DFL 单输出。

## 3. 精度对照（10 张 BDD val，对比 PyTorch FP32）

### FP16

| 指标 | 值 |
|---|---|
| 车道前景 IoU | 0.9986 |
| 斑马线 IoU | 0.9971 |
| 前景线型准确率 | 0.9992 |
| 可行驶区域 mIoU | 0.9987 |
| car/person/TL 框匹配率 | 1.00 / 1.00 / 1.00 |
| 检测框最大绝对差 | 1.26 px |

### INT8（500 张校准）

| 指标 | 值 |
|---|---|
| 车道前景 IoU | 0.8881 |
| 斑马线 IoU | 0.8242 |
| 前景线型准确率 | 0.9193 |
| 可行驶区域 mIoU | 0.8896 |
| car 框匹配率 | 0.948 |
| person 框匹配率 | 1.000 |
| traffic light 框匹配率 | 0.944 |
| 置信度变化 | car −0.013，person −0.015，TL −0.017 |

说明：这里是“与 PyTorch FP32 的一致性”，不是 GT 精度；INT8 没有出现全零、全背景或类别塌缩。

## 4. 结论与风险

- ONNX 与 PyTorch 基本一致（最大差 4e-4），可以进入量化。
- FP16 RKNN 与 PyTorch 几乎无差，可作为板端对照基线。
- INT8 检测端正常，交通灯分数没有塌缩；分割端整体可用，但 **lane_style 是最弱的一项**（前景一致率 0.919，10 张里虚线/细线最容易受量化影响）。
- 输入必须是 **RGB**；BGR 时分割一致率明显下降、检测框差到 100+ px。
- 未做 RK3566 板端实测（当前没有板端测试窗口）；板端 Lite 加载时用 `want_float=1`，输出仍按 float32 解释。

建议：先用 FP16 做板端语义确认，再用 INT8 跑同一批图；如果 lane_style 在板端明显变差，保留 FP16 作为分割优先版本。

## 5. 接入 AI Bridge 需要改的地方

1. 输入改为 `1×3×288×512` RGB uint8 NHWC，保持 `/255` 归一化。
2. 检测 adapter 沿用 Raw-head 12 路解码，不需要改 DFL/NMS 逻辑。
3. 新增 3 个分割输出的反量化/argmax（`want_float=1` 时直接拿 float32）。
4. 输出顺序必须按 `output_contract.json` 的名称绑定，不能依赖索引猜测。
5. 车机侧新增多任务可视化层；不替换现有 B05N P2 正式模型。
