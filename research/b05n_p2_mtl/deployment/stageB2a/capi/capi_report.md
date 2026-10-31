# 实验 C：RKNN C API 原生 INT8 输出测速

日期：2026-10-31
板端：LubanCat RK3566 / RKNN Runtime 2.3.2 / 4×Cortex-A55
输入：`/home/cat/b05g_113421_3s.rgb` 前 60 帧（RGB565）
测试程序：`deployment/stageB2a/capi/rknn_bench.c`，同步 `rknn_run` + `rknn_outputs_get` + 后处理 + `rknn_outputs_release`

## 1. 输出张量属性

B-2a 15 个输出全部为：

- dtype：INT8（`type=2`）
- layout：NCHW（`fmt=0`）
- 量化：`RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC`（per-tensor，单一 scale/zp）

分割头：

| 输出 | shape | zp | scale |
|---|---|---|---|
| lane_cat | 1×9×288×512 | 66 | 0.49434447 |
| lane_style | 1×3×288×512 | 88 | 0.35206303 |
| drivable | 1×3×288×512 | −1 | 0.12870458 |

检测头 reg scale 0.14–0.19，cls scale 约 0.0037；全部 per-tensor。

因为三个分割头都是 per-tensor affine，同一像素 9/3/3 个通道共享 scale/zp，**直接对 int8 取 argmax 与反量化后取 argmax 等价**。

## 2. C API 测速结果（5 轮 ×60 帧 = 300 次）

### B05N P2 INT8

| 项目 | want_float=1 | want_float=0 |
|---|---|---|
| rknn_run p50 / p95 | 23.55 / 62.21 ms | 23.92 / 58.88 ms |
| RKNN_QUERY_PERF_RUN p50 | 23.54 ms | 23.91 ms |
| rknn_outputs_get p50 | 8.70 ms | **4.39 ms** |
| 检测解码 p50 | 3.29 ms | 2.08 ms |
| 完整单帧 p50 | 54.16 ms | **49.02 ms** |
| 连续 FPS | 12.51 | **14.33** |
| 检测框数（300 帧） | 4430 | 4430 |

### B-2a MTL INT8

| 项目 | want_float=1 | want_float=0 |
|---|---|---|
| rknn_run p50 / p95 | 85.18 / 228.47 ms | 87.61 / 223.53 ms |
| RKNN_QUERY_PERF_RUN p50 | 85.17 ms | 87.59 ms |
| rknn_outputs_get p50 | 38.99 ms | **11.18 ms** |
| 检测解码 p50 | 3.75 ms | 1.97 ms |
| 分割 argmax p50 | 64.67 ms | 34.11 ms |
| 完整单帧 p50 | 254.01 ms | **161.21 ms** |
| 连续 FPS | 3.41 | **4.95** |
| 检测框数（300 帧） | 3380 | 3380 |

`want_float=0` 的收益：

- B-2a 输出获取 **38.99 → 11.18 ms，节省 27.81 ms（−71.3%）**
- B05N 输出获取 **8.70 → 4.39 ms，节省 4.31 ms（−49.5%）**

## 3. 原生 INT8 分割一致性

在 60 帧、8,847,360 个像素上，C API `want_float=0` 的 int8 argmax 与 `want_float=1` 的 float argmax 对比：

| 输出 | 像素一致率 |
|---|---|
| lane_cat | **1.000000** |
| lane_style（车道内） | **1.000000** |
| drivable | **1.000000** |

分割类别占比也与 float 完全一致：车道 2.83%、斑马线 1.82%、车道内实线 25.24% / 虚线 74.76%、direct 3.19% / alternative 1.43%。没有全零或全背景。

检测 Raw-head 在 C 中正常解码并做 NMS：B05N 4430 框、B-2a 3380 框（与 Python 的 4440/3395 接近，差异来自预处理插值）。

## 4. 与 Python RKNNLite 对比

| 指标 | Python RKNNLite | C API int8 |
|---|---|---|
| B05N RKNN 调用 p50 | 46.0 ms | 23.9 ms |
| B-2a RKNN 调用 p50 | 161.2 ms | 87.6 ms |
| B-2a 完整单帧 p50 | 244.6 ms | 161.2 ms |
| B-2a 连续 FPS | 3.55 | **4.95** |
| B05N 连续 FPS | 10.45 | **14.33** |

C API 把 B-2a 的 RKNN 调用从约 159–161 ms 降到 **87.6 ms**（−71～73 ms），主要省掉 Python 侧的输出取回/转换；`want_float=0` 再把输出获取从 39 ms 降到 11 ms。

## 5. 能否达到 6 FPS

- B-2a int8 完整单帧 **p50 161.21 ms**，按 p50 理论约 **6.2 FPS**。
- 实测连续 **4.95 FPS**；p95 单帧 394.69 ms，且 Qt/HUD 仍占用约 80% CPU，抖动把平均值拉低。
- 因此：**没有达到稳定的 6 FPS**，但已经从 3.4 FPS 提升到约 5 FPS。

剩余瓶颈：

- `rknn_run` / PERF_RUN 仍约 **87.6 ms**（最大项）
- 分割 argmax **34.1 ms**
- 预处理 **16.4 ms**（C 版双线性，比 Python OpenCV 慢）
- 输出获取已降到 **11.2 ms**，不再是主要瓶颈

## 6. 结论

- C API + `want_float=0` 确实有效：B-2a 输出获取节省 71%，完整 FPS 3.4→4.95。
- 原生 int8 分割与 float 预测 **100% 一致**，per-tensor 量化允许直接跨通道 argmax。
- 仍未达到稳定 6 FPS，主因是模型执行本身 87.6 ms 和分割 argmax 34 ms，不是输出搬运。
- 只完成本实验，未整合进正式 AI Bridge；未改 Qt、PCIe、系统库或模型权重。
- 测试前停止 `adas-ai-bridge-live`，结束后已恢复并确认 active。
