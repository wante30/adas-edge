# B+C 组合实测：Seg-256 + RKNN C API int8

日期：2026-10-31
板端：LubanCat RK3566 / RKNN Runtime 2.3.2
输入：`/home/cat/b05g_113421_3s.rgb` 前 60 帧（RGB565）
模型：已有 `b05n_p2_mtl_b2a_seg256_int8.rknn`（未重新导出/量化）
程序：`rknn_bench.c`，`want_float=0`，同步 run + outputs_get + det NMS + seg argmax

## 1. 测速结果

| 指标 | B：Seg-256（Python） | C：原版 C API int8 | B+C |
|---|---|---|---|
| RKNN 执行 p50 | 117.8 ms* | 90.61 ms | **80.46 ms** |
| outputs_get p50 | 含在调用内 | 11.51 ms | **7.98 ms** |
| 分割后处理 p50 | 14.7 ms | 33.73 ms | **7.40 ms** |
| 检测后处理 p50 | — | 2.02 ms | 2.02 ms |
| 预处理 p50 | 8.3 ms | 16.40 ms | 17.70 ms |
| 完整单帧 p50 | 161.1 ms | 171.75 ms | **122.28 ms** |
| 完整单帧 p95 | 436.2 ms | 366.13 ms | 373.48 ms |
| 连续处理 FPS | 4.79 | 4.95 | **5.97** |
| CPU | — | 42.7% | 38.1% |
| NPU 均值 / 峰值 | — | 55.6% / 72% | 66.6% / 82% |
| 分割一致性 | 基准 | 原版 100% | **100%** |

\* B 的 117.8 ms 是 Python `rknn.inference()` 时间，包含输出获取，不能直接等同于 C 的纯 `rknn_run`。

B+C 的 p50 分解（122.28 ms）：

- `rknn_run` 80.46 ms
- 预处理 17.70 ms
- 输出获取 7.98 ms
- 分割 argmax 7.40 ms
- 检测解码/NMS 2.02 ms
- release ~0.01 ms
- 其余为 inputs_set/框架开销

## 2. 提升

相对 C 原版：

- FPS：4.95 → **5.97**（+20.6%）
- 完整单帧 p50：171.75 → **122.28 ms**（−28.8%）
- RKNN 执行：90.61 → 80.46 ms（−11.2%）
- 输出获取：11.51 → 7.98 ms（−30.7%）
- 分割后处理：33.73 → 7.40 ms（−78.1%）

相对 B（Python Seg-256）：

- FPS：4.79 → **5.97**（+24.6%）
- 完整单帧 p50：161.1 → 122.28 ms（−24.1%）

## 3. 是否达到 6 FPS

- 实测连续 **5.97 FPS**，**没有严格超过 6 FPS**，但已在 6 FPS 边缘（差约 0.5%）。
- p50 单帧 122.28 ms 理论上约 8.2 FPS；p95 373.48 ms 和 Qt/HUD 负载把连续值拉到 5.97。
- 结论：B+C 基本达到 6 FPS 水平，但不能宣称稳定 ≥6 FPS。

## 4. 精度一致性

### C API int8 vs B Python Seg-256

用完全相同的一份 Python 预处理输入（60 帧）分别跑：

- B：Python RKNNLite，`want_float=1`，argmax
- B+C：C API，`want_float=0`，原生 int8 argmax

结果（2,211,840 像素）：

| 输出 | 一致率 | IoU |
|---|---|---|
| lane_cat | **1.000000** | 1.000 |
| lane_style（车道内） | **1.000000** | — |
| drivable | **1.000000** | 1.000（mIoU） |

类别占比也完全一致：车道 2.811%、实线 24.995%、虚线 75.005%。

### 检测

- C API Seg-256：3380 框 / 300 帧
- B Python Seg-256：3395 框 / 300 帧
- 差异约 0.4%，来自 C 预处理的双线性插值与 OpenCV 的细微差别，不是 Raw-head 解码或量化布局错误。

### C int8 vs C float

同一 Seg-256 模型内，`want_float=0` 与 `want_float=1` 的分割 argmax 一致率 **100%**（2,211,840 像素）。输出均为 per-tensor AFFINE 量化，通道共享 scale/zp，直接跨通道 int8 argmax 合法。

## 5. 剩余瓶颈

按 p50 时间排序：

1. **`rknn_run` 80.46 ms（约 66%）** — 模型执行本身，仍是最大项。
2. **预处理 17.70 ms（约 14%）** — C 版双线性，比 Python OpenCV 的 8.3 ms 慢，有优化空间。
3. 输出获取 7.98 ms
4. 分割 argmax 7.40 ms

B+C 已经把输出获取和分割后处理压到很小，继续优化应优先考虑预处理和模型执行，而不是再降分割分辨率。

## 6. 结论

- B+C 实测 **5.97 FPS**，比 C 原版 4.95 提升 20.6%，比 B 4.79 提升 24.6%。
- 未严格达到 6 FPS，但已在 6 FPS 边缘。
- C API 原生 int8 解码与 B 的 Seg-256 输出 **100% 一致**，没有引入额外精度损失；分割精度代价仍只来自 Seg-256 下采样。
- 只完成本轮 B+C 实测，未整合进正式 AI Bridge、未替换正式模型、未修改 Qt/PCIe/HUD。
- 测试前停止 `adas-ai-bridge-live`，结束后已恢复并确认 active。
