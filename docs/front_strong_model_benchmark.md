# FRONT 强模型板端速度评估（YOLO11s）

日期：2026-09-21　平台：**RK3566 / LubanCat**（板子本轮**未开机**，速度 Gate 无实测值）

> **证据等级说明（必须随结论一起引用）**
> - **实测**：PR 中的精确 FLOPs、PC ONNX 延迟、以及 §2 的 11n 板端 4 个实测标定点。
> - **外推**：11s 的板端耗时。基于 11n 的板端标定曲线，假设 NPU 每 FLOP 效率跨模型档位不变。
>   同族 4 点拟合残差 ≤5%，但**跨模型档位未被验证**——开机后必须用 §5 的命令实测。
> - 结论不依赖外推精度：即使取最乐观口径，冻结四路架构仍不可行（§4）。

## 1. 为什么不能直接用 PC 数字

PC ONNX Runtime 上 11s/11n = **2.05×**，而 FLOPs 比值是 **3.31×**。CPU 推理受内存带宽与
算子开销支配，比值被压缩；RK3566 的 NPU 是专用阵列，行为更接近每 FLOP 恒定。
所以板端用「标定 + FLOPs」推断，而把 2.05× 当作**乐观下界**。

## 2. 板端标定（用已有 11n 实测点）

11n 在 RK3566 上的 NPU-only 实测（`competition_rectangular_rk3566_benchmark.md`）：

| 画布 | FLOPs | 实测 NPU mean |
| --- | ---: | ---: |
| 320×192 | 0.97 GF | 16.12 ms |
| 416×256 | 1.68 GF | 23.86 ms |
| 512×288 | 2.34 GF | 30.42 ms |
| 640×352 | 3.58 GF | 47.66 ms |

线性拟合（最小二乘）：**ms = 12.07 × GFLOPs + 3.66**，四点残差 −4.7% / +0.3% / +4.9% / −1.7%（≤5%）。
截距 3.66 ms 是固定开销，斜率 12.07 ms/GF 是 NPU 的等效吞吐（≈ 83 GFLOP/s 有效）。

## 3. YOLO11s 板端耗时外推

| 口径 | NPU-only | + 画布固定开销 28.36 ms（decode 8.80 + preprocess 7.74 + postprocess 11.82，11n@512×288 实测） | full pipeline | FPS |
| --- | ---: | ---: | ---: | ---: |
| **乐观**（按 2.05× 比值） | 62.4 ms | 90.7 ms | 90.7 ms | **11.0** |
| **FLOPs 匹配**（拟合 97.1 / 直接 30.42×3.308 = 100.6） | 97–101 ms | 125–129 ms | 127.2 ms | **7.9** |

即：**最乐观能过 8 FPS，按 FLOPs 匹配则略低于 8 FPS**——速度 Gate 处于边界灰区。

## 4. 决定性约束：四路预算（与 FPS 测量无关）

冻结架构里 preprocess + NPU 串行在同一个推理线程上；当前实测该线程在 10×38.2 + 3×6×22.2 + specialist
（3 × (5.0 + 25.3)）≈ **872 ms/s = 87.2%** 占用，聚合 27.3/28 FPS 已到 97.5% 达成率，没有余量。

把 FRONT 换成 11s 后（单帧串行 = preprocess 7.74 + NPU）：

| FRONT 帧率 | 乐观口径（70.1 ms/帧） | FLOPs 口径（106.6 ms/帧） |
| ---: | ---: | ---: |
| 10 FPS | 119.1% ✗ | 155.6% ✗ |
| 8 FPS | **105.1% ✗** | 134.3% ✗ |
| 6 FPS | 91.1% △ | 113.0% ✗ |
| 5 FPS | 84.1% ✓ | 102.3% ✗ |

（>100% 即物理不可能；90% 以上意味着侧路/后处理线程没有挣扎空间，实测会掉帧。）

**即使按最乐观的 2.05× 口径，8 FPS 也是 105%——超预算。** 要保住冻结四路配置，FRONT 必须降到
5 FPS，而那时 FRONT 的检测帧率与 8-10 FPS 的 ADAS 要求不符，且仍未给 specialist 留足余量。

## 5. 开机后要跑的最小实测（一条命令，10 分钟内出结论）

```bash
# 1) NPU-only（≥600 次推理，含 warmup）
#    板端：先把 yolo11s_rk3566_int8_split_512x288.rknn 放到 ~/adas-edge-rect/front_sprint/
python3 ~/adas-edge-rect/infer_npy_batch.py \
  --model front_sprint/yolo11s_rk3566_int8_split_512x288.rknn \
  --manifest holdout100/canvas_512x288.json \
  --inputs-dir holdout100/canvas_512x288 --out-dir front_sprint/out --warmup 20

# 2) full pipeline 分段（600 帧）
python3 ~/adas-edge-rect/deployment/rknn/benchmark_pipeline.py \
  --model front_sprint/yolo11s_rk3566_int8_split_512x288.rknn \
  --video videos/ch0_front.mp4 --width 512 --height 288 --frames 600 --warmup 20 \
  --out front_sprint/pipeline_s.json
```

判读：`NPU mean` 落在 **62 ms（乐观）～101 ms（FLOPs 口径）** 之间即证实本外推；
超出该区间说明跨档位效率假设失效，需回到 §4 重算预算。

## 6. 未取得的数据（如实列出）

- 11s 的板端 NPU mean/p50/p95、full pipeline 分段、CPU/RAM/温度/NPU 频率：**未测**（板子未开机）。
- 11s RKNN INT8 的 per-class 精度（§17 要求的 holdout 子集复核）：**未测**。
- 四路 + specialist 的 10 min 长测：**未测**（这一项的前置条件 §4 已经不成立，故未启动）。
- RK3568（粤嵌，2 GB RAM）平台：本机不具备，未评估。
