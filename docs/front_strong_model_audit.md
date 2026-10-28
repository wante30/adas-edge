# FRONT 主检测器升级审计（YOLO11s 候选）

日期：2026-09-21　分支：`competition-front-accuracy-sprint`　平台：**RK3566（LubanCat，板子本轮未开机）**

## 1. 本轮范围

只评估「把 FRONT 主检测器从 YOLO11n 换成 YOLO11s」是否值得。SIDE/BACK、traffic-light specialist、
调度器、队列参数全部冻结，本轮零改动。禁止 YOLO11m/l/x。

## 2. 冻结基线（从仓库产物读取，非 prompt 近似值）

来源：`D:/Datasets/adas_edge_training/rect_eval/BASE-512x288.json`（冻结 BDD100K 1000 张 holdout，14908 个 GT 实例）

| 指标 | BASE-N-512（当前 FRONT） |
| --- | ---: |
| mAP50 | 0.1999 |
| mAP50:95 | 0.1183 |
| macro recall | 0.1792 |
| macro F1 | 0.2529 |
| small-object recall | 0.0235 |
| person AP50 / recall | 0.2873 / 0.2156 |
| car | 0.4284 / 0.3368 |
| truck | 0.1980 / 0.2165 |
| bus | 0.2190 / 0.2597 |
| bicycle | 0.0909 / 0.0632 |
| motorcycle | 0.1042 / 0.1111 |
| traffic light | 0.0712 / 0.0516 |
| 板端 NPU-only | 30.42 ms（实测） |
| 板端 full pipeline | 58.51 ms → 17.09 FPS（实测） |

参考：BASE-N-640×352 = mAP50 0.2329（本轮不作为主候选）。

## 3. 候选与构建

`yolo11s.pt`（官方 COCO pretrained，18.4 MB，**未训练**）→ 严格复用既有链路：

```
yolo11s.pt
 → deployment/rknn/export_onnx.py  --imgsz 512x288     → [1,3,288,512] 静态 ONNX
 → deployment/rknn/split_yolo_output.py                → boxes[1,4,3024] + scores[1,80,3024]
 → deployment/rknn/convert_rknn.py --precision int8    → yolo11s_rk3566_int8_split_512x288.rknn (10.9 MB)
```

未新增任何平行管线：预处理、letterbox、量化校准（沿用 `rect_models/calib_512x288.txt`，76 张道路域画布）、
解码、NMS、坐标还原全部是 11n 已冻结的同一套代码。锚点数 3024 与 11n 完全一致（同 stride/同画布）。

## 4. PC 正确性（已实测）

40 张道路图（`holdout100` 子集），conf 0.25 / NMS IoU 0.7：

| 比较 | ref det | cand det | matched | 匹配率 | 平均 IoU | 平均 ｜Δconf｜ | 判定 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| YOLO11s PyTorch FP32 → ONNX | 245 | 245 | 245 | **1.0000** | 0.9997 | 0.0003 | 一致 |

ONNX 张量形状 `[1,3,288,512]`、输出 `boxes[1,4,3024]` + `scores[1,80,3024]`，DFL 解码、类别映射
（COCO-80 索引 0/1/2/3/5/7/9 → person/bicycle/car/motorcycle/bus/truck/traffic light）与坐标还原
均按同一 decoder 验证通过。

## 5. 算力实测（决定一切的物理量）

用 `torch.utils.flop_counter.FlopCounterMode` 在**真实矩形输入**上计数（不用方形近似）：

| 画布 | YOLO11n | YOLO11s | 比值 |
| --- | ---: | ---: | ---: |
| 512×288 | 2.34 GF | **7.74 GF** | **3.308×** |
| 320×192 | 0.97 GF | 3.22 GF | 3.310× |
| 640×352 | 3.58 GF | 11.84 GF | 3.307× |

参数量 2.62 M → 9.46 M（3.61×）。**11s 在这个画布上是 11n 的 3.31 倍计算量**，这是后面所有推断的基础。

PC ONNX Runtime（CPU，median，60 次）：11n 13.5 ms vs 11s 27.7 ms = **2.05×**。
注意这个比值**低于** FLOPs 比值，说明 CPU 推理受内存/开销限制、低估了两者的真实差距；
拿它当板端预期是乐观口径。

## 6. 板端状态

本轮板子未开机，**速度硬 Gate 未取得实测值**。见
[front_strong_model_benchmark.md](front_strong_model_benchmark.md)：用 11n 在 4 个画布上的实测点做标定后外推，
两档口径（乐观 11.0 FPS / FLOPs 匹配 7.9 FPS）**都不能满足冻结四路架构的预算**。

## 7. 数据与训练

**未执行 fine-tune**。原因见最终决定文档：速度与四路预算在任何口径下都不成立，
按本轮规则（速度 Gate 不过 → STOP TRAINING，不允许继续堆模型）直接停在第二阶段。

已就绪、可用于将来重开的数据评估（本轮未使用）：BDD100K val 非 holdout 9000 张（含 23468 个 TL 框）
属于已用过的同域数据；BDD detection train 仍不可得。若将来重开，必须先解决
[§8 的合法道路 detection 数据](front_strong_model_final_decision.md#重开条件)。
