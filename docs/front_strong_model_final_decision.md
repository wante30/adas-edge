# FRONT 主检测器最终决定：KEEP CURRENT YOLO11N

日期：2026-09-21　分支：`competition-front-accuracy-sprint`

## 结论

# KEEP CURRENT YOLO11N

**不升级。** FRONT 主检测器维持 **YOLO11n COCO-80 @ 512×288 @ 10 FPS**；
SIDE/BACK、traffic-light specialist、WRR 10:6:6:6、queue depth 1 全部不动。
**不再尝试 YOLO11m/l/x**，AI 主模型就此冻结。

## 硬 Gate 逐条判定

| # | Gate | 要求 | 实测/推断 | 判定 |
| --- | --- | --- | --- | --- |
| 1 | 板端 full pipeline | ≥ 8 FPS | 乐观 11.0 FPS / FLOPs 口径 **7.9 FPS**（外推，非实测） | **边界灰区** |
| 2 | 四路预算（决定项） | ≤ 100% 推理线程占用 | 乐观口径 8 FPS = **105.1%**；FLOPs 口径 8 FPS = 134.3% | **FAIL** |
| 3 | BASE-S 精度优势 | ≥ +5pp mAP50 | **+8.50pp**（0.1999 → 0.2848） | PASS |
| 4 | 旧类别不退化 | 无灾难性下降 | 全类上升（truck +10.6pp、motorcycle +13.1pp、TL +4.8pp） | PASS |
| 5 | INT8 保留提升 | 板端 per-class 不掉 | **未测**（板子未开机） | 未知 |

**Gate 2 是唯一决定性的一条**：它不是「差一点」，而是结构性不可能——
即使按最乐观的 2.05× 效率口径，FRONT 8 FPS 也要吃掉推理线程 105%，
而当前冻结配置已经用掉 87.2%。要装下 11s，FRONT 只能降到 5 FPS 且零余量，
这与 ADAS 的前向刷新要求不符，也挤压了 specialist。**升级会破坏已冻结的四路形态**，本轮禁止。

Gate 1 未实测（板子未开机），但 Gate 2 的结论不依赖它：无论 8 FPS 还是 11 FPS，预算都超。
Gate 5 因此无需继续测——精度收益再大也装不进去。

## 为什么精度收益（+8.5pp）没能翻盘

BASE-S 的精度提升是真实的，而且比预期好：

| 指标 | BASE-N-512 | BASE-S-512 | Δ |
| --- | ---: | ---: | ---: |
| mAP50 | 0.1999 | **0.2848** | **+8.50 pp** |
| mAP50:95 | 0.1183 | 0.1702 | +5.19 pp |
| macro recall | 0.1792 | 0.2556 | +7.64 pp |
| macro F1 | 0.2529 | 0.3422 | +8.93 pp |
| small-object recall | 0.0235 | **0.0590** | **2.5×** |
| person AP50 / R | 0.2873 / 0.2156 | 0.3945 / **0.2951** | +10.7 pp / +8.0 pp |
| car AP50 | 0.4284 | 0.5076 | +7.9 pp |
| truck AP50 | 0.1980 | 0.3041 | +10.6 pp |
| bus AP50 | 0.2190 | 0.2940 | +7.5 pp |
| bicycle AP50 | 0.0909 | 0.1385 | +4.8 pp |
| motorcycle AP50 | 0.1042 | 0.2356 | +13.1 pp |
| traffic light AP50 | 0.0712 | 0.1195 | +4.8 pp |

（同一冻结 holdout、同一 14908 个 GT、同一 COCO 评估器、同一 conf/匹配口径。）

但**算力是硬墙**：这个提升需要 3.31× 的计算量，而 RK3566 的 NPU 已经被四路配置用到 87%。
一条 27.3 FPS 的流水线装不下一帧 106 ms 的前向。精度再好，跑不动就没有意义。

## 重开条件（唯一）

本轮决策到此关闭。只有满足**全部**下列条件才值得重开，且必须新开一轮，不在本轮继续：

1. **换平台**：RK3588/RK3576 等多核 NPU 平台（当前 RK3566 单核 0.8 TOPS 装不下），
   或比赛把前向降到单路；
2. **拿到合法道路 detection 训练数据**：BDD100K detection train 仍不可得；需 BDD train
   或等价的有真实 bbox 的道路 detection 数据集（**不得**再用 COCO road subset、
   不得把 segmentation mask 当 bbox、不得用冻结 holdout、不得 pseudo-label 冒充 GT）；
3. **保留 80 类 head + COCO replay 20~40%**，并提供 head 行完整保留的证据；
4. 板端先跑 §5 in [front_strong_model_benchmark.md](front_strong_model_benchmark.md#5-开机后要跑的最小实测一条命令10-分钟内出结论)，
   确认 NPU mean 落在 62–101 ms 区间（验证跨档位外推），再谈训练。

如果只是「再调调参看看」，答案已经是否——本轮已经把这个方向结束掉了。
