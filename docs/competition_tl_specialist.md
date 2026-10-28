# Traffic-Light Specialist（FRONT-only 专项支路）构建与验收

更新日期：2026-09-18。主 YOLO、多流调度与既有比赛配置**零改动**；本支路是独立模型、独立画布、独立量化。

## 1. 结论速览

| 问题 | 答案 |
| --- | --- |
| 数据 | BDD100K val **非 holdout** 9000 张（23468 个 TL 框进入训练），与验收集同域；nuImages 5k 无 TL 实例（已排除） |
| 模型 | YOLO11n 单类头（COCO 预训练骨干），独立于主检测器 |
| 画布 | **512×192**，输入 = 全宽 × 顶部 60% ROI 裁剪（1280×432）letterbox |
| 训练 | `imgsz=512, rect=True, mosaic=0`，60 epochs，batch 16，8513 train / 487 val |
| PC FP32 holdout | TL AP50 **0.1362**（主模型最优 0.0830）、R@0.05 **0.272**（主 0.149）、候选覆盖 **0.445**（主 0.323） |
| 板端部署模型 | **混合量化 INT8**（框链 25 层 fp16），NPU **25.3 ms** |
| 板端效果 | TL recall@0.05 **0.199** ≈ 主模型板端（0.095）的 **2.1×** |
| NPU 预算 | 3 FPS +7.6%（推理线程 78%→85%）；5 FPS +12.7%（→91%）。**两档都放得下** |
| Gate | **PASS**（采用 specialist 支路） |

## 2. 数据（S1）

来源与切分规则详见 `D:/Datasets/adas_edge_training/tl_specialist/data_512x192/dataset_card.json`（SHA256 见 manifest）：

- 源：`bdd100k.csv` 的 val split，**按名字排除冻结 holdout 的 1000 张**（holdout 仅用于验收，绝不参与训练/校准/选型）；
- 全部 9000 张非 holdout 图像都进数据集（含约 3900 张无 TL 的背景帧，用于抑制误检）；
- 标注只保留 `traffic light`，框裁剪到 ROI（顶部 60%），裁剪后剩余面积 <30% 的框丢弃（682 个）；
- 切分：`md5(文件名)[:8] % 18 == 0 → val`（487 张 / 1185 框），确定性、无泄漏；
- 图像预 letterbox 到 512×192（与部署逐像素一致）。

## 3. 训练（S2）

- 工具：`tools/train_tl_specialist.py`（配方沿用本项目已验证的 road-ft 矩形训练：`rect=True`、`mosaic=0`）；
- 数据 yaml：`data_512x192/dataset.yaml`（单类 `traffic light`）；
- 最终内部 val（Ultralytics 口径）：P 0.570 / R 0.376 / mAP50 0.367；
- 产物：`tl-512x192/weights/best.pt`（SHA256 前缀 `f5fc9c2b1220d55c`）。

## 4. 验收（S3，冻结 holdout，1000 张 / 2734 个 TL 框）

协议与主模型基线完全一致（`tools/traffic_light_roi_eval.py`，ROI-B-top60 → 512×192 → 映射回全帧）：

| 指标 | 主模型最优（ROI-D+640×352） | 主模型（同 ROI+512 宽） | **Specialist FP32** |
| --- | ---: | ---: | ---: |
| AP50 | 0.0830 | 0.0489 | **0.1362** |
| R@0.05 | 0.149 | 0.083 | **0.272** |
| R@0.25 | 0.051 | 0.028 | 0.142 |
| 候选覆盖率 | 0.323 | 0.203 | **0.445** |
| 无候选 GT 占比 | 67.7% | 79.7% | **55.5%** |

真实道路视频：`ch0_front.mp4` 素材本身不含红绿灯（0 检出，属素材属性）；用含红绿灯的冻结 holdout 帧合成 25 帧片段验证端到端链路，22.2 检出/帧，标注视频 `tl_specialist/verify_annotated.mp4`（板端 `deployment/rknn/tl_specialist_annotate.py` 生成）。

## 5. RK3566 量化（S3，关键工程发现）

普通 INT8（normal+channel，与主模型同一配方）在 specialist 上**严重退化**：AP50 0.101 → 0.029。
逐分支混搭定位（pycocotools AP50，100 张子集）：

| 框来源 | 分数来源 | AP50 |
| --- | --- | ---: |
| FP32 | FP32 | 0.1011 |
| INT8 | INT8 | 0.0295 |
| **FP32** | INT8 | 0.0921 |
| **INT8** | FP32 | 0.0310 |

→ **退化几乎全部在框分支**：box 张量量程 0–512 px，INT8 步长 ~2 px，对 5×8 px 的红绿灯是致命抖动。
（主模型 7 类时同样的 2 px 抖动也存在，只是其 TL AP 本来就低而未被暴露。）

已试并否决：logits 导出（sigmoid 前切分，分数误差不是根因）、训练分布校准集（无效）、
mmse（对框抖动无效）。**有效解：混合量化**——`hybrid_quantization_step1/2` 把框链
（dfl softmax/reshape/transpose、cv2 支路、Concat_2、Mul_2，共 25 层）保持 fp16：

| 版本 | AP50（100 张子集） | R@0.05 | NPU mean |
| --- | ---: | ---: | ---: |
| FP32 参考 | 0.1011 | 0.235 | — |
| 普通 INT8 | 0.0295 | 0.098 | 20.8 ms |
| 混合 v1（框链后处理 fp16） | 0.0607 | 0.191 | 22.1 ms |
| **混合 v2（+cv2/dfl，采用）** | **0.0689** | **0.199** | **25.3 ms** |

剩余差距（ΔR −0.036）来自共享骨干的 int8 激活，继续扩大 fp16 收益递减、NPU 成本上升，止步于此。
工具链改动：`split_yolo_output.py --scores-mode`、`decode_yolo_output.py --score_logits`
（均向后兼容，主模型不受影响）。

## 6. NPU 预算（S3）

四路 WRR-FAIR 满载时推理线程占用 ~78%（LONG20 实测）。追加 FRONT-only specialist：

| 支路帧率 | 额外占用 | 合计 | 判定 |
| --- | ---: | ---: | --- |
| 3 FPS | 3 × 25.3 ms = +7.6% | 85.6% | 可行，有余量 |
| 5 FPS | 5 × 25.3 ms = +12.7% | 90.7% | 可行，偏紧（实测吞吐上限 ~27.5 FPS） |

建议起点 **3 FPS**，红绿灯状态机做时域滤波；若 CPU 余量实测充足再升 5 FPS。

## 7. 交付物与 SHA256

`D:/Datasets/adas_edge_training/tl_specialist/manifest.json`（完整哈希）：
`weights/best.pt`、`tl_specialist_512x192(.onnx/_split.onnx)`、
`tl_specialist_512x192_hybrid2.rknn`（部署模型，前缀 `becfaada5da49c4d`）、数据卡。
工具：`tools/build_tl_specialist_dataset.py`、`tools/train_tl_specialist.py`、
`tools/traffic_light_roi_eval.py`、`deployment/rknn/tl_specialist_annotate.py`。

## 8. 未做与风险

- 未做 QAT；骨干 int8 激活仍是剩余 ΔR 的来源（如需再榨，走 QAT 或全 fp16 头）；
- 四路联调中 specialist 与主检测器的并行调度未实测（预算按单帧 NPU 均值推算，边界已留 9–15%）；
- 训练数据只有 BDD val 同域帧；换更宽的道路域数据预计还能提升（未验证）。
