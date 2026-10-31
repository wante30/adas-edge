# MTL 训练 / 量化证据

| 文件 | 来源 | 说明 |
|---|---|---|
| `stageB1_eval.json` | `runs/stageB1/eval.json` | B-1 分割指标（1000 张 val） |
| `stageB1_summary.json` | `runs/stageB1/summary.json` | B-1 训练元数据 |
| `stageB1_int8_validation.json` / `stageB1_fp16_validation.json` | `deployment/stageB1/` | B-1 RKNN 模拟器一致性 |
| `stageB2a_eval_seg.json` | `runs/stageB2a/eval_seg.json` | B-2a 分割指标 |
| `stageB2a_det_map50.json` | `runs/stageB2a/det_map50_holdout1000.json` | B-2a 检测 mAP（1000 张） |
| `stageB2a_tl50.json` | `runs/stageB2a/tl50.json` | B-2a BDD50 交通灯 |
| `stageB2a_summary.json` | `runs/stageB2a/summary.json` | B-2a 训练元数据 |
| `stageB2b_*` | `runs/stageB2b/` | B-2b 对照 |
| `int8_analysis.json` / `fp16_analysis.json` | `deployment/stageB2a/` | FP16/INT8 模拟器一致性 |
| `output_contract.json` | `deployment/stageB2a/` | 15 输出类别/尺寸协议 |
| `output_contract_seg256.json` | `deployment/stageB2a/seg256/` | Seg-256 协议 |
| `bench_seg_experiments.json` | `deployment/stageB2a/seg256/` | 原版 vs 跳过后处理 vs Seg-256 |
| `consistency_extra.json` | `deployment/stageB2a/seg256/` | Seg-256 一致性 |
| `seg256_report.md` | `deployment/stageB2a/seg256/` | Seg-256 报告 |

指标口径：

- 分割：1000 张 BDD val，车道前景 IoU、斑马线 IoU、前景线型准确率、可行驶 mIoU。
- 检测：1000 张 BDD holdout，conf 0.25 / IoU 0.5。
- 交通灯：BDD50，150 ROI GT，conf 0.25 / IoU 0.5，一对一匹配。
- INT8/FP16：Toolkit2 模拟器，10 张，与 PyTorch FP32 的一致性，不是 GT 精度。

权重/ONNX/RKNN 不入库；SHA256 见 `artifacts/manifests/model_manifest.json`。
