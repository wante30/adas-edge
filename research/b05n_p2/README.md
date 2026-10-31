# B05N P2 十类检测研究


> 运行：从仓库根目录执行，或设置 `PYTHONPATH=.`。共享模块已放在 `tools/`（b05k_common / b05k_runtime / b05n_tl_eval / b05l_eval）。

本目录保存 B05N P2（YOLOv8n-P2）的训练、评估、ONNX/RKNN 转换和交通灯专项工具。
权重、ONNX、RKNN 和数据集不进入普通 Git；SHA256 见
`artifacts/manifests/model_manifest.json`。

## 1. 模型

- 结构：Ultralytics `yolov8-p2.yaml`，4 尺度 P2/P3/P4/P5。
- 输入：512×288，RGB。
- 类别：person/rider/car/truck/bus/train/motor/bike/traffic light/traffic sign。
- 输出：Raw-head，12 路（每尺度 reg 64、cls 10、cls_sum 1），host 端 DFL+NMS。
- 最终权重：
  - `D:\Datasets\adas_edge_training\b05n_final\b05n_p2_multiclass.pt`
  - SHA256 `c249f7fffc828c892c8308c0f5a4768bb4c0a3c3ab28434d2e70e843b5e278ed`
- 部署 RKNN：`b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn`
  - SHA256 `65089ef0697bd7378071290baa218905cdcaed768452678da440b8eb116b2a5f`

## 2. 结果

- BDD holdout-1000：mAP50 0.3711，mAP50-95 0.1944。
- BDD50 交通灯：AP50 0.5205，8–16px Recall 0.4125。
- INT8 Raw-head：mAP50 0.3651，TL AP50 0.4733。
- 对比结论与完整表格见 `reports/B05N_multiclass_P2_comparison.md` 和
  `docs/models/MODEL_COMPARISON.md`。

## 3. 工具

| 文件 | 用途 |
|---|---|
| `tools/b05n_build_dataset.py` | BDD 检测标签 → YOLO 数据集 |
| `tools/b05n_train.py` | P2 训练入口 |
| `tools/b05n_eval_multiclass.py` | 10 类 holdout 评估 |
| `tools/b05n_tl_eval.py` | BDD50 交通灯专项 |
| `tools/b05n_parity.py` | PyTorch/ONNX 一致性 |
| `tools/b05n_video.py` | 官方视频逐帧推理/可视化 |
| `tools/b05n_timing.py` | PC 延迟测量 |
| `tools/split_yolo_multiscale.py` | 多尺度 Raw-head split |
| `tools/rknn_build.py` | RKNN FP16/INT8 转换 |
| `tools/rknn_make_calib.py` / `rknn_make_calib_train.py` | 校准集生成 |
| `tools/rknn_eval.py` | RKNN 精度评估 |
| `tools/rknn_accuracy_analysis.py` | 逐层 cosine / INT8 误差定位 |
| `tools/inspect_onnx_tail.py` / `clip_onnx_outputs.py` | ONNX 输出检查/裁剪 |

## 4. 报告

- `reports/B05N_multiclass_P2_comparison.md`：P2 vs BDD/标准 YOLOv8n、BDD50、视频验证。
- `reports/B05P_quantization_investigation.md`：INT8 精度损失定位。
- `reports/B05P_handoff.md`：板端交接、复现命令、已知坑。
- `reports/B05P_assets_README.md`：资产说明。

## 5. 复现

```bash
python research/b05n_p2/tools/b05n_train.py --help
python research/b05n_p2/tools/b05n_eval_multiclass.py --help
python research/b05n_p2/tools/rknn_build.py --help
```

训练数据/权重路径按本机实际目录调整；不要用仓库内旧 ONNX/PT 覆盖板端正式模型。
