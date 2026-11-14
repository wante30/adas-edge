# RoadSeg-RK (RK3566)

Independent research branch for an RK3566-NPU-oriented road segmentation model.
The FPGA Candidate-plus remains the baseline and is not overwritten.

## Data

Reuse the issue-15 BDD100K / RLMD mirror and the audited 4-channel masks:

- `C:\Users\JingyuWang\datasets\rlmd_issue15\prepared_v1\mixed_manifest.jsonl`
- 62,984 BDD train, 7,012 BDD internal val, 10,000 BDD official val (held-out),
  1,698 RLMD train, 439 RLMD independent group val.

Build the 416x256 five-channel cache once:

```powershell
python -m experiments.training.roadseg_rk.roadseg_data `
  --manifest C:\Users\JingyuWang\datasets\rlmd_issue15\prepared_v1\mixed_manifest.jsonl `
  --output C:\Users\JingyuWang\datasets\bdd100k_issue15\roadseg_cache_416x256_v1 `
  --workers 16
```

Channels: `direct_drivable, alternative_drivable, solid_lane, dashed_lane,
crosswalk`.  Direct/alternative are rasterized from the original BDD `poly2d`
annotations.  BDD crosswalk curves are positive-only weak supervision and are
disabled; RLMD crosswalk is dense.  RLMD has no drivable GT, so channels 0/1
are invalid there and never become implicit background.

## Models

| name | role | input | params | MACs |
|---|---|---:|---:|---:|
| `candidate-plus-a` | original Candidate-plus at 416x256, 4ch merged drivable | 416x256 | 32,524 | 146.8M |
| `roadsegnet-5` | depthwise encoder/decoder, 5ch, width 0.85 | 416x256 | 835,136 | 223.4M |
| `mobilenetv3-lraspp` | MobileNetV3-Small + LR-ASPP + stride-8 refinement | 416x256 | 1,239,973 | 159.3M |

All models use only Conv2d, depthwise Conv2d, BatchNorm, ReLU/Hardswish, Add,
Concat, global average pooling and nearest/bilinear resize.

## Training

```powershell
# A: continue from the original Candidate-plus FP32 checkpoint
python -m experiments.training.roadseg_rk.train_roadseg `
  --index <cache_index.jsonl> --config experiments\training\roadseg_rk\configs\candidate_a_full.json `
  --output outputs\roadseg_rk\training\A_full --amp --workers 8 `
  --init outputs\fpga_road_seg_stage2_qat\checkpoints\candidate-plus\best.pt

# B: 5ch, optional TwinLiteNet+ Small drivable distillation
python -m experiments.training.roadseg_rk.train_roadseg `
  --index <cache_index.jsonl> --config experiments\training\roadseg_rk\configs\roadsegnet5_full.json `
  --output outputs\roadseg_rk\training\B_full --amp --workers 8 `
  --teacher outputs\fpga_road_seg_final\checkpoints\twin-small\best.pt

# C: MobileNetV3-Small LR-ASPP
python -m experiments.training.roadseg_rk.train_roadseg `
  --index <cache_index.jsonl> --config experiments\training\roadseg_rk\configs\mobilenetv3_full.json `
  --output outputs\roadseg_rk\training\C_full --amp --workers 8
```

The training loop supports AMP, exact resume, best/last checkpoints, per-domain
validation, cosine LR, early stopping, boundary-weighted BCE/Dice and optional
drivable distillation.  Model selection uses internal BDD val + RLMD val only.

## Evaluation

```powershell
python -m experiments.training.roadseg_rk.eval_roadseg `
  --checkpoint outputs\roadseg_rk\training\B_full\best.pt `
  --index <cache_index.jsonl> --output outputs\roadseg_rk\evaluation\B_full
```

Reports direct/alternative IoU, merged drivable IoU/F1/Precision/Recall,
solid/dashed IoU/F1 and RLMD dense crosswalk IoU/F1.  The official BDD val
split is never used for training or model selection.

## Export / RKNN

```powershell
python -m experiments.training.roadseg_rk.export_onnx `
  --checkpoint <best.pt> --index <cache_index.jsonl> --output outputs\roadseg_rk\quantization\B
python -m experiments.training.roadseg_rk.prepare_calibration `
  --index <cache_index.jsonl> --output outputs\roadseg_rk\quantization\B\calibration --path-style wsl
```

On WSL with rknn-toolkit2 2.3.2:

```bash
python deployment/rknn/candidate_plus/convert_rknn.py \
  --onnx outputs/roadseg_rk/quantization/B/model.fp32.onnx \
  --output outputs/roadseg_rk/rknn/roadsegnet_int8.rknn \
  --precision int8 --dataset outputs/roadseg_rk/quantization/B/calibration/dataset.txt \
  --optimization-level 2
```

Board scripts are in `deployment/rknn/roadseg_rk/`:
`board_benchmark.py`, `board_batch_infer.py`, `board_video_overlay.py`.

## Visualisation

```powershell
python -m experiments.training.roadseg_rk.visualize_roadseg `
  --checkpoint <best.pt> --video <input.mp4> --output <overlay.mp4> --ego
```

The overlay is composed at network resolution and alpha-blended once at native
resolution.  `--raw` disables morphology; `--raw-masks` writes per-class masks.
The ego-lane heuristic is conservative: it needs a supported left/right lane
pair; otherwise it falls back to plain drivable display.
