# Candidate-plus RK3566 deployment

This directory contains the reproducible deployment path for the PR #16
Candidate-plus road-segmentation model.  It is independent of the YOLO P2
competition path and does not modify the running competition program.

## Files

| File | Purpose |
| --- | --- |
| `common.py` | RGB letterbox, RKNN input, sigmoid/restore, color overlay, legend |
| `prepare_calibration.py` | Extract 192 real-road frames, write NCHW uint8 RGB `.npy` + `calibration.txt` |
| `convert_rknn.py` | ONNX -> FP16 or INT8 RKNN; also used for QAT QDQ ONNX |
| `candidate_plus_torch.py` | Minimal PyTorch architecture for FP32 reference |
| `board_benchmark.py` | Board-only 100-run latency/RSS/CPU/NPU benchmark |
| `board_batch_infer.py` | Board batch inference, saves raw output `.npy` for PC comparison |
| `board_video_overlay.py` | Full 1080p video pipeline with per-stage timings |
| `board_concurrent_bench.py` | YOLO P2 + Candidate-plus single-NPU sharing test |
| `pc_evaluate.py` | PyTorch/ONNX/RKNN accuracy comparison + same-frame figures |
| `pc_overlay_single.py` | One-image FP32 ONNX overlay |
| `prepare_eval_inputs.py` | Build letterboxed evaluation bundle |

## Conversion

Use the existing WSL RKNN-Toolkit2 2.3.2 environment (Python 3.10).

```bash
python prepare_calibration.py \
  --video /path/video_a.mp4 --video /path/video_b.mp4 --video /path/video_c.mp4 \
  --output-dir outputs/rk3566_candidate_plus/calibration --limit 192

python convert_rknn.py \
  --onnx outputs/fpga_road_seg_stage2_qat/exports/model.preprocessed.onnx \
  --output outputs/rk3566_candidate_plus/models/candidate_plus_fp16.rknn \
  --precision fp16

python convert_rknn.py \
  --onnx outputs/fpga_road_seg_stage2_qat/exports/model.preprocessed.onnx \
  --output outputs/rk3566_candidate_plus/models/candidate_plus_int8.rknn \
  --precision int8 \
  --dataset outputs/rk3566_candidate_plus/calibration/calibration.txt

# QAT QDQ direct route: RKNN recognizes the QAT graph and emits INT8.
python convert_rknn.py \
  --onnx outputs/fpga_road_seg_stage2_qat/exports/qat.int8.qdq.onnx \
  --output outputs/rk3566_candidate_plus/models/candidate_plus_int8_qat.rknn \
  --precision fp16 --optimization-level 2
```

`prepare_calibration.py` writes **NCHW** `.npy` because the Toolkit calibration
loader requires the model input layout.  Deployment still feeds NHWC uint8 to
RKNN Lite.

## Board run

```bash
# pure NPU latency, 100 runs
python3 board_benchmark.py --model candidate_plus_int8_qat.rknn \
  --input-npy sample.npy --warmup 10 --runs 100 --json result.json

# full video, 1080p30
python3 board_video_overlay.py --model candidate_plus_int8_qat.rknn \
  --source video_20260929_112507.mp4 --output overlay.mp4 \
  --json video.json --csv video.csv
```

The board's eMMC root was full during this run; use `/tmp` (tmpfs) for board
artifacts.  Do not delete the competition files to make room.

## PC evaluation

```bash
python pc_evaluate.py --manifest eval/manifest.json \
  --onnx model.preprocessed.onnx --onnx-qat qat.int8.qdq.onnx \
  --torch-checkpoint best.pt --repo-root . \
  --rknn-fp16-dir eval/rknn_fp16 --rknn-int8-dir eval/rknn_int8 \
  --rknn-qat-dir eval/rknn_qat --out-dir eval/accuracy
```

## Verified results (2026-11-14)

- FP16: 18.096 ms mean / 55.26 FPS.
- FP32-PTQ INT8: 9.213 ms / 108.54 FPS.
- QAT INT8 (final): 9.228 ms mean, P50 9.086 ms, P95 10.964 ms / 108.37 FPS.
- 15 s 1080p preview: 3.995 FPS wall; postprocess ~165 ms/frame.
- RGB color order confirmed: FP16 vs ONNX RGB MAE 0.0081, vs BGR 1.135.
- FP32-PTQ INT8 was rejected because its mask agreement was worse than the
  QAT direct conversion (e.g. drivable IoU 0.633 vs 0.779 against FP32).
- Full report: `outputs/rk3566_candidate_plus/REPORT_CN.md`.
