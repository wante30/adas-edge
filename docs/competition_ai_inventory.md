# Competition AI Inventory

盘点日期：2026-09-06。以下内容只依据当前工作树、Git 历史和仓库内报告整理；未把工作树中未提交文件当作已发布结论。

## Existing Models

| Model | Input | Precision | Path | Source | Verified on RK3566 |
|---|---:|---|---|---|---|
| YOLO11n | 640 | FP16 RKNN | `deployment/rknn/yolo11n_rk3566_fp16.rknn` | COCO `yolo11n.pt` | YES (deployment README, same-frame smoke check) |
| YOLO11n | 640 | INT8 RKNN, split outputs | `deployment/rknn/yolo11n_rk3566_int8_split.rknn` | COCO `yolo11n.pt`, split `boxes`/`scores` head | YES (RK3566, 30-run benchmark) |
| YOLO11n | 640 | INT8 RKNN, unsplit | `deployment/rknn/yolo11n_rk3566_int8.rknn` | COCO `yolo11n.pt` | Loads/runs, but INVALID detection output: class scores collapse to zero |
| YOLO11n | 320 / 416 | INT8 RKNN | No model file found in current repository | Referenced by board scripts/reports, external board workspace not present | Reports exist, model path is external/unverified here |

No `.pt` or `.onnx` model is tracked in this repository. The documented source checkpoint is `C:\Users\JingyuWang\yolo11n.pt` (COCO; SHA-256 recorded in `docs/environment.md`).

## Existing Inference Pipeline

| Component | Status | Evidence |
|---|---|---|
| Letterbox preprocess (BGR to RGB, NHWC uint8) | DONE | `tools/profile_rknn_single_stream.py` |
| RKNN Lite inference on RK3566 | DONE | `deployment/rknn/benchmark_lite.py`, deployment README |
| YOLO11 raw-head DFL decode | DONE | `tools/profile_rknn_single_stream.py` |
| Split-output decode, confidence filtering, class-wise NMS | DONE | `deployment/rknn/decode_yolo_output.py` |
| Coordinate restoration from letterbox | DONE | raw-head decoder |
| Annotated visualization | PARTIAL | Existing `outputs/...annotated.png` is generated; reusable competition video evaluator is absent |
| Single-stream profiling | DONE | `tools/profile_rknn_single_stream.py` |
| Multi-stream / round-robin replay | DONE | `tools/run_rr_e2_board.py`, `tools/run_rr_hetero.py`, memmap replay tools |
| Unified `CompetitionDetector` API | MISSING | Current public detector is `UltralyticsDetector`; no RKNN adapter with requested metadata API found |

The MOT17 oracle and memmap replay code can reuse the detector-side preprocessing, RKNN invocation, and raw decode for ordinary road video. Its scheduling/evaluation protocol is person-centric and must not be treated as a road-scene acceptance evaluator.

## Existing Benchmarks

All values below are copied from repository reports. Board values are explicitly marked **MEASURED ON RK3566**.

| Input | Mean latency / throughput | p95 | CPU | NPU | Data source |
|---:|---:|---:|---|---|---|
| 320 | End-to-end pipeline 36.081 s / 1000 frames = 27.715 FPS; NPU-only mean 21.804 ms | NPU-only 23.007 ms | Reported via resource CSV | RK3566 NPU | `benchmarks/rk3566_single_stream_2026-09-01/pipeline_320_summary.md` |
| 416 | End-to-end pipeline 46.016 s / 1000 frames = 21.731 FPS; NPU-only mean 32.161 ms | NPU-only 33.466 ms | Reported via resource CSV | RK3566 NPU | `.../pipeline_416_summary.md` |
| 640 | End-to-end profile summary is present; NPU-only mean 70.692 ms | NPU-only 74.438 ms | Reported via resource CSV | RK3566 NPU | `.../pipeline_summary.md`, `.../isolation_validation/` |
| 640 split INT8 | 30-run inference mean 96.206 ms, 10.394 FPS | Not reported in README | RSS 122,990,592 bytes after 30 runs | RK3566 NPU | `deployment/rknn/README.md`, `benchmarks/results/rk3566_yolo11n_int8_split_2026-09-01.json` |

The 320/416 values are from existing RK3566 reports, but the corresponding model binaries are not present in this checkout. No new benchmark was run during this inventory.

## Existing Competition-Relevant Classes

The documented detector uses COCO YOLO11n classes. Confirmed ADAS filter classes are:

- **SUPPORTED:** `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `traffic light`, `stop sign`.
- **UNSUPPORTED by the documented COCO model:** `traffic cone`, `road obstacle`, `zebra crossing`, and `lane marking` as dedicated classes.
- COCO `traffic light` does not encode red/yellow/green state; the architecture document requires an ROI classifier or HSV stage and treats unknown state as STOP.

## Missing Items for Competition

1. A unified RKNN-backed `CompetitionDetector` interface carrying `stream_id`, `frame_id`, and timestamp metadata.
2. A road-video acceptance tool producing MP4, CSV, class counts, per-frame counts, latency statistics, empty-frame rate, and confidence distributions.
3. A competition evaluation manifest and road-video README. No competition road videos were found in the current repository, so no results can be generated without supplied media.
4. A same-video, 320-vs-416 comparison. The current checkout lacks both model binaries and road-video inputs needed for a valid comparison.
5. Failure evidence for deciding fine-tuning. Current artifacts do not establish a road-scene miss rate or justify training.

## Working-Tree and History Notes

`git status` shows pre-existing modifications and many untracked MOT17 audit artifacts, including `tools/profile_rknn_npu_only.py` and generated files. The relevant deployment history includes commits for RKNN deployment, split-output INT8 conversion, postprocess optimization, NPU profiling, and multistream replay. These changes were not altered by this inventory.
