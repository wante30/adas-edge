# RK3566 RKNN deployment

This directory keeps the deployment boundary independent from the PC detector:

```text
Ultralytics .pt -> static 1x3x640x640 ONNX -> RKNN -> RKNN Lite on RK3566
```

## Verified detection route (2026-09-01)

`yolo11n_rk3566_int8_split.rknn` was built from COCO `yolo11n.pt`, with the
existing YOLO head exposed as separate `boxes` and `scores` outputs, and
executed on the connected LubanCat RK3566 NPU. Its board-produced output was
decoded on the PC with class-aware NMS: it contains 21 `car` detections at
confidence >= 0.25. Against the independently decoded FP16 output on this
same frame, 9 of its 10 boxes have IoU >= 0.50 matches; this is a smoke
comparison, not a dataset accuracy claim. The saved annotated proof image is
`outputs/rk3566_npu_int8_split_annotated.png` (generated artifact, not Git).

| Item | Verified value |
| --- | --- |
| RKNN model | INT8 with split outputs, 640 x 640, static NCHW ONNX source |
| Toolkit used for conversion | `rknn-toolkit2 2.3.2` |
| Board Lite API | `rknn-toolkit-lite2 2.3.2` |
| Board runtime | `librknnrt 2.3.2` |
| Board NPU driver | `0.8.2` |
| Input passed to Lite | `1x640x640x3`, `uint8`, NHWC BGR |
| Raw output | `boxes [1x4x8400]` + `scores [1x80x8400]`, `float32`, finite; non-zero class scores |
| 30-run inference mean | **96.206 ms** |
| Effective inference FPS | **10.394 FPS** |
| Process RSS after 30 runs | **122,990,592 bytes** |

The benchmark is stored as
`benchmarks/results/rk3566_yolo11n_int8_split_2026-09-01.json`. It is a real NPU
inference benchmark and a semantic detection check. It does not yet establish
full video-stream throughput, because decode, rendering and video I/O are
measured separately.

## Why the original INT8 model failed

`yolo11n_rk3566_int8.rknn` loaded and ran at about 10.455 FPS, but it is **not a
valid detection model**. YOLO11's exported head is one `1x84x8400` output that
contains both box coordinates (range about 0--640) and class probabilities
(range 0--1). With whole-output INT8 quantization, the output scale was driven
by the coordinate range; all recovered class scores were exactly zero. RGB/BGR
input tests and explicit input quantization produced the same result.

The fix is `deployment/rknn/split_yolo_output.py`: it keeps all network
operations unchanged and exposes the two tensors before the final concat. RKNN
then quantizes box coordinates and probabilities independently, preserving
non-zero scores while keeping the INT8 NPU path.

The decoder is `deployment/rknn/decode_yolo_output.py`; it applies confidence
filtering and per-class NMS after inference, so the model remains reusable by
the future `RKNNDetector` adapter.

## Reproduce conversion in WSL/Linux x86_64

Install a Python version supported by the selected Toolkit2 release. The verified
conversion environment was Python 3.10, `rknn-toolkit2==2.3.2`, `onnx==1.16.1`,
and `protobuf==3.20.3`.

```powershell
python deployment/rknn/export_onnx.py --weights yolo11n.pt --output yolo11n.onnx
python deployment/rknn/split_yolo_output.py --input yolo11n.onnx --output tmp/yolo11n_split_outputs.onnx
python deployment/rknn/convert_rknn.py --onnx tmp/yolo11n_split_outputs.onnx --dataset tmp/rknn_calibration.txt --output deployment/rknn/yolo11n_rk3566_int8_split.rknn --precision int8
```

The calibration manifest must contain paths valid inside the Linux conversion
environment. `fp16` remains available for diagnosis, but INT8 split-output is
the verified deployment route:

```powershell
python deployment/rknn/prepare_calibration.py --input-dir tmp/rknn_calibration --output tmp/rknn_calibration.txt --limit 80 --linux-root /mnt/c/Users/JingyuWang/adas-edge/tmp/rknn_calibration
```

Do not add models or calibration images to Git.

## Runtime compatibility

The factory LubanCat image supplied `librknnrt 1.5.0`, which rejected RKNN model
format v6. The original board library is backed up as
`/usr/lib/librknnrt.so.2023-05-30.bak`; a 1.5.2 test copy remains at
`~/adas-edge-rknn/librknnrt-1.5.2.so`; the active runtime is now Rockchip's
official 2.3.2 `librknnrt.so` in `/usr/lib`. Do not independently upgrade the
NPU kernel driver; use a board-vendor firmware update if a driver upgrade is
required.

## Board benchmark

Copy the `.rknn` model, an NHWC `.npy` uint8 input, and `benchmark_lite.py` to
the board. Install `numpy`, `psutil`, `ruamel.yaml`, and the matching
`rknn_toolkit_lite2` wheel for the board Python and architecture.

```bash
python3 benchmark_lite.py --model yolo11n_rk3566_int8_split.rknn --input-npy rknn_input_bgr_640.npy --runs 30 --warmup 3
```

The test input is a BGR OpenCV frame saved as a NumPy array. The benchmark adds
the batch dimension and feeds NHWC `uint8`; camera/video preprocessing can later
provide the same contract without changing the NPU runner.
