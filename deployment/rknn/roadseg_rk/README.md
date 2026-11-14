# RoadSeg-RK RK3566 deployment bundle

No board test has been run yet.  These scripts are prepared for the real
LubanCat-1 / RK3566 (RKNN Lite2 2.3.2, librknnrt 2.3.2, driver 0.8.2).

## 1. Convert

WSL Ubuntu, rknn-toolkit2 2.3.2, Python 3.10:

```bash
python convert_rknn.py \
  --onnx /mnt/c/.../outputs/roadseg_rk/quantization/C/model.fp32.onnx \
  --output /mnt/c/.../outputs/roadseg_rk/rknn/roadseg_c_fp16.rknn \
  --precision fp16

python convert_rknn.py \
  --onnx /mnt/c/.../outputs/roadseg_rk/quantization/C/model.fp32.onnx \
  --output /mnt/c/.../outputs/roadseg_rk/rknn/roadseg_c_int8.rknn \
  --precision int8 \
  --dataset /mnt/c/.../outputs/roadseg_rk/quantization/C/calibration/dataset.txt \
  --optimization-level 2
```

Calibration tensors are NCHW uint8 RGB `.npy`, already letterboxed to 416x256;
the runtime input is NHWC uint8 RGB.  `mean=0`, `std=255`, no RGB2BGR.

## 2. Board pure-NPU benchmark

```bash
python3 board_benchmark.py --model roadseg_c_int8.rknn \
  --input-npy sample_416x256.npy --warmup 10 --runs 100 --json bench.json
```

Records mean/P50/P95/P99, NPU frequency, process CPU and RSS.  Use a fresh
directory under `/tmp` or a user data partition; the eMMC root was previously
full.

## 3. Board accuracy dump

`board_batch_infer.py` takes a JSON list of `{"name": ..., "input": "...npy"}`
and writes one `.npy` logits file per input.  Compare it with ONNX/PyTorch on
the PC to measure the real board INT8 parity.

## 4. Board video overlay

```bash
python3 board_video_overlay.py --model roadseg_c_int8.rknn \
  --video input.mp4 --output overlay.mp4 --width 416 --height 256 \
  --ego --json video_bench.json
```

Post-processing is composed at 416x256 and alpha-blended once at native
resolution.  `--raw` disables morphology.  This script reports decode,
preprocess, NPU, postprocess and write time separately; the pipeline FPS is
not the pure NPU FPS.
