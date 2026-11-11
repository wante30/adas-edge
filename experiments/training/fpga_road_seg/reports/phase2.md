# Phase 2 evidence: software model and quantization

Evidence collected on RTX 4060 Laptop, Python 3.10, PyTorch 2.5.1+cu121,
ONNX 1.17.0 and ONNX Runtime 1.23.2. Outputs and weights are local ignored
artifacts; no model weights or dataset are committed.

## TwinLiteNet+ reference

The official Small checkpoint (SHA-256
`0dd3bd7356a5b70e1d21ae60cd09b55641b48bde738393177f94daff17ab6360`) loaded
strictly and produced two original two-class heads on six upstream demo images
at 640x384. Small has 121,552 parameters. The corresponding FP32 ONNX output
maximum absolute logit difference from PyTorch was 0.0001374. The official Nano
checkpoint also loaded strictly; maximum difference was 0.0008965. **These are
runtime/export parity checks, not accuracy measurements**; the demo set has no
ground truth. Official checkpoint files remain outside Git.

At 320x192 the measured software profile is:

| Model | Parameters | MACs | FLOPs (2 x MAC) |
|---|---:|---:|---:|
| TwinLiteNet+ Small, original two heads | 121,552 | 323,968,256 | 647,936,512 |
| TwinLiteNet+ Nano, original two heads | 33,379 | 123,010,048 | 246,020,096 |
| TwinLiteNet+ Small, four-logit software extension | 117,716 | 290,329,856 | 580,659,712 |
| RoadCandidate four-logit | 8,508 | 25,351,680 | 50,703,360 |

Counter includes Conv/Depthwise/Pointwise/ConvTranspose/Linear/MatMul and excludes
normalization, activations, pooling, Add, Resize and memory traffic. It is not a
latency or FPGA utilization estimate. Small's software operator family includes
Conv, PReLU, dilation up to 16, ConvTranspose, CAAM MatMul/Linear/Softmax,
AveragePool, Add, Concat, reshape and transpose. This is a substantial set of
unconfirmed accelerator requirements. CAAM shape probes fail at 384x216; tested
valid sizes are 320x192, 512x288 and 640x384.

The 8,508-parameter candidate uses Conv/Depthwise/Pointwise, BatchNorm folded by
export, ReLU, Add and nearest-neighbor Resize. At 320x192 its FP32 ONNX graph
contains 14 Conv, 11 Relu, 2 Add, 3 Resize, 3 Shape, 12 Constant, 3 Slice and
3 Concat nodes. No operator support is assumed from this inventory.

## FP32 and INT8 export smoke

The fixed-shape candidate export produced FP32 ONNX and ONNX Runtime static QDQ
from 16 training-only calibration fixtures. All 14 Conv weights are signed int8,
per-output-channel; Conv activations are signed int8 and bias metadata is int32.
Add/Resize and graph I/O remain float32. The artifact is not fully integer and is
not a PG2L200H weight image. Per-layer inventories, scales, zero points, bias
scale contract, and an illustrative Q30 multiplier/right shift are emitted by
`export.py`; exact rounding/saturation and RTL encoding await hardware review.

Measured on four **synthetic validation fixtures**:

| Check | Result |
|---|---:|
| PyTorch vs FP32 ONNX maximum logit error | 3.73e-7 |
| FP32 vs QDQ maximum logit error | 0.05796 |
| Mean probability absolute error | 0.001441 |
| Threshold agreement over valid synthetic pixels | 97.93% |

The synthetic checkpoint failed to learn solid lane and crosswalk positives in
the short smoke run. These values establish only that serialization, execution
and comparison code runs; they do not measure road quality or establish PTQ
acceptance. Repeat calibration and compare against a real grouped held-out road
set before selecting a quantized model.

## 4060 throughput estimate

Resident synthetic batch benchmark, batch 8, 10 warmups and 50 timed steps:

| Model/input | Images/s | Peak allocated | 70k x 50 epoch arithmetic estimate |
|---|---:|---:|---:|
| Candidate 320x192 | 1,025.8 | 145 MiB | 0.95 h |
| Small 320x192 | 254.1 | 343 MiB | 3.83 h |
| Candidate 512x288 | 520.2 | 366 MiB | 1.87 h |
| Small 512x288 | 153.0 | 789 MiB | 6.35 h |

This benchmark excludes image decode, storage, augmentation, validation,
checkpointing and data transfer, so estimates are optimistic lower bounds. The
full train CLI on only 16 synthetic samples ran at 31.3 images/s in its first
epoch (startup dominated), then the candidate's 12-epoch average was about 139
images/s; TwinLiteNet+ Small's second-epoch rate was 85.8 images/s. Blindly
extrapolating those tiny-fixture rates yields about 7.0h for Candidate and 11.3h
for Small at 320x192, but the sample is far too small to model a 70k-image data
loader. These observations bracket GPU-only performance, not a promised runtime.
Measure the recovered real BDD loader and validation cost before choosing a 4090;
the local 4060 is currently sufficient for development and an overnight run.
