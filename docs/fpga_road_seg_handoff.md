# PG2L200H road-mask handoff

Issue #15 software handoff. This document is a candidate interface for FPGA
review; it is not an approved hardware contract or a deployment claim.

## Status

| Item | Status |
|---|---|
| Python train/eval/infer and synthetic smoke | **已运行通过** |
| TwinLiteNet+ Small/Nano strict checkpoint load and ONNX parity | **已运行通过**, demo images only; no ground-truth accuracy evaluation |
| Candidate FP32 ONNX and Conv-only static QDQ export | **已运行通过** on synthetic fixture; real-data PTQ acceptance pending |
| Four-class real training and held-out metrics | **需要真实数据/GPU**; historical BDD source is on AutoDL and current access is unconfirmed |
| PG2L200H operator/clock/resource/timing fit | **等待 FPGA 队友确认** |
| FPGA bitstream, RTL weight image, board latency/FPS | **板端实测: none** |
| Existing RK3566 FAST-Q4, PCIe, HUD, Tesla BEV integration | No production path changed; this draft has no live transport |

## Candidate model I/O

Software candidate input is fixed-shape NCHW `float32 [1,3,192,320]`, RGB,
values in `[0,1]` (`uint8 / 255`), letterboxed with fill 114. Output is raw
logits `[1,4,192,320]`, ordered `drivable_area`, `solid_lane`, `dashed_lane`,
`crosswalk`; independent sigmoid thresholds produce overlapping masks. The
candidate has 8,508 parameters and 50,703,360 multiply-accumulates at 320x192
(FLOPs reported as 2x MAC). These are software shape/profile counts, not device
resource estimates.

The TwinLiteNet+ Small reference has 121,552 parameters and 647,936,512 MACs at
320x192 under the same counter. It uses PReLU, dilated convolutions, transpose
convolution, CAAM MatMul/Linear/Softmax, pooling and reshape/transpose. The
four-head software extension adds 117,716 parameters and 580,659,712 MACs.
Reference architectures and the four-head variant are compatibility comparators,
not the default candidate. Small CAAM fails at 384x216; tested valid sizes are
320x192, 512x288 and 640x384. Ask the software team before changing dimensions.

Candidate FP32 operators observed in ONNX: Conv (including depthwise/pointwise),
Relu, Add, Resize, Shape, Constant, Slice, Concat. Static PTQ inserts QuantizeLinear
and DequantizeLinear around 14 Conv nodes; Add, Resize, graph I/O and remaining
boundaries stay float32. This is not a fully integer graph. The TwinLiteNet+
reference operators above are not asserted as PG2L200H-supported.

## Quantization handoff

PTQ uses ONNX Runtime static calibration with only the explicitly selected
training split, symmetric signed int8 activations and per-output-channel int8
weights for Conv. Each exported Conv has activation/weight/output scale and
zero-point metadata. Bias is represented by ONNX QDQ as int32 at
`input_scale * weight_scale`, zero-point 0. The export report includes a Q30
integer multiplier/right-shift approximation for the real requantization ratio.
Accumulator width, rounding, saturation, per-channel axis and exact multiplier
convention require FPGA review; the Q30 values are illustrative, not vendor RTL
parameters. Calibration manifest records sample IDs, split, group, path and image
SHA-256. Do not calibrate on validation/test or adjacent holdout frames.

No weights are committed. ONNX QDQ is not a PG2L200H memory image. RTL owner must
define binary layout, endianness, alignment, padding, bias encoding, activation
scales, accumulator behavior and loader/checksum before converting weights.

## Offline mask envelope

The Python mock draft `Q4MASK-DRAFT-1` carries `source_id=Q4`,
`coordinate_space=q4_native`, `stream_epoch`, `frame_id`,
`timestamp_ns` in capture monotonic clock, `[4,H,W]`, class order, encoding,
per-class confidence, `valid`, `stale`, payload length and CRC32. The offline
provider rejects identity/geometry mismatches, invalid/stale data, future capture
timestamps and expired masks. This is not assigned to an existing PCIe, HSST,
BAR, HUD or service register.

Supported prototype encodings are row-padded LSB-first binary bitpack,
flattened alternating zero-first little-endian uint32 run lengths, and uint8
probability. At 320x192, payload-only sizes are:

| Encoding | Bytes/frame | At 30 FPS |
|---|---:|---:|
| 4x1-bit row-padded | 30,720 | 0.922 MB/s |
| 4xuint8 probability | 245,760 | 7.373 MB/s |
| RLE uint32 | content-dependent; worst case 983,044 | worst case 29.491 MB/s |

Rates exclude headers, alignment, transport framing, CRC and PCIe transactions.
Real producer and YOLO consumer must agree on whether Q4 is a native crop or a
particular quadrant of a mosaic, pixel origin, resize/letterbox reversal, frame
counter wrap, monotonic clock source and stream reset/epoch behavior. This draft
uses Q4-native pixels and must not silently assume mosaic coordinates.

## FPGA owner checklist

Before network trimming or weight packing, provide PG2L200H part/speed grade,
remaining LUT/DSP/BRAM/URAM if present, target clock and post-route slack,
supported Conv/Depthwise/Pointwise/ReLU/Add/Resize modes, maximum tensor shape,
padding/stride/dilation limits, line-buffer and weight memory limits, INT8
rounding/saturation rules, and AXI/PCIe/HSST route and throughput/latency goals.
Confirm whether `Resize` means nearest-neighbor and specify exact coordinate mode.
No capacity or timing conclusion can be made without these reports.

Recommended integration order: validate standalone operator micro-tests against
RTL; compare each layer against ONNX reference; validate full-frame logits and
mask encoding; replay recorded frames through an offline adapter; measure
frame-aligned end-to-end latency and drop/staleness behavior; then conduct
hardware-in-loop A/B against the unchanged FAST-Q4 flow. Track IoU, crosswalk
recall, solid/dashed F1, FPS, CPU, end-to-end latency and stale-frame rate. Keep
mask use descriptive until the system owner approves downstream semantics.
