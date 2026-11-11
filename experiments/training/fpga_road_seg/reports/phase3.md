# Phase 3 evidence: offline Q4 contract

The isolated `mask_protocol.py` implements a draft envelope with frame/epoch,
capture-monotonic timestamp, Q4-native shape, ordered class names, confidence,
valid/stale flags, encoding, payload size and CRC. Binary masks are row-padded
LSB-first; probability is uint8; RLE uses alternating little-endian uint32 runs.
The receiver rejects malformed sizes/padding/CRC, future timestamps, expired
masks, invalid/stale packets and mismatched frame/timestamp/epoch/geometry.

The executable `offline_q4.py` accepts only explicit native-Q4 input or an
explicit 2x2 mosaic whose bottom-right quadrant is Q4. It checks that detections
and masks use the same source, frame, timestamp, epoch and pixel shape. Its
lower-bounding-box mask overlap is descriptive mock fusion, not a distance or
safety-control estimate.

Executed once on a local BDD validation image (1280x720), synthetic four-channel
mask, and synthetic detection box. The envelope round-tripped with frame 120,
epoch 3, timestamp age 1ms and `valid=true`; the output was labelled
`MOCK MASK + OFFLINE DETECTIONS / NO HARDWARE`. No physical device, service,
PCIe/HSST route or live FAST-Q4 code was accessed or modified. The image and
mock outputs stay in ignored local `outputs/issue15/offline-q4-demo/`.

No production FAST-Q4, PCIe, Tesla BEV, HUD, or teammate FPGA model files changed.
The Q4 quadrant, pixel transform, frame counter/epoch behavior, timestamp clock,
transport, hardware encoding and FPGA resource/timing limits all remain subject
to owner confirmation. Full transport envelope requirements are in
[`docs/fpga_road_seg_handoff.md`](../../../../docs/fpga_road_seg_handoff.md).
