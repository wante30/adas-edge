# B05N B Model Validation, 2026-10-29

## Selection

The competition model selection is **B: B05N P2, ten classes, Raw-head INT8, single model**. The RKNN file on the RK3566 matches SHA256 `65089ef0697bd7378071290baa218905cdcaed768452678da440b8eb116b2a5f`.

Code and the Q4 replay correction are on PR #3, branch `feature/lvt1-ego-bev-20261028`, latest tested source commit `280abfe`.

## RK3566 Local-Video Replay

This was a model and bridge test on the real RK3566, not a live PCIe test. The source was `phase10/videos/ch0_front.mp4` (1280x720, 25 FPS, 21 seconds), sampled to 210 local images at 10 FPS. The bridge packed selected images into quadrant Q4 (zero-based index 3), the same quadrant configured for the current front-road feed.

The B05N model ran through the bridge request, scheduler, raw-head decoder, detection mapping, hazard pipeline, and MVA2 packet publication path. Qt display and live PCIe capture were excluded. Run parameters were 5 warm-up requests followed by 60 requests at 10 Hz.

Per-frame timing and class-count output: [b05n_B_Q4_frontvideo_replay_20261029.jsonl](evidence/b05n_B_Q4_frontvideo_replay_20261029.jsonl).

| Measurement | Result |
|---|---:|
| Requests received | 65 |
| Frames analyzed | 55 total; 51 measured after warm-up |
| Requests replaced by latest-only scheduling | 10 |
| Measured result rate | 8.50 FPS |
| Main model total, p50 / p95 | 67.91 / 201.34 ms |
| RKNN call, p50 / p95 | 49.87 / 176.01 ms |
| Decode and NMS, p50 / p95 | 10.70 / 38.00 ms |
| Bridge pipeline, p50 / p95 | 69.86 / 207.45 ms |
| Hazard postprocess, p50 / p95 | 33.28 / 86.47 ms |
| Process CPU | 87.4% of one core; 21.9% of four-core machine |
| NPU load | 100% at 600 MHz during sampled run |
| Inference errors | 0 |

Across the 51 measured frames, detector outputs contained 1,359 car, 2 person, 3 truck, and 1 bus detections. These are per-frame detections, not unique tracked objects. Per-frame detector counts ranged from 23 to 39. The MVA2 packet is capped at 32 boxes; 11 frames reached that cap, so some detections may have been omitted from published packets.

No hazard event was emitted in these 51 frames. The clip has no frame-level event labels, so this is an observation, not a verified true-negative result. It contains no validated pedestrian crossing, front-collision, or cut-in positive. Synthetic trajectory tests cover positive and negative logic cases; they do not replace video validation.

## Live PCIe And HMI Status

The live PCIe capture service was not healthy during this work: `/dev/shm/pango-video/status.json` reported `capture_io` with zero frames, and `lspci` returned the FPGA endpoint as `0755:0755 (rev ff)`. The MFR1 snapshot timestamp remained unchanged. Therefore no current live-PCIe B result, live hazard event, Qt BEV highlight, or HMI screenshot was verified.

After each isolated B run, `adas-ai-bridge-live.service` was restored and confirmed active with its existing A model command. The board's production model was not switched during the PCIe fault.

## Verification And Next Test

The focused suite passed: 48 tests covering hazard trajectories, raw-head decoding, Q4 replay placement, MVA2 hazard details, and per-frame class-count logging. The bridge Python files compiled successfully.

Once PCIe capture is healthy, repeat B on the live Q4 feed with Qt connected. Verify class counts and target IDs in the MVA2 packets, confirm BEV target highlighting and hazard-card details, and run labeled positive and negative clips before making claims about event accuracy. The current local-video replay does not establish Qt HMI FPS or live PCIe behavior.
