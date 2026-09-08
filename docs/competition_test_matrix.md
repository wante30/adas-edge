# Competition test matrix

| Area | Test | Evidence/status |
|---|---|---|
| Input/display | HDMI, OV5640×2, Ethernet, HDMI OUT, four-way mosaic | BLOCKED BY HARDWARE |
| Preprocess | BGR/RGB/NV12 conversion, 320/416 letterbox | PASS: pytest |
| PCIe | descriptor round-trip, malformed packet rejection, stream routing | PASS: software mock; supplied Host BAR0 snapshot ABI documented; physical timeout/stale-frame rejection BLOCKED BY HARDWARE |
| AI | RKNN load/inference/decode 320/416 | PASS: RK3566 single-image smoke |
| Scene | person/vehicle/traffic-light detection and HSV state adapter | PASS: unit tests; no road-video acceptance set |
| Decision | red/green/yellow/unknown, pedestrian, obstacle, dashed/solid lane | PASS: deterministic unit tests; live inputs pending |
| OSD/return | metadata to FPGA and HDMI overlay | BLOCKED BY HARDWARE |
| Scenarios | intersection, traffic light, pedestrian, lane, night, navigation, obstacle | BLOCKED: clips/FPGA integration pending |
| Performance | NPU latency, CPU/NPU load, multistream replay | PASS for historical RK3566 isolated/replay metrics; PCIe E2E BLOCKED |
