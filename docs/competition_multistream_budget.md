# Competition multistream budget

Historical NPU-only measurement: 320 input about 43.72 FPS; historical full pipeline 320 about 27.715 FPS. These are not FPGA E2E measurements.

- 4x6 = 24 aggregate FPS: conservative target.
- 4x8 = 32 aggregate FPS: stress target.
- 4x10 = 40 aggregate FPS: planning target only, not a demonstrated rate.

The 32/40 aggregate figures exceed the historical full-pipeline number, but neither may be compared directly with isolated 43.72 FPS NPU-only service. Physical four-camera, PCIe and OSD measurements are `FPGA/PCIe HARDWARE TEST`.
