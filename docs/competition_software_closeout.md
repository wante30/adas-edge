# Competition software closeout

## 已完成

- Canonical `adas_edge.competition` package with protocol, pixel adapters, detector injection, policy, temporal traffic-light stabilizer, pipeline and multi-stream transport.
- Frozen PCIe metadata v1 with generated C/Verilog headers and golden vectors.
- Deterministic bbox clipping and source-coordinate result encoding.
- Scenario-aware replay interface with annotated video, detection CSV, ADAS log, and summary JSON outputs.

## 软件 E2E

The descriptor -> pixel adapter -> injected detector -> policy -> result descriptor path is validated as `SOFTWARE E2E MOCK`. It is not physical PCIe.

## 仍未完成

- real FPGA video capture
- FPGA bitstream
- physical PCIe endpoint / DMA
- FPGA OSD / HDMI output
- four physical camera/video inputs
- real RK3566 continuous road-video acceptance
- approved road-video acceptance set
- final model fine-tuning decision
- final 1080P demo
- final competition package
