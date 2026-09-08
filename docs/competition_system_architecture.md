# Competition system architecture

`OV5640/HDMI IN/Ethernet → FPGA capture, buffer and preprocessing → PCIe DMA frame descriptors → RK3566 CompetitionDetector + scene adapters + ADAS state machine → PCIe metadata descriptors → FPGA OSD/mosaic → HDMI OUT`.

The software boundary is `FrameSource → CompetitionDetector → scene adapter → RuleBasedDecisionEngine → ControlTransport`. The detector never owns PCIe or display state. FPGA responsibilities and unresolved hardware gates are tracked in `competition_requirement_matrix.md`.
