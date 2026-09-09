# Competition system architecture

```text
FPGA frame descriptor -> validate v1 header/payload -> payload_to_bgr
-> CompetitionDetector -> scene adapters -> CompetitionPolicy
-> ResultDescriptor metadata
```

`CompetitionPipeline.process_descriptor()` preserves `stream_id`, `frame_id`, and `timestamp_ns`, and emits clipped source-pixel unsigned-16-bit boxes. `RoundRobinTransport` is the software-only multi-stream boundary.

Evidence labels are strict: `SOFTWARE UNIT TEST`, `SOFTWARE E2E MOCK`, `RK3566 BOARD TEST`, and `FPGA/PCIe HARDWARE TEST` are never interchangeable. Real FPGA capture, endpoint/DMA, OSD/HDMI, and physical cameras remain outside this checkout.
