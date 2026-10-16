# Competition PCIe protocol (software freeze)

This is a host-side metadata contract pending FPGA endpoint confirmation; it is not the BAR0 capture ABI used by the supplied Host tool. The supplied Host reference is authoritative for bring-up: PCI ID `0755:0755`, BAR0 >= 8 KiB, little-endian registers, 1280x720 RGB565, 1,843,200-byte frozen-frame PIO snapshot, 128-byte blocks, and commands `1=采集`, `2=读取块`, `3=整帧 DMA`. Its documented mode is host-triggered frozen-frame capture and does not promise continuous 30 FPS. The Python mock below models frame/result metadata above that capture boundary.

## FPGA → RK3566 frame descriptor

The exact existing Python layout is `struct.Struct("<4sHHIQQII BBH III")`: a **52-byte**, little-endian packed header without implicit alignment, followed immediately by the payload. The table corrects earlier documentation; it does not change the code or wire ABI.

| Byte offset | Bytes | Type | Field / current encoder value |
|---|---|---|---|
| 0 | 4 | bytes | magic = `ADSF` |
| 4 | 2 | u16 | version = 1 |
| 6 | 2 | u16 | header_len = 52 |
| 8 | 4 | u32 | stream_id |
| 12 | 8 | u64 | frame_id |
| 20 | 8 | u64 | timestamp_ns |
| 28 | 4 | u32 | width |
| 32 | 4 | u32 | height |
| 36 | 1 | u8 | pixel_format (`BGR888=1`, `RGB888=2`, `NV12=3`) |
| 37 | 1 | u8 | reserved = 0 |
| 38 | 2 | u16 | stride |
| 40 | 4 | u32 | payload_size |
| 44 | 4 | u32 | flags |
| 48 | 4 | u32 | CRC placeholder = 0; not calculated or checked |
| 52 | payload_size | bytes | payload |

The current encoder does **not** append a CRC after the payload. The decoder checks magic, version, header length and payload byte count; it does not verify the reserved/CRC placeholder, pixel-format semantics, stride or geometry-to-payload consistency. Actual DMA buffer ownership, publication ordering and format validation remain driver/consumer responsibilities and are not implemented by this packed descriptor.

## RK3566 → FPGA result descriptor

Header: `magic[4]="ADSR"`, `version u16`, `header_len u16`, `stream_id u32`, `frame_id u64`, `timestamp_ns u64`, `num_objects u16`, `traffic_light_state u8` (`unknown/red/yellow/green`), `lane_state u8` (`unknown/solid/dashed`), `adas_action u8` (`stop/forward/left/right`), `warning_flags u32`, `payload_size u32`. Each detection is `class_id u16`, `confidence_q15 u16`, `x1/y1/x2/y2 u16` in source-pixel coordinates.

The current result header is `struct.Struct("<4sHHIQQHBBBBII")`, **42 bytes**, packed little-endian. Its reserved u8 after `adas_action` is written as zero. Fixed offsets are: magic 0, version 4, header_len 6, stream_id 8, frame_id 12, timestamp_ns 20, num_objects 28, traffic_light_state 30, lane_state 31, adas_action 32, reserved 33, warning_flags 34 and payload_size 38. Detection records begin at 42 and are 12 bytes each. ADSR has no CRC. The names above describe intended semantics; this early codec does not enforce an action/lane/light enum or a valid confidence probability range.

Results are metadata intended for an FPGA consumer retaining the corresponding original frame for OSD. Matching stream/frame IDs, rejecting stale results, clearing permission on timeout and handling session reset are **required consumer behavior, not proven features of this early mock**. `competition_io/mock_transport.py` only queues serialized frames/results; it does not enforce TTL, validate an expected source frame, map a clock domain or perform hardware I/O. Neither a round-trip test nor this layout document proves a physical PCIe/FPGA implementation.

The later HUD v1 and ADST v1 are separate interfaces and cannot be appended to ADSR or written to an assumed register address. See [the 2026-10-15 signal audit](competition_rk_fpga_signal_audit_20261015.md) and [the HUD/ADST handoff](pcie_hud_pc_handoff.md) for their scope and outstanding integration work.
