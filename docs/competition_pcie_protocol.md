# ADAS PCIe metadata protocol v1

This document freezes the software metadata ABI used between the FPGA-side application and RK3566. It is an ADAS metadata ABI, not the supplied vendor Host reference ABI.

## ABI boundary

The Host reference ABI remains a separate bring-up contract: PCI ID `0755:0755`, BAR0 at least 8 KiB, 1280x720 little-endian RGB565, 1,843,200-byte frozen frame, 128-byte blocks, and commands `1=采集`, `2=读取块`, `3=整帧 DMA`. It is not a continuous metadata stream.

## Wire rules

- Little-endian, packed, no native alignment.
- `version=1`; header lengths are fixed and included in the header.
- Zero `crc32` disables CRC. When `flags & 1`, CRC32 covers payload.
- Frame payload length is `payload_size`; pixel adapters perform format-specific stride/size checks.

| Structure | Size | Python format |
|---|---:|---|
| FrameDescriptor | 50 bytes | `<4sHHIQQHHBBIIII` |
| ResultDescriptor | 46 bytes | `<4sHHIQQHBBBBIII` |
| DetectionRecord | 12 bytes | `<HHHHHH` |

Frame offsets are `magic=0`, `version=4`, `header_len=6`, `stream_id=8`, `frame_id=12`, `timestamp_ns=20`, `width=28`, `height=30`, `pixel_format=32`, `reserved=33`, `stride=34`, `payload_size=38`, `flags=42`, `crc32=46`. Generated constants are in `docs/generated/adas_pcie_v1.h` and `.vh`.

Result fields are `magic`, `version`, `header_len`, `stream_id`, `frame_id`, `timestamp_ns`, `num_objects`, `traffic_light_state`, `lane_state`, `adas_action`, `reserved`, `warning_flags`, `payload_size`, and `crc32` in that order. Payload is `num_objects` consecutive DetectionRecords.

## Pixel format enum

`UNKNOWN=0`, `BGR888=1`, `RGB888=2`, `NV12=3`, `RGB565=4`. RGB565 is little-endian `R5:G6:B5`; stride is bytes per row and payload is `stride*height` (NV12 uses `stride*height*3/2`).

## Semantic enums

`TrafficLightState`: `UNKNOWN=0`, `RED=1`, `YELLOW=2`, `GREEN=3`.
`LaneState`: `UNKNOWN=0`, `SOLID=1`, `DASHED=2`.
`AdasAction`: `STOP=0`, `FORWARD=1`, `LEFT=2`, `RIGHT=3`.

Warning bits: `RED_LIGHT=0x001`, `YELLOW_LIGHT=0x002`, `TRAFFIC_LIGHT_UNKNOWN=0x004`, `PEDESTRIAN_DANGER=0x008`, `COLLISION_WARNING=0x010`, `SOLID_LANE_CHANGE_FORBIDDEN=0x020`, `LANE_CHANGE_UNSAFE=0x040`, `OVERSPEED=0x080`, `TURN_SIGNAL_REQUIRED=0x100`, `NAVIGATION_CONFLICT=0x200`.

Python-only names `GO`, `BRAKE`, and `WARNING` are not wire actions: `GO -> FORWARD`; `BRAKE -> STOP + warning`; `WARNING -> FORWARD/STOP + warning` according to policy.
