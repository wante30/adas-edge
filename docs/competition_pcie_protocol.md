# Competition PCIe protocol (software freeze)

This is a host-side metadata contract pending FPGA endpoint confirmation; it is not the BAR0 capture ABI used by the supplied Host tool. The supplied Host reference is authoritative for bring-up: PCI ID `0755:0755`, BAR0 >= 8 KiB, little-endian registers, 1280x720 RGB565, 1,843,200-byte frozen-frame PIO snapshot, 128-byte blocks, and commands `1=采集`, `2=读取块`, `3=整帧 DMA`. Its documented mode is host-triggered frozen-frame capture and does not promise continuous 30 FPS. The Python mock below models frame/result metadata above that capture boundary.

## FPGA → RK3566 frame descriptor

Little-endian packed header: `magic[4]="ADSF"`, `version u16`, `header_len u16`, `stream_id u32`, `frame_id u64`, `timestamp_ns u64`, `width u16`, `height u16`, `pixel_format u8` (`BGR888=1`, `RGB888=2`, `NV12=3`), `stride u32`, `payload_size u32`, `flags u32`, followed by payload and optional CRC32. DMA rings own buffers; producer publishes descriptor only after payload is complete.

## RK3566 → FPGA result descriptor

Header: `magic[4]="ADSR"`, `version u16`, `header_len u16`, `stream_id u32`, `frame_id u64`, `timestamp_ns u64`, `num_objects u16`, `traffic_light_state u8` (`unknown/red/yellow/green`), `lane_state u8` (`unknown/solid/dashed`), `adas_action u8` (`stop/forward/left/right`), `warning_flags u32`, `payload_size u32`. Each detection is `class_id u16`, `confidence_q15 u16`, `x1/y1/x2/y2 u16` in source-pixel coordinates.

Results are metadata only; FPGA keeps the original frame for OSD. Unknown or stale results fail safe to no permission indication and may raise an alarm. A result must echo stream/frame IDs; host rejects mismatches and applies a timeout.
