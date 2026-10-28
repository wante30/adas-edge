# AVM FPGA data handoff

This archive contains the frozen data contract for RTL/FPGA co-simulation. It does **not** contain Verilog, a synthesized bitstream, board constraints, or a hardware validation result.

## Variants

- `4ch/`: local camera slots 0..3 = `front_right`, `side_left`, `side_right`, `rear_center`.
- `6ch/`: local camera slots 0..5 = `front_center`, `front_left`, `front_right`, `side_left`, `side_right`, `rear_center`.

The 4CH package remaps the global A2D2 camera IDs to dense local slots. Slot 7 is the invalid/sentinel camera ID.

## FPGA inputs and outputs

- Each camera input frame: 640×320 RGB565 little-endian, row-major, 409,600 bytes.
- BEV output frame: 480×640 RGB565 little-endian, row-major, 614,400 bytes.
- Source camera image buffers require four-neighbor access at `addr`, `addr+1`, `addr+640`, `addr+641`.
- LUT address: `y * 640 + x`, one 12-byte entry per BEV pixel, row-major.
- Frozen coordinate fraction: Q0.4; frozen blend alpha: Q0.6 (`alpha_unity=64`).

## LUT files

- `lut/lut_96bit_le.bin`: 12-byte little-endian entries for firmware/DDR loading.
- `lut/lut_readmemh96.hex`: one 96-bit hexadecimal word per BEV pixel for RTL `$readmemh`-style initialization.
- `lut/valid_u8.bin` and `lut/blend_u8.bin`: audit/debug masks.
- `lut/alpha_u16_le.bin`: audit/debug alpha values.

Each 96-bit entry contains, from low bit to high bit: `addr_a[17:0]`, `addr_b[17:0]`, `fx_a[7:0]`, `fy_a[7:0]`, `fx_b[7:0]`, `fy_b[7:0]`, `alpha_q[10:0]`, `cam_a[2:0]`, `cam_b[2:0]`, `valid`, `blend`, and reserved zero padding.

## Golden co-simulation data

Each variant has five frame directories under `golden/` for frame indices 0, 24, 49, 74, and 99:

- `inputs/slot_*_*.bin/.hex`: RGB565 camera input frames.
- `expected_rgb565.bin/.hex`: expected packed BEV output.
- `expected_rgb565.png`: human-readable expected output.
- `float_reference_rgb888.png`: floating reference for visual diagnosis.

`golden/source_manifest.csv` records source indices and URI provenance. The original four views have timestamp-alignment evidence. The added front-left/front-right views use a source-index sequence but lack timestamp sidecars, so six-camera timestamp skew remains UNKNOWN until revalidated.

## Verification

From the archive root:

```text
python verify_fpga_handoff.py
```

The verifier checks LUT byte/HEX equivalence, entry counts, input/output frame sizes, and golden-file completeness. It does not verify camera timing or FPGA hardware behavior.
