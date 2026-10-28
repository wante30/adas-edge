"""Verify the generated AVM FPGA handoff archive without MATLAB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


BEV_W, BEV_H = 480, 640
INPUT_BYTES = 640 * 320 * 2
OUTPUT_BYTES = BEV_W * BEV_H * 2
LUT_BYTES = BEV_W * BEV_H * 12
GOLDEN = (0, 24, 49, 74, 99)


def check_variant(root: Path, variant: str) -> None:
    folder = root / variant
    params = json.loads((folder / "parameters.json").read_text(encoding="utf-8"))
    assert params["lut_entries"] == BEV_W * BEV_H
    assert params["lut_entry_bytes"] == 12
    assert params["pixel_order"] == "row-major"
    assert params["binary_endianness"] == "little"
    assert params["channel_count"] == (4 if variant == "4ch" else 6)

    lut = folder / "lut" / "lut_96bit_le.bin"
    hex_path = folder / "lut" / "lut_readmemh96.hex"
    data = lut.read_bytes()
    lines = hex_path.read_text(encoding="utf-8").splitlines()
    assert len(data) == LUT_BYTES, (variant, len(data))
    assert len(lines) == BEV_W * BEV_H, (variant, len(lines))
    for i, line in enumerate(lines):
        assert len(line) == 24 and int(line, 16) == int.from_bytes(
            data[i * 12 : (i + 1) * 12], "little"
        ), (variant, i)

    slots = params["channel_count"]
    for frame in GOLDEN:
        fdir = folder / "golden" / f"frame_{frame:03d}"
        assert (fdir / "expected_rgb565.bin").stat().st_size == OUTPUT_BYTES
        assert len((fdir / "expected_rgb565.hex").read_text(encoding="utf-8").splitlines()) == BEV_W * BEV_H
        for slot in range(slots):
            candidates = list((fdir / "inputs").glob(f"slot_{slot}_*.bin"))
            assert len(candidates) == 1, (variant, frame, slot)
            assert candidates[0].stat().st_size == INPUT_BYTES

    rows = (folder / "golden" / "source_manifest.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + len(GOLDEN) * slots
    print(f"{variant}: PASS ({len(data)} LUT bytes, {len(GOLDEN)} golden frames)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    check_variant(args.root, "4ch")
    check_variant(args.root, "6ch")
    print("AVM FPGA handoff verification PASS")


if __name__ == "__main__":
    main()
