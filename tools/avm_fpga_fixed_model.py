"""Fixed-point FPGA data-path model for the frozen AVM handoff.

This is intentionally a data-path/cycle-order model, not a replacement for a
vendor RTL simulator.  It consumes the frozen 96-bit LUT and RGB565 vectors,
implements the Q0.4 bilinear interpolation and Q0.6 camera blend contract,
then compares the packed RGB565 stream with the golden output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


IN_W = 640
IN_H = 320
OUT_W = 480
OUT_H = 640
COORD_F = 4
ALPHA_F = 6
GOLDEN = (0, 24, 49, 74, 99)


def _field(raw: np.ndarray, offset: int, width: int) -> np.ndarray:
    """Extract a little-endian bit field from packed 96-bit entries."""
    first = offset // 8
    count = (offset + width + 7) // 8 - first
    value = np.zeros(raw.shape[0], dtype=np.uint64)
    for i in range(count):
        value |= raw[:, first + i].astype(np.uint64) << (8 * i)
    return ((value >> (offset % 8)) & ((1 << width) - 1)).astype(np.uint32)


def read_lut(path: Path, out_w: int = OUT_W, out_h: int = OUT_H) -> dict[str, np.ndarray]:
    raw = np.fromfile(path, dtype=np.uint8)
    expected = out_w * out_h * 12
    if raw.size != expected:
        raise ValueError(f"{path}: {raw.size} bytes, expected {expected}")
    raw = raw.reshape(-1, 12)
    fields = {
        "addr_a": (0, 18),
        "addr_b": (18, 18),
        "fx_a": (36, 8),
        "fy_a": (44, 8),
        "fx_b": (52, 8),
        "fy_b": (60, 8),
        "alpha_q": (68, 11),
        "cam_a": (79, 3),
        "cam_b": (82, 3),
        "valid": (85, 1),
        "blend": (86, 1),
    }
    return {name: _field(raw, offset, width) for name, (offset, width) in fields.items()}


def read_rgb565(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype="<u2")
    expected = IN_W * IN_H
    if data.size != expected:
        raise ValueError(f"{path}: {data.size} pixels, expected {expected}")
    return data.reshape(IN_H, IN_W)


def read_output_rgb565(path: Path, out_w: int = OUT_W, out_h: int = OUT_H) -> np.ndarray:
    data = np.fromfile(path, dtype="<u2")
    expected = out_w * out_h
    if data.size != expected:
        raise ValueError(f"{path}: {data.size} pixels, expected {expected}")
    return data.reshape(out_h, out_w)


def rgb888_planes(word: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # The frozen golden export uses avm_render(..., 'rgb565', ...): unpack
    # RGB565 and expand to 8-bit before interpolation, then repack to RGB565.
    r5 = (word >> 11) & 0x1F
    g6 = (word >> 5) & 0x3F
    b5 = word & 0x1F
    return (
        ((r5 << 3) | (r5 >> 2)).astype(np.uint32),
        ((g6 << 2) | (g6 >> 4)).astype(np.uint32),
        ((b5 << 3) | (b5 >> 2)).astype(np.uint32),
    )


def sample_plane(
    planes: list[np.ndarray],
    cam: np.ndarray,
    addr: np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """Four-neighbour Q0.4 interpolation for one color code plane."""
    out = np.zeros(addr.size, dtype=np.uint32)
    w0 = (16 - fx) * (16 - fy)
    w1 = fx * (16 - fy)
    w2 = (16 - fx) * fy
    w3 = fx * fy
    for slot, plane in enumerate(planes):
        mask = active & (cam == slot)
        if not np.any(mask):
            continue
        flat = plane.reshape(-1)
        a = addr[mask].astype(np.int64)
        weighted = (
            flat[a]
            * w0[mask]
            + flat[a + 1]
            * w1[mask]
            + flat[a + IN_W]
            * w2[mask]
            + flat[a + IN_W + 1]
            * w3[mask]
        )
        # Q0.4 in both axes: round and drop eight fractional bits.
        out[mask] = (weighted + 128) >> 8
    return out


def render(lut: dict[str, np.ndarray], frames: list[np.ndarray], out_w: int = OUT_W, out_h: int = OUT_H) -> np.ndarray:
    planes = [rgb888_planes(frame) for frame in frames]
    n = out_w * out_h
    valid = lut["valid"].astype(bool)
    blend = lut["blend"].astype(bool)
    output = np.zeros(n, dtype=np.uint16)
    channels = []
    for channel_index in range(3):
        a = sample_plane(
            [p[channel_index] for p in planes],
            lut["cam_a"],
            lut["addr_a"],
            lut["fx_a"],
            lut["fy_a"],
            valid,
        )
        b = sample_plane(
            [p[channel_index] for p in planes],
            lut["cam_b"],
            lut["addr_b"],
            lut["fx_b"],
            lut["fy_b"],
            blend,
        )
        # Q0.6 alpha blend, alpha_q=64 means all camera A.
        mixed = (lut["alpha_q"] * a + (64 - lut["alpha_q"]) * b + 32) >> 6
        channels.append(np.minimum(mixed, 255).astype(np.uint16))
    # Same truncating pack565 operation as the MATLAB reference.
    output = ((channels[0] >> 3) << 11) | ((channels[1] >> 2) << 5) | (channels[2] >> 3)
    output[~valid] = 0
    return output.reshape(out_h, out_w)


def find_input(folder: Path, slot: int) -> Path:
    matches = sorted((folder / "inputs").glob(f"slot_{slot}_*.bin"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one input for slot {slot}, found {len(matches)}")
    return matches[0]


def run_variant(root: Path, variant: str, frames_to_run: tuple[int, ...]) -> None:
    candidate = root / variant
    folder = candidate if (candidate / "parameters.json").is_file() else root
    params = __import__("json").loads((folder / "parameters.json").read_text(encoding="utf-8"))
    slots = int(params["channel_count"])
    out_w = int(params["output_width"])
    out_h = int(params["output_height"])
    lut = read_lut(folder / "lut" / "lut_96bit_le.bin", out_w, out_h)
    for frame_id in frames_to_run:
        golden = folder / "golden" / f"frame_{frame_id:03d}"
        frames = [read_rgb565(find_input(golden, slot)) for slot in range(slots)]
        actual = render(lut, frames, out_w, out_h)
        expected = read_output_rgb565(golden / "expected_rgb565.bin", out_w, out_h)
        if not np.array_equal(actual, expected):
            diff = np.flatnonzero(actual.reshape(-1) != expected.reshape(-1))
            first = int(diff[0])
            y, x = divmod(first, out_w)
            raise AssertionError(
                f"{variant} frame {frame_id}: mismatch at ({x},{y}), "
                f"actual=0x{int(actual[y, x]):04x}, expected=0x{int(expected[y, x]):04x}, "
                f"mismatches={diff.size}"
            )
        print(f"{variant} frame_{frame_id:03d}: PASS ({out_w}x{out_h}, RGB565)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="avm_fpga_handoff root")
    parser.add_argument("--variant", choices=("4ch", "6ch", "both"), default="both")
    parser.add_argument("--frame", type=int, action="append", dest="frames")
    args = parser.parse_args()
    selected = tuple(args.frames) if args.frames else GOLDEN
    variants = ("4ch", "6ch") if args.variant == "both" else (args.variant,)
    for variant in variants:
        run_variant(args.root, variant, selected)
    print("AVM fixed-point FPGA data-path simulation PASS")


if __name__ == "__main__":
    main()
