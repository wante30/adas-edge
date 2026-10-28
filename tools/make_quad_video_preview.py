"""Make a visual preview of the four 640x320 FPGA video tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


W, H = 640, 320
OUT_W, OUT_H, TOP_BAR = 1280, 720, 40


def rgb565_to_rgb888(words: np.ndarray) -> np.ndarray:
    r = (words >> 11) & 0x1F
    g = (words >> 5) & 0x3F
    b = words & 0x1F
    return np.stack(((r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)), axis=-1).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("handoff_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()
    folder = args.handoff_root / "4ch" / "golden" / f"frame_{args.frame:03d}" / "inputs"
    names = sorted(folder.glob("slot_*_*.bin"))
    if len(names) != 4:
        raise SystemExit(f"expected four input tiles, found {len(names)} in {folder}")
    canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
    for index, path in enumerate(names):
        words = np.fromfile(path, dtype="<u2")
        if words.size != W * H:
            raise SystemExit(f"{path}: expected {W*H} pixels, found {words.size}")
        tile = rgb565_to_rgb888(words.reshape(H, W))
        row, col = divmod(index, 2)
        y = TOP_BAR + row * H
        x = col * W
        canvas[y : y + H, x : x + W] = tile
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGB").save(args.output)
    print(json.dumps({"output": str(args.output), "layout": "TL/TR/BL/BR", "frame": args.frame}, ensure_ascii=False))


if __name__ == "__main__":
    main()
