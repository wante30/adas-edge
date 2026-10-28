"""Render a continuous RGB565 fixed-LUT AVM video from A2D2 camera frames.

This is the PC-side counterpart of the frozen 640x640 RTL contract.  It uses
the same packed LUT and RGB565 interpolation model as
``tools.avm_fpga_fixed_model``; the output is therefore an AVM data-path
preview, not a board capture.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.avm_fpga_fixed_model import read_lut, render


INPUT_W = 640
INPUT_H = 320
OUTPUT_W = 640
OUTPUT_H = 640

# Dataset directory names for the camera names used by the LUT variants.
SOURCE_DIR_BY_CAMERA = {
    "front_center": "front",
    "front_right": "front_right",
    "side_left": "left",
    "side_right": "right",
    "rear_center": "back",
}


def rgb888_to_rgb565(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.uint16)
    g = rgb[..., 1].astype(np.uint16)
    b = rgb[..., 2].astype(np.uint16)
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def rgb565_to_bgr(words: np.ndarray) -> np.ndarray:
    r5 = (words >> 11) & 0x1F
    g6 = (words >> 5) & 0x3F
    b5 = words & 0x1F
    rgb = np.stack(
        ((r5 << 3) | (r5 >> 2), (g6 << 2) | (g6 >> 4), (b5 << 3) | (b5 >> 2)),
        axis=-1,
    ).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_frame(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    image = cv2.resize(image, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb888_to_rgb565(rgb)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lut-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()

    params = json.loads((args.lut_root / "parameters.json").read_text(encoding="utf-8"))
    if (params["input_width"], params["input_height"]) != (INPUT_W, INPUT_H):
        raise SystemExit("the renderer is frozen for 640x320 inputs")
    if (params["output_width"], params["output_height"]) != (OUTPUT_W, OUTPUT_H):
        raise SystemExit("the renderer is frozen for 640x640 output")
    expected_slots = tuple(params["slot_camera_names"])
    try:
        slot_dirs = tuple(SOURCE_DIR_BY_CAMERA[name] for name in expected_slots)
    except KeyError as exc:
        raise SystemExit(f"no source directory mapping for LUT camera {exc.args[0]!r}") from exc

    lut = read_lut(args.lut_root / "lut" / "lut_96bit_le.bin", OUTPUT_W, OUTPUT_H)
    frame_paths = [sorted((args.source_root / folder).glob("*.png")) for folder in slot_dirs]
    available = min(len(paths) for paths in frame_paths)
    count = min(max(1, args.frames), available)
    if count == 0:
        raise SystemExit("no PNG frames found in all four camera directories")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (OUTPUT_W, OUTPUT_H),
    )
    if not writer.isOpened():
        raise SystemExit(f"cannot create {args.output}")

    first_bgr = None
    try:
        for index in range(count):
            frames = [load_frame(paths[index]) for paths in frame_paths]
            output565 = render(lut, frames, OUTPUT_W, OUTPUT_H)
            bgr = rgb565_to_bgr(output565)
            writer.write(bgr)
            if first_bgr is None:
                first_bgr = bgr
    finally:
        writer.release()

    if args.preview and first_bgr is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview), first_bgr)

    report = {
        "output": str(args.output),
        "frames": count,
        "fps": args.fps,
        "resolution": [OUTPUT_W, OUTPUT_H],
        "lut": str(args.lut_root),
        "slot_camera_names": list(expected_slots),
        "slot_dirs": list(slot_dirs),
        "renderer": "frozen_rgb565_lut_model",
        "hardware_verified": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
