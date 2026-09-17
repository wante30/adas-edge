"""Write letterboxed NHWC uint8 canvases for on-board inference.

The board should not re-implement letterboxing. Producing the exact canvas on the
PC and shipping the bytes means the board and the PC evaluate byte-identical
inputs, and the letterbox metadata (``scale``, ``pad_x``, ``pad_y``) needed to map
boxes back to original-image coordinates travels with them.

Output per canvas: ``<stem>.npy`` holding uint8 ``(1, H, W, 3)`` RGB, plus a
manifest JSON listing every canvas with its metadata.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # allow `python tools/rect_canvas_dump.py` from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competition_ai.input_shape import validate_model_shape
from tools.prepare_rect_calibration import iter_source_frames
from tools.profile_rknn_single_stream import letterbox_bgr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="video file or image directory")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    try:
        width, height = validate_model_shape((args.width, args.height))
    except ValueError as exc:
        parser.error(str(exc))
    if args.limit < 1 or args.stride < 1:
        parser.error("--limit and --stride must be positive")
    if not args.source.exists():
        parser.error(f"source does not exist: {args.source}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for identifier, frame in iter_source_frames(args.source, args.stride):
        if len(entries) >= args.limit:
            break
        canvas, scale, pad_x, pad_y = letterbox_bgr(frame, (width, height))
        if canvas.shape != (height, width, 3):
            raise RuntimeError(f"letterbox produced {canvas.shape}, expected {(height, width, 3)}")
        index = len(entries)
        stem = f"canvas_{index:04d}"
        path = args.output_dir / f"{stem}.npy"
        # NHWC uint8 because that is what the board passes to RKNN Lite.
        np.save(path, np.ascontiguousarray(canvas[None, ...]))
        entries.append(
            {
                "index": index,
                "stem": stem,
                "image": identifier,
                "path": str(path),
                "file": path.name,
                "scale": float(scale),
                "pad_x": int(pad_x),
                "pad_y": int(pad_y),
                "source": str(args.source),
            }
        )

    if not entries:
        raise SystemExit(f"no frames were extracted from {args.source}")
    manifest = {
        "canvas": f"{width}x{height}",
        "layout": "NHWC",
        "channel_order": "RGB",
        "dtype": "uint8",
        "array_shape": [1, height, width, 3],
        "source": str(args.source),
        "stride": args.stride,
        "count": len(entries),
        "entries": entries,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest), encoding="utf-8")
    print(
        f"{manifest['canvas']}: wrote {len(entries)} canvases {manifest['array_shape']} "
        f"({manifest['layout']} {manifest['channel_order']}) -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
