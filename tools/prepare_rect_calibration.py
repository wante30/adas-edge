"""Build RKNN INT8 calibration samples that match the deployed preprocessing exactly.

Why this exists: the previous INT8 build calibrated on 25 raw COCO128 JPEGs
(``tmp/rk3566_profile/calibration_wsl.txt``). Those frames are neither road-domain
nor the deployed canvas, so the toolkit resized them itself and the quantizer
never saw the grey letterbox padding that every inference frame carries. The
scale factors it produced therefore described a different input distribution than
the one the model runs on.

What this tool produces instead, for each sampled frame:

* the frame letterboxed to the **exact static canvas** by the same
  ``letterbox_bgr`` the deployment path uses, so the padding is in the data;
* a ``.npy`` array holding that canvas in **RGB, uint8, NCHW** order. RKNN's ``.npy``
  loader requires the model's input layout and rejects NHWC outright
  (``expect 'nchw' like (1, 3, 192, 320)``), and an image path would have gone
  through ``cv2.imread`` (BGR) instead - so writing the array ourselves is what
  makes both the layout and the channel order explicit and identical to inference;
* a PNG copy of the same canvas for visual inspection.

The frozen holdout is refused as a source: it may not be used for calibration
sample selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # allow `python tools/prepare_rect_calibration.py` from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from competition_ai.input_shape import validate_model_shape
from tools.profile_rknn_single_stream import letterbox_bgr

DEFAULT_FROZEN_HOLDOUT = Path("data/competition_eval/bdd100k_validation/bdd100k")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_source_frames(source: Path, stride: int):
    """Yield ``(identifier, frame)`` pairs from a video file or an image directory.

    The identifier is the image file name for a directory source and
    ``frame_<index>`` for a video, so every downstream artifact (calibration
    canvas, detection dump) can be keyed on the same string and compared across
    backends.
    """
    if source.is_dir():
        paths = sorted(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        for index, path in enumerate(paths):
            if index % stride:
                continue
            import cv2

            frame = cv2.imread(str(path))
            if frame is not None:
                yield path.name, frame
        return
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video {source}")
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                yield f"frame_{index:06d}", frame
            index += 1
    finally:
        capture.release()


def guard_source(source: Path, frozen_holdout: Path) -> None:
    """Refuse a calibration source that lives inside the frozen holdout."""
    resolved = source.resolve()
    frozen = frozen_holdout.resolve()
    if resolved == frozen or frozen in resolved.parents:
        raise SystemExit(
            f"refusing {source}: the frozen BDD100K holdout may not be used for calibration selection"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, action="append", required=True, help="road video or image directory; repeatable")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--stride", type=int, default=30, help="take every Nth frame from a video")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="text list of .npy paths for convert_rknn.py --dataset")
    parser.add_argument("--linux-root", help="prefix written into the manifest (WSL mount path)")
    parser.add_argument("--frozen-holdout", type=Path, default=DEFAULT_FROZEN_HOLDOUT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    try:
        width, height = validate_model_shape((args.width, args.height))
    except ValueError as exc:
        parser.error(str(exc))
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.stride < 1:
        parser.error("--stride must be positive")

    for source in args.source:
        if not source.exists():
            parser.error(f"source does not exist: {source}")
        guard_source(source, args.frozen_holdout)

    import cv2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[Path, Path]] = []
    per_source: list[dict] = []
    for source in args.source:
        produced = 0
        for _identifier, frame in iter_source_frames(source, args.stride):
            if len(entries) >= args.limit:
                break
            canvas, scale, pad_x, pad_y = letterbox_bgr(frame, (width, height))
            if canvas.shape != (height, width, 3):
                raise RuntimeError(f"letterbox produced {canvas.shape}, expected {(height, width, 3)}")
            index = len(entries)
            npy_path = args.output_dir / f"calib_{index:04d}.npy"
            png_path = args.output_dir / f"calib_{index:04d}.png"
            np.save(npy_path, np.ascontiguousarray(canvas.transpose(2, 0, 1)[None, ...]))
            cv2.imwrite(str(png_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            entries.append((npy_path, png_path))
            produced += 1
        per_source.append({"source": str(source), "frames": produced})
        if len(entries) >= args.limit:
            break

    if not entries:
        raise SystemExit("no calibration frames were produced")

    root = args.linux_root.rstrip("/") if args.linux_root else None
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{root}/{path.name}" if root else str(path.resolve())
        for path, _ in entries
    ]
    args.manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    first = np.load(entries[0][0])
    report = {
        "canvas": f"{width}x{height}",
        "canvas_pixels": width * height,
        "channel_order": "RGB",
        "dtype": str(first.dtype),
        "layout": "NCHW",
        "array_shape": list(first.shape),
        "expected_by_toolkit": f"(1, 3, {height}, {width})",
        "samples": len(entries),
        "stride": args.stride,
        "sources": per_source,
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "frozen_holdout": str(args.frozen_holdout),
        "frozen_holdout_used": False,
    }
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {len(entries)} calibration samples to {args.output_dir}")


if __name__ == "__main__":
    main()
