"""Build a small evaluation bundle (letterboxed .npy + originals + manifest)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from common import INPUT_SIZE, letterbox_rgb, write_json


def add_image(path: Path, name: str, source: str, out_dir: Path, rows: list):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb, geometry = letterbox_rgb(bgr, size=INPUT_SIZE)
    input_path = out_dir / "inputs" / (name + ".npy")
    original_path = out_dir / "originals" / (name + ".jpg")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    original_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(input_path, rgb[None, ...].astype(np.uint8))
    cv2.imwrite(str(original_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    rows.append(
        {
            "name": name,
            "source": source,
            "input": str(input_path.resolve()),
            "original": str(original_path.resolve()),
            "geometry": geometry,
        }
    )


def add_video(path: Path, count: int, out_dir: Path, rows: list):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("cannot open video: %s" % path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError("cannot determine frame count: %s" % path)
    indices = np.linspace(0, total - 1, count + 2, dtype=np.int64)[1:-1]
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        name = "%s_f%05d" % (path.stem, int(idx))
        rgb, geometry = letterbox_rgb(frame, size=INPUT_SIZE)
        input_path = out_dir / "inputs" / (name + ".npy")
        original_path = out_dir / "originals" / (name + ".jpg")
        input_path.parent.mkdir(parents=True, exist_ok=True)
        original_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(input_path, rgb[None, ...].astype(np.uint8))
        cv2.imwrite(str(original_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append(
            {
                "name": name,
                "source": str(path),
                "frame_index": int(idx),
                "input": str(input_path.resolve()),
                "original": str(original_path.resolve()),
                "geometry": geometry,
            }
        )
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", default=[], help="image path; repeatable")
    parser.add_argument("--image-dir", action="append", default=[], help="directory of images")
    parser.add_argument("--video", action="append", default=[], help="held-out video; repeatable")
    parser.add_argument("--video-frames", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    rows = []
    image_paths = [Path(p) for p in args.image]
    for directory in args.image_dir:
        for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_paths.extend(sorted(Path(directory).glob(suffix)))
    for path in image_paths:
        add_image(path, path.stem, str(path), out_dir, rows)
    for video in args.video:
        add_video(Path(video), args.video_frames, out_dir, rows)
    if not rows:
        raise RuntimeError("no evaluation inputs")

    write_json(out_dir / "manifest.json", rows)
    print("wrote %d evaluation inputs to %s" % (len(rows), out_dir))


if __name__ == "__main__":
    main()
