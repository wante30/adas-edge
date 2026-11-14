"""Build letterboxed RGB .npy calibration data for RKNN INT8.

Each output is exactly the tensor RKNN Lite receives:
shape [1, 192, 320, 3], dtype uint8, RGB, letterbox fill 114.
The dataset.txt references the .npy files so RKNN's image loader cannot
silently stretch the frame or reorder channels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from common import INPUT_SIZE, letterbox_rgb, write_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_video(path: Path, count: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("cannot open video: %s" % path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0:
        cap.release()
        raise RuntimeError("cannot determine frame count: %s" % path)
    # Evenly spaced, avoiding the first/last frame.
    indices = np.linspace(0, max(0, total - 1), count + 2, dtype=np.int64)[1:-1]
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        yield int(index), frame, dict(total_frames=total, fps=fps, width=width, height=height)
    cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", action="append", required=True, help="source video; repeatable")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=192)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    image_dir = output_dir / "images"
    preview_dir = output_dir / "previews"
    image_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    videos = [Path(v) for v in args.video]
    per_video = max(1, args.limit // len(videos))
    rows = []
    serial = 0
    for video in videos:
        for frame_index, frame, meta in sample_video(video, per_video):
            if serial >= args.limit:
                break
            rgb, geometry = letterbox_rgb(frame, size=INPUT_SIZE)
            name = "%04d" % serial
            npy_path = image_dir / (name + ".npy")
            preview_path = preview_dir / (name + ".jpg")
            # RKNN Toolkit's .npy calibration loader expects the model input
            # layout (NCHW), not the RKNN Lite runtime NHWC layout.
            np.save(
                npy_path,
                np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.uint8),
            )
            cv2.imwrite(str(preview_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            rows.append(
                {
                    "id": name,
                    "source": str(video),
                    "frame_index": frame_index,
                    "npy": str(npy_path),
                    "preview": str(preview_path),
                    "geometry": geometry,
                    "source_meta": meta,
                    "sha256": sha256_file(npy_path),
                }
            )
            serial += 1
        if serial >= args.limit:
            break

    if not rows:
        raise RuntimeError("no calibration frames were extracted")
    dataset_path = output_dir / "calibration.txt"
    dataset_path.write_text(
        "\n".join(row["npy"] for row in rows) + "\n", encoding="utf-8"
    )
    write_json(
        output_dir / "manifest.json",
        {
            "input_contract": {
                "color": "RGB",
                "shape": [1, 3, INPUT_SIZE[1], INPUT_SIZE[0]],
                "layout": "NCHW (RKNN Toolkit calibration .npy)",
                "dtype": "uint8",
                "letterbox_fill": 114,
            },
            "count": len(rows),
            "rows": rows,
            "dataset_txt": str(dataset_path),
        },
    )
    print("wrote %d calibration samples to %s" % (len(rows), output_dir))


if __name__ == "__main__":
    main()
