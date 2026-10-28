"""Run the lightweight four-camera BEV compositor on synchronized videos.

This is the board-portable part of the upstream surround-view approach: all
camera geometry is cached as perspective maps and each frame uses OpenCV only.
It intentionally does not download or run PyTorch depth/lane models.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

try:
    from tools.adas_hmi_sim.backend.surround_bev import SurroundBEVComposer, load_bev_config
    from tools.adas_hmi_sim.backend.calibrated_surround import CalibratedSurroundComposer
except ImportError:  # flat copy used for the board smoke test
    from surround_bev import SurroundBEVComposer, load_bev_config
    from calibrated_surround import CalibratedSurroundComposer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--rear", "--back", dest="rear", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--lut-dir", type=Path, help="precomputed calibrated LUT directory")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    captures = {role: cv2.VideoCapture(str(path)) for role, path in (("front", args.front), ("left", args.left), ("right", args.right), ("rear", args.rear))}
    if not all(cap.isOpened() for cap in captures.values()):
        bad = [role for role, cap in captures.items() if not cap.isOpened()]
        raise SystemExit(f"cannot open camera videos: {', '.join(bad)}")
    composer = CalibratedSurroundComposer(args.lut_dir) if args.lut_dir else SurroundBEVComposer(load_bev_config(args.config))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (composer.width, composer.height))
    if not writer.isOpened():
        raise SystemExit(f"cannot create {args.output}")
    measured = 0
    decode_seconds = 0.0
    compose_seconds = 0.0
    started = time.perf_counter()
    try:
        for _ in range(max(1, args.frames)):
            frames = {}
            decode_started = time.perf_counter()
            for role, capture in captures.items():
                ok, frame = capture.read()
                if not ok:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = capture.read()
                if not ok:
                    raise SystemExit(f"{role} video ended without a readable frame")
                frames[role] = frame
            decode_seconds += time.perf_counter() - decode_started
            compose_started = time.perf_counter()
            writer.write(composer.compose(frames))
            compose_seconds += time.perf_counter() - compose_started
            measured += 1
    finally:
        writer.release()
        for capture in captures.values():
            capture.release()
        composer.close()
    elapsed = max(1e-6, time.perf_counter() - started)
    print({"frames": measured, "elapsed_s": round(elapsed, 3), "compose_fps": round(measured / elapsed, 3), "decode_mean_ms": round(decode_seconds * 1000 / measured, 3), "compose_mean_ms": round(compose_seconds * 1000 / measured, 3), "output": str(args.output), **composer.metadata()})


if __name__ == "__main__":
    main()
