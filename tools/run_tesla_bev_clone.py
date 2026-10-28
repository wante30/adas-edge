"""Run the upstream Tesla-BevCam-Clone on a real front-camera video.

This is a thin experiment wrapper. The upstream repository is kept under
``.deps/Tesla-BevCam-Clone`` and is not copied into the competition pipeline.
The wrapper only supplies video/model paths and records which optional stages
are enabled. It remains separate from the RKNN/FPGA path.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / ".deps" / "Tesla-BevCam-Clone" / "main.py"


def _load_upstream():
    if not UPSTREAM.exists():
        raise SystemExit(
            "Missing upstream checkout. Run: "
            "git clone --depth 1 https://github.com/Ga0512/Tesla-BevCam-Clone.git "
            ".deps/Tesla-BevCam-Clone"
        )
    spec = importlib.util.spec_from_file_location("tesla_bev_clone", UPSTREAM)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load upstream module: {UPSTREAM}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run upstream Tesla-BevCam on a real front video")
    p.add_argument("--video", type=Path, required=True, help="front-camera MP4")
    p.add_argument("--output", type=Path, required=True, help="generated BEV demo MP4")
    p.add_argument("--model", type=Path, default=ROOT / "yolov8n.pt", help="Ultralytics .pt model")
    p.add_argument("--frames", type=int, default=150, help="number of frames; use 0 for the upstream default")
    p.add_argument("--depth", action="store_true", help="enable upstream Depth Anything V2 stage")
    p.add_argument("--ufld", type=Path, help="UFLDv2 Tusimple .pth; omitted keeps lane stage disabled")
    return p


def main() -> None:
    args = parser().parse_args()
    video = args.video.resolve()
    model = args.model.resolve()
    output = args.output.resolve()
    if not video.exists():
        raise SystemExit(f"Input video does not exist: {video}")
    if not model.exists():
        raise SystemExit(f"YOLO model does not exist: {model}")

    module = _load_upstream()
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)

    # The upstream main() constructs YOLO("yolov8n.pt") internally. Replace
    # that constructor without modifying the checked-out upstream source.
    from ultralytics import YOLO

    module.YOLO = lambda _unused: YOLO(str(model))
    module.VIDEOS = {"front": str(video), "back": None, "left": None, "right": None}
    module.DEPTH_MODEL = "depth_anything_v2" if args.depth else None
    module.UFLD_MODEL = str(args.ufld.resolve()) if args.ufld else None
    module.MAX_FRAMES = args.frames if args.frames > 0 else module.FPS * 25
    module.OUT_VIDEO = str(output)
    module.OUT_MAP = str(output.with_name(output.stem + "_map.mp4"))

    print(f"input={video}")
    print(f"model={model}")
    print(f"output={output}")
    print(f"depth={'enabled' if args.depth else 'disabled'}")
    print(f"ufld={'enabled' if args.ufld else 'disabled'}")
    module.main()


if __name__ == "__main__":
    main()
