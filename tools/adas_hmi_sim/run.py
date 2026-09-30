"""Start the PC ADAS HMI simulator.

Examples::

    python tools/adas_hmi_sim/run.py --mode mock
    python tools/adas_hmi_sim/run.py --mode replay --video road.mp4 --detections detections.jsonl
    python tools/adas_hmi_sim/run.py --mode live --video road.mp4 --model yolo11n.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE))

from backend.navigation_state import NavigationState  # noqa: E402
from backend.scene_builder import SceneBuilder  # noqa: E402
from backend.server import SimulatorOptions, create_app  # noqa: E402
from backend.video_pipeline import LiveSource, MockSource, ReplaySource  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="PC ADAS driving-perception HMI simulator")
    result.add_argument("--mode", choices=("mock", "replay", "live"), default="mock")
    result.add_argument("--video", type=Path, help="Input road video for replay/live")
    result.add_argument("--detections", type=Path, help="JSONL detections for replay mode")
    result.add_argument("--model", type=Path, help=".pt (PC Ultralytics) or .rknn (CompetitionDetector) for live mode")
    result.add_argument("--config", type=Path, default=PACKAGE / "configs" / "default.json")
    result.add_argument("--route", type=Path, default=PACKAGE / "configs" / "demo_route.json")
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8765)
    result.add_argument("--fps", type=float, default=30.0, help="WebSocket scene update target")
    result.add_argument("--mock-start-ms", type=int, default=0, help="Seek the deterministic mock timeline before serving")
    result.add_argument("--no-browser", action="store_true", help="Do not launch a browser")
    return result


def make_source(args: argparse.Namespace):
    if args.mode == "mock":
        return MockSource(start_ms=args.mock_start_ms)
    if args.video is None:
        raise SystemExit(f"--video is required for --mode {args.mode}")
    if args.mode == "replay":
        if args.detections is None:
            raise SystemExit("--detections is required for replay mode")
        return ReplaySource(args.video, args.detections)
    model = args.model
    if model is None:
        for candidate in (ROOT / "yolo11n.pt", ROOT / "yolo11n-road7.pt", ROOT / "yolo11n.rknn"):
            if candidate.exists():
                model = candidate
                break
    if model is None:
        raise SystemExit("Live mode needs --model (the simulator never downloads or trains a model)")
    return LiveSource(args.video, model)


def main() -> None:
    args = parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}
    route = NavigationState.from_file(args.route) if args.route.exists() else NavigationState()
    source = make_source(args)
    options = SimulatorOptions(source, SceneBuilder(config), route, PACKAGE / "frontend", args.mode, max(1.0, args.fps))
    app = create_app(options)
    url = f"http://{args.host}:{args.port}/"
    print(f"ADAS HMI simulator ({args.mode}) -> {url}", flush=True)
    print("SceneFrame distance mode: PSEUDO_DISTANCE; press D in the browser for debug details.", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
