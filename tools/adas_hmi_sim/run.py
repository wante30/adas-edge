"""Start the PC ADAS HMI simulator.

Examples::

    python tools/adas_hmi_sim/run.py --mode mock
    python tools/adas_hmi_sim/run.py --mode pc-system
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
from backend.tesla_bevcam_renderer import TeslaBevCamRenderer  # noqa: E402
from backend.video_pipeline import FourStreamMockSource, LiveSource, MockSource, ReplaySource, SurroundVideoSource  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="PC ADAS driving-perception HMI simulator")
    result.add_argument("--mode", choices=("mock", "pc-system", "surround", "replay", "live", "front"), default="mock")
    result.add_argument("--video", type=Path, help="Input road video for replay/live")
    result.add_argument("--start-frame", type=int, default=0, help="seek the live/front video before serving")
    result.add_argument("--loop-video", action="store_true", help="loop the live/front input when it reaches EOF")
    result.add_argument("--detections", type=Path, help="JSONL detections for replay mode")
    result.add_argument("--model", type=Path, help=".pt for live mode, or .rknn for live/surround front detection")
    result.add_argument("--input-size", default=None, help="front/live detector canvas, e.g. 512x288 or 640x384")
    result.add_argument("--confidence", type=float, default=0.15, help="detector confidence threshold")
    result.add_argument("--detector-fps", type=float, default=12.0, help="front/live detector update target")
    result.add_argument("--traffic-light-min-confidence", type=float, default=0.25, help="minimum experimental traffic-light color candidate confidence; does not establish ego-lane association")
    result.add_argument("--traffic-light-hold-frames", type=int, default=4, help="hold the last confirmed vehicle-signal color through detector gaps")
    result.add_argument("--traffic-light-association", choices=("unverified", "geometry-experimental"), default="unverified", help="optional experimental signal-to-ego-lane geometry gate")
    result.add_argument("--lane-stride", type=int, default=None, help="run CPU lane candidate every N detector updates; RKNN defaults to 3, PC to 1")
    result.add_argument("--signal-roi", action="store_true", help="PC .pt only: run a second enlarged upper-signal ROI pass")
    result.add_argument("--signal-roi-confidence", type=float, default=0.10, help="PC signal ROI detector threshold")
    result.add_argument("--signal-roi-input-size", default="640x384", help="PC signal ROI detector canvas")
    result.add_argument("--signal-roi-top", type=float, default=0.55, help="PC signal ROI top-of-frame fraction")
    result.add_argument("--traffic-light-model", type=Path, default=None, help="optional dedicated traffic-light specialist (.pt on PC or .rknn on RK3566)")
    result.add_argument("--traffic-light-roi-mode", choices=("top60", "adaptive"), default="top60", help="same 512x192 model, one pass per signal refresh")
    result.add_argument("--traffic-light-confidence", type=float, default=0.05, help="specialist candidate threshold")
    result.add_argument("--traffic-light-fps", type=float, default=3.0, help="specialist refresh target; 3 FPS matches the competition budget")
    result.add_argument("--traffic-light-roi-top", type=float, default=0.60, help="specialist top-of-frame ROI fraction")
    result.add_argument("--traffic-light-specialist-mode", choices=("fuse", "replace"), default="fuse", help="fuse specialist candidates with main detector or replace main traffic-light boxes")
    result.add_argument("--surround-detector-fps", type=float, default=10.0, help="front RKNN detector target for surround mode")
    result.add_argument("--surround-detector-input-size", default="320", help="front RKNN shape, e.g. 320 or 320x192")
    result.add_argument("--surround-dir", type=Path, help="directory containing front/left/right/rear.mp4 for surround mode")
    result.add_argument("--bev-config", type=Path, default=PACKAGE / "configs" / "surround_bev.json")
    result.add_argument("--bev-lut-dir", type=Path, help="calibrated Open-3D-Surround-View LUT directory; overrides --bev-config")
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
    if args.mode == "pc-system":
        return FourStreamMockSource(start_ms=args.mock_start_ms)
    if args.mode == "surround":
        if args.surround_dir is None:
            raise SystemExit("--surround-dir is required for --mode surround")
        detector_input_size = args.surround_detector_input_size
        if "x" in detector_input_size.lower():
            width, height = detector_input_size.lower().split("x", 1)
            detector_input_size = (int(width), int(height))
        else:
            detector_input_size = int(detector_input_size)
        return SurroundVideoSource(args.surround_dir, args.bev_config, detector_model=args.model, detector_fps=args.surround_detector_fps, detector_input_size=detector_input_size, lut_dir=args.bev_lut_dir)
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
    if args.input_size is None:
        input_size = "512x288" if args.mode == "front" else 640
    else:
        input_size = args.input_size
    if isinstance(input_size, str) and "x" in input_size.lower():
        width, height = input_size.lower().split("x", 1)
        input_size = (int(width), int(height))
    elif isinstance(input_size, str):
        input_size = int(input_size)
    signal_roi_input_size = args.signal_roi_input_size
    if isinstance(signal_roi_input_size, str) and "x" in signal_roi_input_size.lower():
        width, height = signal_roi_input_size.lower().split("x", 1)
        signal_roi_input_size = (int(width), int(height))
    elif isinstance(signal_roi_input_size, str):
        signal_roi_input_size = int(signal_roi_input_size)
    lane_stride = args.lane_stride
    if lane_stride is None:
        lane_stride = 3 if Path(model).suffix.lower() == ".rknn" else 1
    return LiveSource(
        args.video,
        model,
        confidence=args.confidence,
        fps_limit=args.detector_fps,
        input_size=input_size,
        traffic_light_min_confidence=args.traffic_light_min_confidence,
        traffic_light_hold_frames=args.traffic_light_hold_frames,
        traffic_light_association=args.traffic_light_association,
        lane_stride=lane_stride,
        signal_roi=args.signal_roi,
        signal_roi_confidence=args.signal_roi_confidence,
        signal_roi_input_size=signal_roi_input_size,
        signal_roi_top=args.signal_roi_top,
        traffic_light_model=args.traffic_light_model,
        traffic_light_roi_mode=args.traffic_light_roi_mode,
        traffic_light_confidence=args.traffic_light_confidence,
        traffic_light_fps=args.traffic_light_fps,
        traffic_light_roi_top=args.traffic_light_roi_top,
        traffic_light_specialist_mode=args.traffic_light_specialist_mode,
        start_frame=args.start_frame,
        loop=args.loop_video,
    )


def main() -> None:
    args = parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config.exists() else {}
    route = NavigationState.from_file(args.route) if args.route.exists() else NavigationState()
    source = make_source(args)
    options = SimulatorOptions(source, SceneBuilder(config), route, PACKAGE / "frontend", args.mode, max(1.0, args.fps), tesla_renderer=TeslaBevCamRenderer())
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
