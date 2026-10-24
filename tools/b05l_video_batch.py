"""B-05L batch video runner for the three traffic-light Specialists.

Runs the existing B-05K frame-by-frame benchmark with identical input,
preprocessing, decoding and thresholds for every model.  Full MP4 files stay on
the Windows machine; only summaries and selected keyframes are committed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.b05k_video_benchmark import run as run_video  # noqa: E402

DEFAULT_VIDEOS = Path(r"D:\BaiduNetdiskDownload\紫光同创杯赛题三参考素材")
DEFAULT_OUT = Path(r"D:\Datasets\adas_edge_training\b05l_pc_validation")
FOUR_VIDEOS = [
    "video_20260929_113421.mp4",
    "video_20260929_112507.mp4",
    "video_20260929_113539.mp4",
    "video_20260929_112739.mp4",
]


def parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("model must be LABEL=PATH")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", type=parse_model, action="append", required=True,
                        help="LABEL=path/to/best.pt; repeat for each model")
    parser.add_argument("--video", action="append", default=None, help="optional video filename filter")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--preprocess", choices=("letterbox", "resize"), default="letterbox")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.video:
        names = args.video
    else:
        names = FOUR_VIDEOS
    videos = [args.videos_dir / name for name in names]
    missing = [str(path) for path in videos if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing official videos: " + ", ".join(missing))
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    started = time.perf_counter()
    for label, model_path in args.model:
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        for video in videos:
            annotated = args.output / "videos" / f"annotated_{label}" / video.name
            json_path = args.output / "metrics" / "timelines" / label / f"{video.stem}.json"
            csv_path = args.output / "logs" / "runtime" / label / f"{video.stem}.csv"
            keyframes = args.output / "visualizations" / "keyframes" / label / video.stem
            if args.skip_existing and json_path.exists() and annotated.exists():
                summaries.append(json.loads(json_path.read_text(encoding="utf-8"))["summary"])
                print(f"SKIP {label} {video.name}", flush=True)
                continue
            print(f"RUN  {label} {video.name}", flush=True)
            args_ns = SimpleNamespace(
                video=str(video), model=model_path, model_manifest=None,
                backend="pytorch", output=str(annotated), conf=args.conf, iou=args.iou,
                preprocess=args.preprocess, visualize=True, save_json=True,
                json_path=str(json_path), runtime_csv=str(csv_path),
                environment_json=str(args.output / "logs" / "environment.json"),
                keyframes_dir=str(keyframes), schedule="every-frame",
                device=args.device, roi_top=0.60, input_width=512, input_height=192,
                max_det=300, warmup=2, overwrite=True, model_label=label,
            )
            summary = run_video(args_ns)
            summary["label"] = label
            summaries.append(summary)
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - cleanup is best-effort
                pass
            print(f"DONE {label} {video.name} det_frames={summary['frames_with_new_detections']} "
                  f"fps={summary['offline_processing_fps']:.2f}", flush=True)
    payload = {
        "videos_dir": str(args.videos_dir.resolve()),
        "models": [{"label": label, "path": str(path.resolve())} for label, path in args.model],
        "conf": args.conf,
        "iou": args.iou,
        "preprocess": args.preprocess,
        "schedule": "every-frame",
        "elapsed_seconds": time.perf_counter() - started,
        "summaries": summaries,
    }
    metrics = args.output / "metrics" / "video_detection_timeline.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"output": str(metrics.resolve()), "runs": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
