"""Measure the full single-model inference pipeline on the RK3566.

``infer_npy_batch.py`` reports NPU-only time from pre-letterboxed canvases, which
answers "what does the graph cost". This answers the deployment question: what does
one frame cost end to end through the detector, split into its stages.

Stages, all measured with ``time.perf_counter`` on the board:

    decode      cv2.VideoCapture.read of one frame
    preprocess  letterbox to the frozen canvas (the same letterbox_bgr as deployment)
    npu         RKNNLite.inference
    postprocess decode_outputs: sigmoid, confidence filter, per-class NMS,
                coordinate restore back to the source frame

CPU and RSS are sampled around the run. No PC wall-clock is reused anywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
for _candidate in (_HERE.parents[2], _HERE.parents[2] / "tools", _HERE.parent):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

try:
    import cv2
except ImportError as exc:  # pragma: no cover - board only
    raise SystemExit("OpenCV is required") from exc

try:
    from competition_ai.input_shape import validate_model_shape
    from tools.profile_rknn_single_stream import letterbox_bgr
except ImportError:  # pragma: no cover - flat board layout
    from input_shape import validate_model_shape
    from profile_rknn_single_stream import letterbox_bgr

from decode_yolo_output import decode_outputs

STAGES = ("decode", "preprocess", "npu", "postprocess")


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": round(float(array.mean()), 3),
        "p50_ms": round(float(np.percentile(array, 50)), 3),
        "p95_ms": round(float(np.percentile(array, 95)), 3),
        "min_ms": round(float(array.min()), 3),
        "max_ms": round(float(array.max()), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        width, height = validate_model_shape((args.width, args.height))
    except ValueError as exc:
        parser.error(str(exc))

    try:
        import psutil

        process = psutil.Process(os.getpid())
    except ImportError:
        process = None

    from rknnlite.api import RKNNLite

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    runner = RKNNLite(verbose=False)
    timings: dict[str, list[float]] = {stage: [] for stage in STAGES}
    totals: list[float] = []
    detections = 0
    rss_after_init = None
    try:
        if runner.load_rknn(str(args.model)) != 0 or runner.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
            raise SystemExit(f"failed to initialize {args.model}")
        if process:
            rss_after_init = process.memory_info().rss
        cpu_before = process.cpu_percent(interval=None) if hasattr(process, "cpu_percent") else None
        ran = 0
        while ran < args.frames + args.warmup:
            started = time.perf_counter()
            ok, frame = capture.read()
            decode_ms = (time.perf_counter() - started) * 1000
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
                if not ok:
                    break
            started = time.perf_counter()
            canvas, scale, pad_x, pad_y = letterbox_bgr(frame, (width, height))
            preprocess_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            outputs = runner.inference(inputs=[canvas[None, ...]], data_format="nhwc")
            npu_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            decoded = decode_outputs(outputs, args.conf, args.iou)
            postprocess_ms = (time.perf_counter() - started) * 1000
            ran += 1
            if ran <= args.warmup:
                continue  # warmup frames are not recorded
            timings["decode"].append(decode_ms)
            timings["preprocess"].append(preprocess_ms)
            timings["npu"].append(npu_ms)
            timings["postprocess"].append(postprocess_ms)
            totals.append(decode_ms + preprocess_ms + npu_ms + postprocess_ms)
            detections += len(decoded)
        cpu_after = process.cpu_percent(interval=None) if hasattr(process, "cpu_percent") else None
        rss_after = process.memory_info().rss if process else None
    finally:
        runner.release()
        capture.release()

    if not totals:
        raise SystemExit("no frames were measured")

    stages = {stage: summarize(values) for stage, values in timings.items()}
    report = {
        "model": str(args.model),
        "video": str(args.video),
        "canvas": f"{width}x{height}",
        "canvas_pixels": width * height,
        "layout": "NHWC",
        "channel_order": "RGB",
        "dtype": "uint8",
        "input_array_shape": [1, height, width, 3],
        "conf": args.conf,
        "iou_nms": args.iou,
        "warmup_frames": args.warmup,
        "measured_frames": len(totals),
        "stages": stages,
        "total": summarize(totals),
        "effective_fps": round(1000.0 / float(np.asarray(totals).mean()), 3),
        "npu_only_fps": round(1000.0 / stages["npu"]["mean_ms"], 3),
        "total_detections": detections,
        "detections_per_frame": round(detections / len(totals), 3),
        "stage_share_percent": {
            stage: round(stages[stage]["mean_ms"] / float(np.asarray(totals).mean()) * 100, 2) for stage in STAGES
        },
        "process_rss_bytes": {"after_init": rss_after_init, "after_runs": rss_after},
        "process_cpu_percent": {"before": cpu_before, "after": cpu_after},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
