"""B-05N video inference for the ten-class full-frame model.

Default mode draws traffic lights only, using the same HSV color function as the
dual-model pipeline.  --classes all draws every BDD class for visual inspection.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_ai.traffic_light_tracking import classify_lamp_color  # noqa: E402
from tools.b05n_tl_eval import decode_class, prepare_full  # noqa: E402

COLOR_BGR = {"RED": (40, 40, 235), "GREEN": (40, 210, 60), "YELLOW": (30, 220, 240), "UNKNOWN": (230, 230, 230)}
CLASS_BGR = {
    "person": (0, 200, 255), "rider": (255, 200, 0), "car": (0, 255, 0), "truck": (0, 128, 255),
    "bus": (255, 0, 255), "train": (128, 0, 255), "motor": (255, 128, 0), "bike": (0, 255, 255),
    "traffic light": (0, 0, 255), "traffic sign": (255, 255, 0),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--classes", choices=("tl", "all"), default="tl")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(args.output)
    from ultralytics import YOLO
    import torch

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model)).model.to(args.device).eval()
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(args.video)
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not output.isOpened():
        raise RuntimeError(args.output)
    rows = []
    times = []
    colors = Counter()
    first = None
    started = time.perf_counter()
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        tensor, transform = prepare_full(frame, 512, 288)
        input_tensor = torch.from_numpy(tensor).to(args.device)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            raw = model(input_tensor)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t0) * 1000.0
        times.append(infer_ms)
        detections = []
        class_ids = [8] if args.classes == "tl" else list(range(10))
        for class_id in class_ids:
            for item in decode_class(raw, class_id, args.conf, args.iou):
                x1, y1, x2, y2 = item["bbox"]
                item["bbox"] = [x1 / transform["scale_x"], y1 / transform["scale_y"],
                                x2 / transform["scale_x"], y2 / transform["scale_y"]]
                item["class_id"] = class_id
                item["class_name"] = "traffic light" if class_id == 8 else str(model.names[class_id])
                if class_id == 8:
                    observation = classify_lamp_color(frame, item["bbox"])
                    item["color"] = observation.color.value
                    item["color_valid"] = bool(observation.valid)
                    colors[item["color"]] += 1
                detections.append(item)
        if detections and first is None:
            first = {"frame": frame_index, "seconds": round(frame_index / fps, 3), **detections[0]}
        canvas = frame.copy()
        for item in detections:
            x1, y1, x2, y2 = (int(round(v)) for v in item["bbox"])
            if item["class_id"] == 8:
                color = COLOR_BGR.get(item.get("color", "UNKNOWN"), COLOR_BGR["UNKNOWN"])
                label = f"{item.get('color', 'UNKNOWN')} {item['confidence']:.2f}"
            else:
                color = CLASS_BGR.get(item["class_name"], (200, 200, 200))
                label = f"{item['class_name']} {item['confidence']:.2f}"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(canvas, label, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        elapsed = max(1e-9, time.perf_counter() - started)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 38), (18, 24, 28), -1)
        cv2.putText(canvas, f"Frame {frame_index:06d} {frame_index/fps:07.3f}s B05N-P2-10cls "
                            f"{args.classes} {infer_ms:.1f}ms offline {(frame_index+1)/elapsed:.1f}FPS",
                    (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 248, 250), 1, cv2.LINE_AA)
        output.write(canvas)
        if args.save_json:
            rows.append({"frame": frame_index, "time_s": frame_index / fps, "inference_ms": infer_ms,
                         "detections": detections})
        frame_index += 1
    capture.release()
    output.release()
    summary = {
        "video": str(args.video.resolve()), "model": str(args.model.resolve()), "classes": args.classes,
        "frames": frame_index, "fps": fps, "first_detection": first,
        "detections": sum(len(row["detections"]) for row in rows) if rows else None,
        "color_counts": {key: colors.get(key, 0) for key in ("RED", "GREEN", "YELLOW", "UNKNOWN")},
        "mean_inference_ms": statistics.fmean(times), "p95_inference_ms": float(np.percentile(times, 95)),
        "offline_fps": frame_index / (time.perf_counter() - started), "output": str(args.output.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.with_suffix(".json")).write_text(
        json.dumps({"summary": summary, "frames": rows} if args.save_json else {"summary": summary}, indent=2),
        encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
