"""B-05L side-by-side same-frame comparison for selected official-video times."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_ai.traffic_light_tracking import classify_lamp_color  # noqa: E402
from tools.b05k_common import decode_yolo11, prepare_input  # noqa: E402
from tools.b05k_runtime import ModelRuntime  # noqa: E402

COLOR_BGR = {"RED": (40, 40, 235), "GREEN": (40, 210, 60), "YELLOW": (30, 220, 240), "UNKNOWN": (220, 220, 220)}


def parse_model(value: str) -> tuple[str, Path]:
    label, path = value.split("=", 1)
    return label, Path(path)


def frame_at(capture: cv2.VideoCapture, second: float) -> tuple[int, np.ndarray]:
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    index = int(round(second * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"cannot read frame {index} at {second:.3f}s")
    return index, frame


def annotate(frame: np.ndarray, detections: list[dict[str, Any]], label: str, ms: float) -> np.ndarray:
    canvas = frame.copy()
    for item in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in item["bbox"])
        color = str(item.get("color", "UNKNOWN"))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_BGR.get(color, COLOR_BGR["UNKNOWN"]), 2)
        text = f"{color} {item['confidence']:.2f}"
        cv2.putText(canvas, text, (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    COLOR_BGR.get(color, COLOR_BGR["UNKNOWN"]), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 38), (18, 24, 28), -1)
    cv2.putText(canvas, f"{label}   {len(detections)} det   {ms:.1f} ms", (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 245, 248), 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--times", type=float, nargs="+", required=True)
    parser.add_argument("--model", type=parse_model, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--preprocess", choices=("letterbox", "resize"), default="letterbox")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stack", choices=("vertical", "horizontal"), default="vertical")
    args = parser.parse_args()
    runtimes = {label: ModelRuntime("pytorch", path, device=args.device) for label, path in args.model}
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(args.video)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for second in args.times:
        index, frame = frame_at(capture, second)
        panels = []
        entries = []
        for label, runtime in runtimes.items():
            tensor, transform = prepare_input(frame, preprocess=args.preprocess)
            raw, ms = runtime.run(tensor)
            detections = decode_yolo11(raw, conf=args.conf, iou=args.iou)
            for item in detections:
                item["bbox"] = transform.network_to_frame(item["bbox"])
                observation = classify_lamp_color(frame, item["bbox"])
                item["color"] = observation.color.value
                item["color_valid"] = bool(observation.valid)
            entries.append({"label": label, "inference_ms": ms, "detections": detections})
            panels.append(annotate(frame, detections, label, ms))
        if args.stack == "vertical":
            sheet = np.vstack(panels)
        else:
            sheet = np.hstack(panels)
        stem = f"{args.video.stem}_t{second:07.3f}_frame{index:06d}"
        cv2.imwrite(str(args.output / f"{stem}_compare.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({"video": args.video.name, "second": second, "frame": index, "models": entries})
    capture.release()
    (args.output / f"{args.video.stem}_same_frame.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output.resolve()), "frames": len(records)}, indent=2))


if __name__ == "__main__":
    main()
