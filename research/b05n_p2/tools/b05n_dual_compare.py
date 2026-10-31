"""Render one official video with dual BDD+Specialist and single P2 side by side."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_ai.traffic_light_tracking import classify_lamp_color  # noqa: E402
from tools.b05k_common import decode_yolo11, prepare_input  # noqa: E402
from tools.b05n_tl_eval import decode_class, prepare_full  # noqa: E402

CLASS_BGR = {
    "person": (0, 200, 255), "rider": (255, 200, 0), "car": (0, 255, 0), "truck": (0, 128, 255),
    "bus": (255, 0, 255), "train": (128, 0, 255), "motor": (255, 128, 0), "bike": (0, 255, 255),
    "traffic light": (0, 0, 255), "traffic sign": (255, 255, 0),
}
TL_COLOR = {"RED": (40, 40, 235), "GREEN": (40, 210, 60), "YELLOW": (30, 220, 240), "UNKNOWN": (230, 230, 230)}


def detections_full(model: Any, frame: np.ndarray, *, conf: float, iou: float, skip_tl: bool) -> list[dict]:
    import torch
    tensor, transform = prepare_full(frame, 512, 288)
    with torch.inference_mode():
        raw = model(torch.from_numpy(tensor).cuda())
    out = []
    for class_id, name in model.names.items():
        if class_id == 8 and skip_tl:
            continue
        for item in decode_class(raw, int(class_id), conf, iou):
            x1, y1, x2, y2 = item["bbox"]
            item["bbox"] = [x1 / transform["scale_x"], y1 / transform["scale_y"], x2 / transform["scale_x"], y2 / transform["scale_y"]]
            item["class_name"] = str(name)
            if name == "traffic light":
                obs = classify_lamp_color(frame, item["bbox"])
                item["color"] = obs.color.value
            out.append(item)
    return out


def detections_specialist(model: Any, frame: np.ndarray, *, conf: float, iou: float) -> list[dict]:
    import torch
    tensor, transform = prepare_input(frame, input_width=512, input_height=192, roi_top=0.60, preprocess="resize")
    with torch.inference_mode():
        raw = model(torch.from_numpy(tensor).cuda())
    out = []
    for item in decode_yolo11(raw, conf=conf, iou=iou):
        item["bbox"] = transform.network_to_frame(item["bbox"])
        obs = classify_lamp_color(frame, item["bbox"])
        item["class_name"] = "traffic light"
        item["color"] = obs.color.value
        out.append(item)
    return out


def draw(frame: np.ndarray, detections: list[dict], title: str, infer_ms: float) -> np.ndarray:
    canvas = frame.copy()
    for item in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in item["bbox"])
        if item.get("class_name") == "traffic light":
            color = TL_COLOR.get(item.get("color", "UNKNOWN"), TL_COLOR["UNKNOWN"])
            label = f"{item.get('color', 'UNKNOWN')} {item['confidence']:.2f}"
        else:
            color = CLASS_BGR.get(item["class_name"], (220, 220, 220))
            label = f"{item['class_name']} {item['confidence']:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, label, (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (18, 24, 28), -1)
    cv2.putText(canvas, f"{title}   {len(detections)} det   {infer_ms:.1f} ms", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 248, 250), 2, cv2.LINE_AA)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--bdd", type=Path, required=True)
    ap.add_argument("--specialist", type=Path, required=True)
    ap.add_argument("--p2", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--keyframe-seconds", type=float, nargs="*", default=[1.8])
    args = ap.parse_args()
    from ultralytics import YOLO
    import torch

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = {
        "bdd": YOLO(str(args.bdd)).model.cuda().eval(),
        "spec": YOLO(str(args.specialist)).model.cuda().eval(),
        "p2": YOLO(str(args.p2)).model.cuda().eval(),
    }
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(args.video)
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dual_writer = cv2.VideoWriter(str(args.output_dir / f"{args.video.stem}_dual.mp4"),
                                   cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    p2_writer = cv2.VideoWriter(str(args.output_dir / f"{args.video.stem}_p2_single.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    comp_writer = cv2.VideoWriter(str(args.output_dir / f"{args.video.stem}_side_by_side.mp4"),
                                   cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
    index = 0
    records = []
    started = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        t0 = time.perf_counter()
        bdd = detections_full(models["bdd"], frame, conf=args.conf, iou=args.iou, skip_tl=True)
        spec = detections_specialist(models["spec"], frame, conf=args.conf, iou=args.iou)
        t_dual = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        p2 = detections_full(models["p2"], frame, conf=args.conf, iou=args.iou, skip_tl=False)
        t_p2 = (time.perf_counter() - t0) * 1000.0
        dual_frame = draw(frame, bdd + spec, "DUAL: BDD-YOLOv8n + P2 Specialist (both every frame)", t_dual)
        p2_frame = draw(frame, p2, "SINGLE: YOLOv8n-P2 10-class (every frame)", t_p2)
        dual_writer.write(dual_frame)
        p2_writer.write(p2_frame)
        left = cv2.resize(dual_frame, (960, 540), interpolation=cv2.INTER_AREA)
        right = cv2.resize(p2_frame, (960, 540), interpolation=cv2.INTER_AREA)
        comp = np.hstack([left, right])
        cv2.rectangle(comp, (0, 0), (1920, 34), (16, 20, 24), -1)
        cv2.putText(comp, f"Frame {index:06d}  {index/fps:07.3f}s  |  DUAL (left)  vs  P2 SINGLE (right)",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 248, 250), 2, cv2.LINE_AA)
        comp_writer.write(comp)
        if any(abs(index / fps - t) < 0.5 / fps for t in args.keyframe_seconds):
            cv2.imwrite(str(args.output_dir / f"{args.video.stem}_frame_{index:06d}_compare.jpg"), comp,
                        [cv2.IMWRITE_JPEG_QUALITY, 94])
        records.append({"frame": index, "time_s": round(index / fps, 3), "dual_ms": round(t_dual, 2),
                        "p2_ms": round(t_p2, 2), "dual_detections": len(bdd) + len(spec), "p2_detections": len(p2)})
        index += 1
        if index % 60 == 0:
            print(f"frame {index} / {index/fps:.1f}s", flush=True)
    capture.release()
    for writer in (dual_writer, p2_writer, comp_writer):
        writer.release()
    summary = {"video": str(args.video.resolve()), "frames": index, "fps": fps,
               "outputs": {"dual": str(args.output_dir / f"{args.video.stem}_dual.mp4"),
                           "p2_single": str(args.output_dir / f"{args.video.stem}_p2_single.mp4"),
                           "side_by_side": str(args.output_dir / f"{args.video.stem}_side_by_side.mp4")},
               "elapsed_s": round(time.perf_counter() - started, 1), "records": records}
    (args.output_dir / f"{args.video.stem}_compare.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
