"""Run the traffic-light specialist on a full video and save traceable output."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_ai.traffic_light import TrafficLightState  # noqa: E402
from competition_ai.traffic_light_tracking import (  # noqa: E402
    TrafficLightTracker,
    classify_lamp_color,
)
from tools.b05k_common import decode_yolo11, prepare_input, should_run_inference  # noqa: E402
from tools.b05k_runtime import ModelRuntime, resolve_model_path  # noqa: E402


COLOR_BGR = {
    "RED": (40, 40, 235),
    "GREEN": (40, 210, 60),
    "YELLOW": (30, 220, 240),
    "UNKNOWN": (230, 230, 230),
}


def _save_json(path: Path, payload: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {path} (use --overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _environment(runtime: ModelRuntime) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "runtime_providers": runtime.providers,
        "backend_device": runtime.device,
    }
    try:
        import torch

        info["pytorch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["pytorch"] = None
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["onnxruntime_available_providers"] = ort.get_available_providers()
    except ImportError:
        info["onnxruntime"] = None
    return info


def _annotate(
    frame: np.ndarray,
    *,
    frame_index: int,
    timestamp_s: float,
    model_name: str,
    infer_ms: float | None,
    offline_fps: float,
    detections: list[dict[str, Any]],
    tracks: list[dict[str, Any]],
    schedule: str,
) -> np.ndarray:
    canvas = frame.copy()
    if schedule == "every-frame":
        entries = [(item["bbox"], item, False) for item in detections]
    else:
        entries = [(item["bbox"], item, True) for item in tracks]
    for bbox, item, tracked in entries:
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        color = str(item.get("color", "UNKNOWN"))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_BGR.get(color, COLOR_BGR["UNKNOWN"]), 2)
        if tracked:
            label = (
                f"TRACKED {color} det@{item.get('detection_frame_id', '?')} "
                f"age={item.get('detection_age_ms', 0):.0f}ms {item.get('confidence', 0):.2f}"
            )
        else:
            label = f"{color} {item['confidence']:.2f}"
        top = max(16, y1 - 5)
        cv2.putText(canvas, label, (x1, top), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    COLOR_BGR.get(color, COLOR_BGR["UNKNOWN"]), 2, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 72), (18, 24, 28), -1)
    first_line = f"Frame {frame_index:06d}   {timestamp_s:07.3f}s   {model_name}"
    second_line = (
        f"{schedule}   infer {infer_ms:.2f} ms" if infer_ms is not None
        else f"{schedule}   inference skipped"
    )
    second_line += f"   offline {offline_fps:.1f} FPS"
    cv2.putText(canvas, first_line, (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (245, 248, 250), 2, cv2.LINE_AA)
    cv2.putText(canvas, second_line, (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (180, 215, 220), 1, cv2.LINE_AA)
    return canvas


def _runs(frame_presence: list[bool], fps: float) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    start = None
    for index, present in enumerate(frame_presence + [False]):
        if present and start is None:
            start = index
        elif not present and start is not None:
            end = index - 1
            runs.append({
                "start_frame": start,
                "end_frame": end,
                "start_seconds": round(start / fps, 3),
                "end_seconds": round(end / fps, 3),
            })
            start = None
    return runs


def _keyframe(path: Path, image: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])


def iter_video_frames(capture: cv2.VideoCapture, first_frame: np.ndarray):
    """Yield the already-read warmup frame, then continue from the capture cursor."""
    yield first_frame
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        yield frame


def run(args: argparse.Namespace) -> dict[str, Any]:
    video_path = Path(args.video).resolve()
    model_path = resolve_model_path(
        backend=args.backend, model_path=args.model, manifest_path=args.model_manifest
    )
    output_path = Path(args.output).resolve()
    json_path = Path(args.json_path).resolve() if args.json_path else output_path.with_suffix(".json")
    csv_path = Path(args.runtime_csv).resolve() if args.runtime_csv else output_path.with_suffix(".runtime.csv")
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    planned = [csv_path]
    if args.visualize:
        planned.append(output_path)
    if args.save_json:
        planned.append(json_path)
    environment_path = Path(args.environment_json).resolve() if args.environment_json else None
    if environment_path is not None:
        planned.append(environment_path)
    collisions = [path for path in planned if path.exists()]
    if collisions and not args.overwrite:
        raise FileExistsError("refusing to overwrite: " + ", ".join(map(str, collisions)) + " (use --overwrite)")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_count_hint = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    runtime = ModelRuntime(args.backend, model_path, device=args.device)
    runtime.validate_canvas(args.input_width, args.input_height)
    if args.input_width % 32 or args.input_height % 32:
        raise ValueError("input width and height must be multiples of 32")
    if args.schedule == "1hz" and args.conf < 0.25:
        raise ValueError("the existing B-05E tracker accepts detections at conf >= 0.25; use --conf 0.25 or higher")

    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError(f"video has no decodable frames: {video_path}")
    warmup_tensor, _ = prepare_input(
        first_frame, input_width=args.input_width, input_height=args.input_height,
        roi_top=args.roi_top, preprocess=args.preprocess,
    )
    runtime.warmup(warmup_tensor, runs=args.warmup)

    writer = None
    if args.visualize:
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"cannot create annotated video: {output_path}")

    tracker = TrafficLightTracker() if args.schedule == "1hz" else None
    model_label = getattr(args, "model_label", None) or (
        model_path.parent.parent.name if model_path.parent.name.lower() == "weights" else model_path.stem
    )
    rows: list[dict[str, Any]] = []
    presence: list[bool] = []
    inference_ms: list[float] = []
    color_counts: Counter[str] = Counter()
    track_color_counts: Counter[str] = Counter()
    track_boxes: dict[int, list[list[float]]] = defaultdict(list)
    first_detection = None
    first_trusted_detection = None
    event_saved: set[str] = set()
    csv_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for frame_index, frame in enumerate(iter_video_frames(capture, first_frame)):
        frame_start = time.perf_counter()
        timestamp_s = frame_index / fps
        scheduled = should_run_inference(frame_index, fps, args.schedule)
        tensor = None
        preprocess_ms = inference_elapsed = postprocess_ms = None
        detections: list[dict[str, Any]] = []
        tracks: list[dict[str, Any]] = []
        input_hash = None
        if scheduled:
            t0 = time.perf_counter()
            tensor, transform = prepare_input(
                frame, input_width=args.input_width, input_height=args.input_height,
                roi_top=args.roi_top, preprocess=args.preprocess,
            )
            preprocess_ms = (time.perf_counter() - t0) * 1000.0
            if args.save_json:
                input_hash = hashlib.sha256(tensor.tobytes()).hexdigest()
            raw, inference_elapsed = runtime.run(tensor)
            inference_ms.append(inference_elapsed)
            t0 = time.perf_counter()
            decoded = decode_yolo11(raw, conf=args.conf, iou=args.iou, max_det=args.max_det)
            for detection in decoded:
                detection["bbox"] = transform.network_to_frame(detection["bbox"])
                observation = classify_lamp_color(frame, detection["bbox"])
                detection["color"] = observation.color.value
                detection["color_valid"] = bool(observation.valid)
                detection["color_confidence"] = float(observation.confidence)
                color_counts[detection["color"]] += 1
            detections = decoded
            postprocess_ms = (time.perf_counter() - t0) * 1000.0

        if tracker is not None:
            timestamp_ns = int(round(timestamp_s * 1_000_000_000))
            if scheduled:
                track_inputs = [
                    {"class_name": "traffic light", "bbox": item["bbox"], "confidence": item["confidence"]}
                    for item in detections
                ]
                tracks = tracker.update_detections(
                    frame, track_inputs, source_frame_id=frame_index,
                    timestamp_ns=timestamp_ns, generation=video_path.name,
                )
            else:
                tracks = tracker.update_frame(
                    frame, source_frame_id=frame_index,
                    timestamp_ns=timestamp_ns, generation=video_path.name,
                )
            for track in tracks:
                track_color_counts[str(track["color"])] += 1
                if track["track_valid"]:
                    track_boxes[int(track["track_id"])].append(list(map(float, track["bbox"])))

        presence.append(bool(detections))
        if detections and first_detection is None:
            first_detection = {"frame": frame_index, "seconds": round(timestamp_s, 3), **detections[0]}
        trusted = next((item for item in detections if item["confidence"] >= max(args.conf, 0.25)), None)
        if trusted is not None and first_trusted_detection is None:
            first_trusted_detection = {"frame": frame_index, "seconds": round(timestamp_s, 3), **trusted}

        row = {
            "video": video_path.name,
            "frame": frame_index,
            "video_time_ms": round(timestamp_s * 1000.0, 3),
            "video_time": f"{int(timestamp_s // 60):02d}:{timestamp_s % 60:06.3f}",
            "frame_size": [frame_width, frame_height],
            "input_size": [args.input_height, args.input_width],
            "backend": args.backend,
            "device": runtime.device,
            "preprocess": args.preprocess,
            "inference_performed": scheduled,
            "input_sha256": input_hash,
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_elapsed,
            "postprocess_ms": postprocess_ms,
            "detections": detections,
            "tracks": tracks,
        }
        rows.append(row)

        elapsed = max(1e-9, time.perf_counter() - started)
        offline_fps = (frame_index + 1) / elapsed
        annotated = None
        if writer is not None or args.keyframes_dir:
            annotated = _annotate(
                frame, frame_index=frame_index, timestamp_s=timestamp_s,
                model_name=model_label, infer_ms=inference_elapsed,
                offline_fps=offline_fps, detections=detections, tracks=tracks,
                schedule=args.schedule,
            )
        if writer is not None:
            writer.write(annotated)

        if args.keyframes_dir and annotated is not None:
            keyframe_root = Path(args.keyframes_dir).resolve()
            events = []
            if first_detection and first_detection["frame"] == frame_index:
                events.append("first_detection")
            if first_trusted_detection and first_trusted_detection["frame"] == frame_index:
                events.append("first_trusted_detection")
            if len(detections) >= 2:
                events.append("multiple_lights")
            if any(item["color"] == TrafficLightState.UNKNOWN.value for item in detections):
                events.append("color_unknown")
            for event in events:
                if event in event_saved:
                    continue
                target = keyframe_root / event / f"{video_path.stem}_frame_{frame_index:06d}.jpg"
                _keyframe(target, annotated, args.overwrite)
                event_saved.add(event)

        csv_rows.append({
            "video": video_path.name,
            "frame": frame_index,
            "video_time_ms": round(timestamp_s * 1000.0, 3),
            "backend": args.backend,
            "schedule": args.schedule,
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_elapsed,
            "postprocess_ms": postprocess_ms,
            "offline_fps": round(offline_fps, 4),
        })

    capture.release()
    if writer is not None:
        writer.release()
    total_elapsed = time.perf_counter() - started

    stable_stats = {}
    for track_id, boxes in track_boxes.items():
        if len(boxes) < 2:
            continue
        values = np.asarray(boxes, dtype=np.float64)
        stable_stats[str(track_id)] = {
            "tracked_frames": len(boxes),
            "bbox_coordinate_std_px": [round(float(value), 3) for value in values.std(axis=0)],
        }

    summary = {
        "video": str(video_path),
        "video_name": video_path.name,
        "model": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "backend": args.backend,
        "runtime_providers": runtime.providers,
        "device": runtime.device,
        "preprocess": args.preprocess,
        "thresholds": {
            "confidence": args.conf,
            "nms_iou": args.iou,
            "max_det": args.max_det,
        },
        "roi_top_fraction": args.roi_top,
        "frame_size": [frame_width, frame_height],
        "input_size": [args.input_height, args.input_width],
        "source_fps": fps,
        "metadata_frame_count_hint": frame_count_hint,
        "decoded_frames": len(rows),
        "frames_processed": len(rows),
        "schedule": args.schedule,
        "inference_frames": len(inference_ms),
        "frames_with_new_detections": sum(presence),
        "total_new_detections": sum(color_counts.values()),
        "detection_color_counts": {color: color_counts.get(color, 0) for color in ("RED", "GREEN", "YELLOW", "UNKNOWN")},
        "tracked_color_frame_counts": {color: track_color_counts.get(color, 0) for color in ("RED", "GREEN", "YELLOW", "UNKNOWN")},
        "first_detection": first_detection,
        "first_trusted_detection_conf_ge_0_25": first_trusted_detection,
        "new_detection_runs": _runs(presence, fps),
        "mean_inference_ms": statistics.fmean(inference_ms) if inference_ms else None,
        "p95_inference_ms": float(np.percentile(inference_ms, 95)) if inference_ms else None,
        "processed_seconds": round(total_elapsed, 3),
        "offline_processing_fps": len(rows) / total_elapsed if total_elapsed else None,
        "track_bbox_coordinate_std_px": stable_stats,
        "outputs": {
            "annotated_video": str(output_path) if args.visualize else None,
            "timeline_json": str(json_path) if args.save_json else None,
            "runtime_csv": str(csv_path),
        },
    }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(csv_rows[0]) if csv_rows else ["video", "frame"]
        writer_csv = csv.DictWriter(handle, fieldnames=fields)
        writer_csv.writeheader()
        writer_csv.writerows(csv_rows)
    if args.save_json:
        _save_json(json_path, {"summary": summary, "frames": rows}, args.overwrite)
    if environment_path is not None:
        _save_json(environment_path, _environment(runtime), args.overwrite)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="source video file")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=Path, help="Ultralytics .pt or fixed-shape .onnx")
    model_group.add_argument("--model-manifest", type=Path, help="resolve the original model by SHA-256")
    parser.add_argument("--backend", choices=("pytorch", "onnx"), required=True)
    parser.add_argument("--output", required=True, help="annotated MP4 destination")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--preprocess", choices=("letterbox", "resize"), default="letterbox")
    parser.add_argument("--visualize", action="store_true", help="write the annotated MP4")
    parser.add_argument("--save-json", action="store_true", help="write per-frame detections and summary JSON")
    parser.add_argument("--json-path", help="optional JSON destination; defaults beside --output")
    parser.add_argument("--runtime-csv", help="optional per-frame runtime CSV destination")
    parser.add_argument("--environment-json", help="write Python, CUDA, OpenCV and ONNX Runtime versions")
    parser.add_argument("--model-label", help="display label written into the annotated video")
    parser.add_argument("--keyframes-dir", help="save first-detection, multi-light, and UNKNOWN keyframes")
    parser.add_argument("--schedule", choices=("every-frame", "1hz"), default="every-frame")
    parser.add_argument("--device", default="auto", help="PyTorch device, or cpu for ONNX Runtime")
    parser.add_argument("--roi-top", type=float, default=0.60)
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--input-height", type=int, default=192)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true", help="replace existing output files")
    parser.add_argument("--dry-run", action="store_true", help="open dependencies and validate paths/shapes only")
    args = parser.parse_args()
    if not 0 <= args.conf <= 1 or not 0 < args.iou <= 1:
        parser.error("--conf must be in [0,1] and --iou in (0,1]")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.dry_run:
        model_path = resolve_model_path(
            backend=parsed.backend, model_path=parsed.model, manifest_path=parsed.model_manifest
        )
        runtime = ModelRuntime(parsed.backend, model_path, device=parsed.device)
        runtime.validate_canvas(parsed.input_width, parsed.input_height)
        print(json.dumps({"model": str(model_path), "backend": parsed.backend, "device": runtime.device, "providers": runtime.providers}, indent=2))
    else:
        run(parsed)
