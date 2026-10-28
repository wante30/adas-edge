"""Benchmark one real front video through a PC or RKNN detector.

The output is deliberately evidence-oriented: annotated video, per-frame CSV,
and a JSON summary containing model/input provenance, inference latency, empty
frames, and traffic-light state estimates. It does not claim FPGA or PCIe
coverage; those stages are outside this process.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from tools.adas_hmi_sim.backend.traffic_light_runtime import TrafficLightTemporalFilter
    from tools.adas_hmi_sim.backend.lane_runtime import FrontLaneCandidate
    from tools.adas_hmi_sim.backend.traffic_light_specialist import TrafficLightSpecialist, SignalCandidateCache, signal_refresh_due
except ModuleNotFoundError:  # board-side standalone benchmark
    from traffic_light_runtime import TrafficLightTemporalFilter
    from lane_runtime import FrontLaneCandidate
    from traffic_light_specialist import TrafficLightSpecialist, SignalCandidateCache, signal_refresh_due


def _shape(value: str) -> int | tuple[int, int]:
    text = value.lower().replace("×", "x")
    if "x" not in text:
        return int(text)
    width, height = text.split("x", 1)
    return int(width), int(height)


def _detector(model: Path, input_size, confidence: float, device: str):
    if model.suffix.lower() == ".rknn":
        from competition_ai.detector import CompetitionDetector

        return CompetitionDetector(model, input_size=input_size, conf_threshold=confidence)
    from adas_edge.perception.ultralytics_detector import UltralyticsDetector

    return UltralyticsDetector(model, confidence=confidence, input_size=input_size, device=device, exact_input_canvas=True)


def _row(item) -> dict[str, object]:
    value = item.to_dict() if hasattr(item, "to_dict") else dict(item)
    bbox = value.get("bbox") or value.get("bbox_xyxy")
    if bbox is None:
        bbox = [value.get(key, 0) for key in ("x1", "y1", "x2", "y2")]
    return {
        "class_name": str(value.get("class_name", "unknown")),
        "confidence": float(value.get("confidence", 0.0)),
        "bbox": [float(v) for v in (bbox or (0, 0, 0, 0))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-size", default="512x288")
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--traffic-light-min-confidence", type=float, default=0.25)
    parser.add_argument("--traffic-light-model", type=Path, default=None, help="dedicated .pt/.rknn specialist; uses the frozen top-60%% 512x192 contract")
    parser.add_argument("--traffic-light-confidence", type=float, default=0.05)
    parser.add_argument("--traffic-light-fps", type=float, default=3.0)
    parser.add_argument("--traffic-light-roi-top", type=float, default=0.60)
    parser.add_argument("--traffic-light-roi-mode", choices=("top60", "adaptive"), default="top60")
    parser.add_argument("--traffic-light-specialist-mode", choices=("fuse", "replace"), default="fuse")
    parser.add_argument("--signal-roi", action="store_true", help="PC .pt only: add a second enlarged upper-signal pass")
    parser.add_argument("--signal-roi-confidence", type=float, default=0.10, help="PC signal ROI detector threshold")
    parser.add_argument("--signal-roi-input-size", default="640x384", help="PC signal ROI detector canvas")
    parser.add_argument("--signal-roi-top", type=float, default=0.55, help="PC signal ROI top-of-frame fraction")
    parser.add_argument("--traffic-light-hold-frames", type=int, default=4)
    parser.add_argument("--traffic-light-association", choices=("unverified", "geometry-experimental"), default="unverified", help="optional experimental signal-to-ego-lane geometry gate")
    parser.add_argument("--lane-stride", type=int, default=1, help="run CPU lane candidate every N frames")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--start-frame", type=int, default=0, help="seek to this source frame before benchmarking")
    parser.add_argument("--loop-video", action="store_true", help="loop the input when EOF is reached; useful for board stability runs")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.video.exists():
        raise SystemExit(f"missing video: {args.video}")
    if not args.model.exists():
        raise SystemExit(f"missing model: {args.model}")
    if args.traffic_light_model is not None and not args.traffic_light_model.exists():
        raise SystemExit(f"missing traffic-light specialist: {args.traffic_light_model}")
    input_size = _shape(args.input_size)
    signal_roi_input_size = _shape(args.signal_roi_input_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {args.video}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    writer = cv2.VideoWriter(
        str(args.output_dir / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    detector = _detector(args.model, input_size, args.confidence, args.device)
    roi_detector = None
    if args.signal_roi and args.traffic_light_model is None:
        if args.model.suffix.lower() != ".pt":
            raise SystemExit("--signal-roi currently requires a .pt main model")
        roi_detector = _detector(args.model, signal_roi_input_size, args.signal_roi_confidence, args.device)
    specialist = TrafficLightSpecialist(
        args.traffic_light_model,
        confidence=args.traffic_light_confidence,
        roi_top=args.traffic_light_roi_top,
        device=args.device,
        roi_mode=args.traffic_light_roi_mode,
    ) if args.traffic_light_model is not None else None
    specialist_last_source_s = -1.0
    specialist_detections: list[dict[str, object]] = []
    signal_cache = SignalCandidateCache(confidence=args.traffic_light_confidence)
    specialist_latencies: list[float] = []
    specialist_updates = 0
    traffic = TrafficLightTemporalFilter(min_confidence=args.traffic_light_min_confidence, hold_frames=args.traffic_light_hold_frames, association_mode=args.traffic_light_association,
                                       specialist_min_confidence=args.traffic_light_confidence, ego_x_max=.95, pedestrian_hold_frames=0)
    lanes = FrontLaneCandidate(update_stride=args.lane_stride)
    rows: list[dict[str, object]] = []
    latencies: list[float] = []
    detector_latencies: list[float] = []
    state_counts: dict[str, int] = {}
    signal_type_counts: dict[str, int] = {}
    pedestrian_state_counts: dict[str, int] = {}
    association_counts: dict[str, int] = {}
    lane_coverage_counts = {"any": 0, "both": 0}
    class_counts: dict[str, int] = {}
    started = time.perf_counter()
    frame_id = 0
    video_loops = 0
    try:
        while args.max_frames <= 0 or frame_id < args.max_frames:
            ok, frame = cap.read()
            if not ok and args.loop_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.start_frame))
                traffic.reset()
                lanes.reset()
                specialist_detections = []
                signal_cache.reset()
                if specialist is not None:
                    specialist.reset_roi()
                specialist_last_source_s = -1.0
                ok, frame = cap.read()
                video_loops += 1
            if not ok:
                break
            tick = time.perf_counter()
            if hasattr(detector, "infer"):
                result = detector.infer(frame, frame_id=frame_id, timestamp_ns=int(frame_id * 1_000_000_000 / source_fps))
            else:
                from adas_edge.contracts import Frame, PixelFormat

                detect_started = time.perf_counter()
                detections_core = detector.detect(Frame("front", frame_id, datetime.now(timezone.utc), frame, PixelFormat.BGR888))
                result = SimpleNamespace(
                    detections=detections_core,
                    inference_ms=(time.perf_counter() - detect_started) * 1000.0,
                )
            detections = [_row(item) for item in result.detections]
            signal_roi_ms = None
            if roi_detector is not None:
                from adas_edge.contracts import Frame, PixelFormat

                roi_started = time.perf_counter()
                frame_h, frame_w = frame.shape[:2]
                x0, x1 = int(frame_w * 0.10), int(frame_w * 0.90)
                y0, y1 = 0, int(frame_h * args.signal_roi_top)
                crop = frame[y0:y1, x0:x1]
                roi_frame = Frame("front_signal_roi", frame_id, datetime.now(timezone.utc), crop, PixelFormat.BGR888)
                roi_raw = roi_detector.detect(roi_frame)
                for item in roi_raw:
                    mapped = _row(item)
                    if mapped["class_name"] != "traffic light":
                        continue
                    rx1, ry1, rx2, ry2 = mapped["bbox"]
                    mapped["bbox"] = [
                        x0 + rx1 * (x1 - x0) / max(1, crop.shape[1]),
                        y0 + ry1 * (y1 - y0) / max(1, crop.shape[0]),
                        x0 + rx2 * (x1 - x0) / max(1, crop.shape[1]),
                        y0 + ry2 * (y1 - y0) / max(1, crop.shape[0]),
                    ]
                    detections.append(mapped)
                signal_roi_ms = round((time.perf_counter() - roi_started) * 1000.0, 3)
            specialist_updated = False
            specialist_ms = None
            if specialist is not None:
                source_s = frame_id / max(1.0, source_fps)
                signal_due = signal_refresh_due(source_s, None if specialist_last_source_s < 0. else specialist_last_source_s, args.traffic_light_fps)
                if args.traffic_light_specialist_mode == "replace":
                    detections = [
                        item for item in detections
                        if str(item.get("class_name", "")).replace("_", " ").lower() != "traffic light"
                    ]
                if signal_due:
                    specialist_detections, specialist_ms = specialist.detect(frame, frame_id=frame_id)
                    signal_cache.store(frame, specialist_detections, frame_id=frame_id,
                                       timestamp_ms=source_s * 1000.)
                    specialist_last_source_s = source_s
                    specialist_latencies.append(float(specialist_ms))
                    specialist_updates += 1
                    specialist_updated = True
                detections.extend(signal_cache.current(source_s * 1000.))
            lane_estimate = lanes.update(frame)
            estimate = traffic.update(frame, detections, lane_metadata=lane_estimate.as_metadata())
            latency = (time.perf_counter() - tick) * 1000.0
            latencies.append(latency)
            detector_latency = float(getattr(result, "inference_ms", latency))
            detector_latencies.append(detector_latency)
            state_counts[estimate.state] = state_counts.get(estimate.state, 0) + 1
            signal_type_counts[estimate.signal_type] = signal_type_counts.get(estimate.signal_type, 0) + 1
            pedestrian_state_counts[estimate.pedestrian_state] = pedestrian_state_counts.get(estimate.pedestrian_state, 0) + 1
            association_counts[estimate.association] = association_counts.get(estimate.association, 0) + 1
            if lane_estimate.coverage > 0:
                lane_coverage_counts["any"] += 1
            if lane_estimate.coverage >= 1.0:
                lane_coverage_counts["both"] += 1
            for item in detections:
                name = str(item["class_name"])
                class_counts[name] = class_counts.get(name, 0) + 1
                x1, y1, x2, y2 = [int(round(value)) for value in item["bbox"]]
                color = (0, 255, 255) if name == "traffic light" else (220, 220, 220)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} {float(item['confidence']):.2f}", (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cv2.putText(frame, f"TL {estimate.state} / {estimate.association}  OBJ {len(detections)}  {latency:.1f} ms", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255) if estimate.state != "UNKNOWN" else (180, 190, 190), 2, cv2.LINE_AA)
            writer.write(frame)
            rows.append({
                "frame_id": frame_id,
                "timestamp_ms": round(frame_id * 1000.0 / source_fps, 3),
                "detections": len(detections),
                "traffic_light_state": estimate.state,
                "traffic_light_raw_state": estimate.raw_state,
                "traffic_light_confidence": estimate.confidence,
                "traffic_light_candidates": estimate.candidate_count,
                "traffic_light_decision_state": estimate.decision_state,
                "traffic_light_association": estimate.association,
                "traffic_light_association_confidence": estimate.association_confidence,
                "traffic_light_signal_type": estimate.signal_type,
                "traffic_light_pedestrian_state": estimate.pedestrian_state,
                "traffic_light_vehicle_candidates": estimate.vehicle_candidate_count,
                "traffic_light_pedestrian_candidates": estimate.pedestrian_candidate_count,
                "lane_coverage": lane_estimate.coverage,
                "lane_source": lane_estimate.source,
                "inference_ms": detector_latency,
                "perception_ms": latency,
                "traffic_light_specialist_ms": specialist_ms,
                "traffic_light_specialist_updated": specialist_updated,
                "traffic_light_signal_roi_ms": signal_roi_ms,
            })
            frame_id += 1
    finally:
        cap.release()
        writer.release()
        if hasattr(detector, "close"):
            detector.close()
        if roi_detector is not None and hasattr(roi_detector, "close"):
            roi_detector.close()
        if specialist is not None:
            specialist.close()

    elapsed = max(1e-6, time.perf_counter() - started)
    summary = {
        "video": str(args.video.resolve()),
        "model": str(args.model.resolve()),
        "input_size": list(input_size) if isinstance(input_size, tuple) else input_size,
        "confidence": args.confidence,
        "traffic_light_min_confidence": args.traffic_light_min_confidence,
        "traffic_light_specialist": {
            "enabled": specialist is not None,
            "model": str(args.traffic_light_model.resolve()) if args.traffic_light_model is not None else None,
            "confidence": args.traffic_light_confidence,
            "fps_target": args.traffic_light_fps if specialist is not None else 0.0,
            "roi_top": args.traffic_light_roi_top if specialist is not None else None,
            "roi_mode": args.traffic_light_roi_mode if specialist is not None else None,
            "mode": args.traffic_light_specialist_mode if specialist is not None else None,
            "updates": specialist_updates,
            "inference_ms": {
                "mean": statistics.fmean(specialist_latencies) if specialist_latencies else 0.0,
                "p50": statistics.median(specialist_latencies) if specialist_latencies else 0.0,
                "p95": sorted(specialist_latencies)[max(0, int(len(specialist_latencies) * 0.95) - 1)] if specialist_latencies else 0.0,
            },
        },
        "signal_roi": {
            "enabled": roi_detector is not None,
            "x_range": [0.10, 0.90] if roi_detector is not None else None,
            "y_range": [0.0, args.signal_roi_top] if roi_detector is not None else None,
            "confidence": args.signal_roi_confidence if roi_detector is not None else None,
            "input_size": list(signal_roi_input_size) if isinstance(signal_roi_input_size, tuple) and roi_detector is not None else signal_roi_input_size if roi_detector is not None else None,
        },
        "traffic_light_hold_frames": args.traffic_light_hold_frames,
        "traffic_light_association": args.traffic_light_association,
        "source_fps": source_fps,
        "source_size": [width, height],
        "source_start_frame": max(0, args.start_frame),
        "loop_video": bool(args.loop_video),
        "video_loops": video_loops,
        "frames": frame_id,
        "wall_seconds": elapsed,
        "wall_fps": frame_id / elapsed if frame_id else 0.0,
        "empty_frame_ratio": sum(1 for row in rows if row["detections"] == 0) / max(1, frame_id),
        "inference_ms": {
            "mean": statistics.fmean(detector_latencies) if detector_latencies else 0.0,
            "p50": statistics.median(detector_latencies) if detector_latencies else 0.0,
            "p95": sorted(detector_latencies)[max(0, int(len(detector_latencies) * 0.95) - 1)] if detector_latencies else 0.0,
        },
        "perception_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": statistics.median(latencies) if latencies else 0.0,
            "p95": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0.0,
        },
        "traffic_light_states": state_counts,
        "traffic_light_signal_types": signal_type_counts,
        "traffic_light_pedestrian_states": pedestrian_state_counts,
        "traffic_light_association_states": association_counts,
        "traffic_light_validation": "EXPERIMENTAL_GEOMETRY_ASSOCIATION" if args.traffic_light_association == "geometry-experimental" else "EXPERIMENTAL_COLOR_CANDIDATES_UNASSOCIATED_WITH_EGO_LANE",
        "lane_coverage": {
            "any_side_ratio": lane_coverage_counts["any"] / max(1, frame_id),
            "both_sides_ratio": lane_coverage_counts["both"] / max(1, frame_id),
            "source": "OPENCV_HOUGH_EXPERIMENTAL",
            "validation": "DISPLAY_CANDIDATE_ONLY",
            "update_stride": args.lane_stride,
        },
        "detections_by_class": class_counts,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0]) if rows else ["frame_id"]
        writer_csv = csv.DictWriter(handle, fieldnames=fields)
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

