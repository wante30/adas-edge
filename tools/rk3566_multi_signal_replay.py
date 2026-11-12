#!/usr/bin/env python3
"""Replay a real video through the RK3566 INT8 detector and multi-signal stack.

The script is deliberately a standalone board acceptance tool.  It uses the
same RknnConsoleDetector/CompetitionDetector path as the AI sidecar, while
reading the highest-resolution decoded BGR frame available to OpenCV for the
native ROI classifier.  It never falls back to a PC ONNX model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from competition_ai.multi_signal import MultiSignalTracker, candidates_from_rows
from competition_ai.signal_shape import ContourTemplateBackend
from tools.competition_console.rknn_detector import RknnConsoleDetector


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    values = [float(v) for v in values]
    return {
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
        "max": round(max(values), 4),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_board_sample() -> dict[str, object]:
    """Read cheap, read-only board telemetry when the Linux interfaces exist."""
    sample: dict[str, object] = {"time_ns": time.monotonic_ns()}
    try:
        sample["loadavg_1m"] = float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        pass
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                sample["rss_kb"] = int(line.split()[1])
                break
    except (OSError, ValueError, IndexError):
        pass
    npu_values = []
    for path in (Path("/sys/class/devfreq/fde40000.npu/load"),
                 Path("/sys/class/devfreq/fde60000.npu/load")):
        try:
            text = path.read_text().strip()
            npu_values.append({"path": str(path), "value": text})
        except OSError:
            continue
    if npu_values:
        sample["npu_load"] = npu_values
    return sample


def shape_label(value: str) -> str:
    return {
        "LEFT_ARROW": "LEFT", "STRAIGHT_ARROW": "STRAIGHT",
        "RIGHT_ARROW": "RIGHT", "UTURN_ARROW": "UTURN",
        "ROUND": "ROUND", "UNKNOWN": "UNKNOWN",
    }.get(str(value), "UNKNOWN")


def observed_directions(shape: str) -> list[str]:
    return {"LEFT_ARROW": ["left"], "STRAIGHT_ARROW": ["straight"],
            "RIGHT_ARROW": ["right"], "UTURN_ARROW": ["uturn"]}.get(shape, [])


def draw_signals(frame: np.ndarray, signals: list[dict], *, updated: bool,
                 main_updated: bool, specialist_updated: bool, source_fps: float,
                 update_fps: float) -> None:
    colors = {"RED": (40, 40, 235), "YELLOW": (30, 220, 240),
              "GREEN": (40, 210, 40), "UNKNOWN": (180, 180, 180)}
    for signal in signals:
        x1, y1, x2, y2 = [int(round(v)) for v in signal.get("bbox", (0, 0, 0, 0))]
        if x2 <= x1 or y2 <= y1:
            continue
        color_name = str(signal.get("color", "UNKNOWN"))
        shape = shape_label(signal.get("shape", "UNKNOWN"))
        valid = bool(signal.get("valid", False))
        line = colors.get(color_name, colors["UNKNOWN"])
        if not valid:
            line = (110, 110, 110)
        cv2.rectangle(frame, (x1, y1), (x2, y2), line, 2)
        text = (f"TL#{signal.get('track_id', '?')} {color_name} {shape} "
                f"c={float(signal.get('confidence', 0.0)):.2f} "
                f"{'VALID' if valid else 'STALE'}")
        baseline = max(16, y1 - 5)
        cv2.putText(frame, text, (x1, baseline), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, line, 1, cv2.LINE_AA)
    footer = (f"RKNN INT8 | src={source_fps:.2f}fps | result={update_fps:.2f}fps "
              f"| update={'Y' if updated else 'N'} "
              f"M={'Y' if main_updated else 'N'} S={'Y' if specialist_updated else 'N'}")
    cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 900), 25), (0, 0, 0), -1)
    cv2.putText(frame, footer, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (230, 230, 230), 1, cv2.LINE_AA)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="real video decoded by OpenCV/GStreamer")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--main-model", required=True)
    parser.add_argument("--specialist-model")
    parser.add_argument("--class-profile", default="bdd10")
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--input-height", type=int, default=288)
    parser.add_argument("--main-hz", type=float, default=10.0)
    parser.add_argument("--specialist-hz", type=float, default=2.0)
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--specialist-conf-threshold", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--specialist-nms-threshold", type=float, default=0.70)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--failure-dir")
    parser.add_argument("--codec", default="mp4v")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if not 0 < args.main_hz or not 0 < args.specialist_hz:
        raise SystemExit("model update frequencies must be positive")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "rk3566_multi_signal.jsonl"
    metrics_path = out_dir / "rk3566_performance.json"
    csv_path = out_dir / "rk3566_performance.csv"
    video_path = out_dir / "rk3566_multi_signal.mp4"
    failure_dir = Path(args.failure_dir) if args.failure_dir else out_dir / "failures"

    input_path = Path(args.input)
    main_path = Path(args.main_model)
    specialist_path = Path(args.specialist_model) if args.specialist_model else None
    for path in (input_path, main_path, specialist_path):
        if path is not None and not path.is_file():
            raise SystemExit(f"missing input/model: {path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {input_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(source_fps) or source_fps <= 0:
        source_fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise SystemExit("video has no usable dimensions")
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*args.codec),
                             source_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"cannot open video writer: {video_path}")

    main_detector = RknnConsoleDetector(
        main_path, (args.input_width, args.input_height), class_profile=args.class_profile,
        include_traffic_lights=True, confidence_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold)
    specialist_detector = None
    if specialist_path:
        specialist_detector = RknnConsoleDetector(
            specialist_path, (args.input_width, args.input_height), specialist=True,
            class_names=("traffic light",), include_traffic_lights=True,
            confidence_threshold=args.specialist_conf_threshold,
            nms_threshold=args.specialist_nms_threshold)
    tracker = MultiSignalTracker(backend=ContourTemplateBackend())
    context = ("RK3566_REPLAY", main_path.name, specialist_path.name if specialist_path else "")
    timing: dict[str, list[float]] = {
        "npu_ms": [], "decode_nms_ms": [], "preprocess_ms": [], "detector_total_ms": [],
        "color_ms": [], "arrow_ms": [], "tracking_ms": [], "pipeline_ms": [],
        "frame_wall_ms": [], "frame_cpu_ms": [],
    }
    telemetry: list[dict[str, object]] = []
    rows_for_csv: list[dict[str, object]] = []
    frames = 0
    main_updates = specialist_updates = result_updates = 0
    last_main_t = last_specialist_t = -1e9
    start_wall = time.perf_counter()
    last_frame_wall = start_wall
    snapshots = []
    failure_count = 0

    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_started = time.perf_counter_ns()
            cpu_started = time.thread_time_ns()
            timestamp_ns = int(round(frames / source_fps * 1e9))
            media_t = frames / source_fps
            main_due = frames == 0 or media_t - last_main_t >= 1.0 / args.main_hz - 1e-9
            specialist_due = (specialist_detector is not None and
                              (frames == 0 or media_t - last_specialist_t >= 1.0 / args.specialist_hz - 1e-9))
            all_candidates = []
            per_frame_npu = per_frame_decode = per_frame_pre = 0.0
            per_frame_detector = 0.0
            if main_due:
                detector_started = time.perf_counter_ns()
                main_rows, _ = main_detector.detect(frame)
                detector_elapsed = (time.perf_counter_ns() - detector_started) / 1e6
                last_main_t = media_t
                main_updates += 1
                all_candidates.extend(candidates_from_rows(
                    main_rows, frame_id=frames, timestamp_ns=timestamp_ns,
                    space="q4_native", width=width, height=height, detector="main"))
                last = main_detector.last_timing or {}
                per_frame_npu += float(last.get("rknn_call_ms", 0.0))
                per_frame_decode += float(last.get("decode_nms_ms", 0.0))
                per_frame_pre += float(last.get("preprocess_ms", 0.0))
                per_frame_detector += detector_elapsed
            if specialist_due and specialist_detector is not None:
                detector_started = time.perf_counter_ns()
                specialist_rows, _ = specialist_detector.detect(frame)
                detector_elapsed = (time.perf_counter_ns() - detector_started) / 1e6
                last_specialist_t = media_t
                specialist_updates += 1
                all_candidates.extend(candidates_from_rows(
                    specialist_rows, frame_id=frames, timestamp_ns=timestamp_ns,
                    space="q4_native", width=width, height=height, detector="specialist"))
                last = specialist_detector.last_timing or {}
                per_frame_npu += float(last.get("rknn_call_ms", 0.0))
                per_frame_decode += float(last.get("decode_nms_ms", 0.0))
                per_frame_pre += float(last.get("preprocess_ms", 0.0))
                per_frame_detector += detector_elapsed
            updated = bool(main_due or specialist_due)
            if updated:
                tracking_started = time.perf_counter_ns()
                tracker.update(frame, all_candidates, frame_id=frames,
                               timestamp_ns=timestamp_ns, context=context)
                tracking_ms = (time.perf_counter_ns() - tracking_started) / 1e6
                result_updates += 1
            else:
                tracking_ms = 0.0
            snapshot = tracker.snapshot(timestamp_ns, context=context, available=True)
            signals = snapshot.get("signals", [])
            for signal in signals:
                signal["observed_directions"] = observed_directions(str(signal.get("shape", "UNKNOWN")))
                signal["failure_reason"] = signal.get("reason", "")
            color_ms = sum(float(s.get("evidence", {}).get("timing_ms", {}).get("color_ms", 0.0)) for s in signals)
            arrow_ms = sum(float(s.get("evidence", {}).get("timing_ms", {}).get("shape_ms", 0.0)) for s in signals)
            if updated and signals and failure_count < 64:
                failure_dir.mkdir(parents=True, exist_ok=True)
                for signal in signals:
                    if signal.get("shape") != "UNKNOWN" and signal.get("valid"):
                        continue
                    x1, y1, x2, y2 = [int(round(v)) for v in signal.get("bbox", (0, 0, 0, 0))]
                    crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                    if crop.size == 0:
                        continue
                    stem = f"frame_{frames:06d}_track_{signal.get('track_id', 0)}"
                    ok_png, encoded = cv2.imencode(".png", crop)
                    if ok_png:
                        (failure_dir / f"{stem}.png").write_bytes(encoded.tobytes())
                        (failure_dir / f"{stem}.json").write_text(json.dumps(signal, ensure_ascii=False), encoding="utf-8")
                        failure_count += 1
            elapsed_ms = (time.perf_counter_ns() - frame_started) / 1e6
            cpu_ms = (time.thread_time_ns() - cpu_started) / 1e6
            timing["npu_ms"].append(per_frame_npu)
            timing["decode_nms_ms"].append(per_frame_decode)
            timing["preprocess_ms"].append(per_frame_pre)
            timing["detector_total_ms"].append(per_frame_detector)
            timing["color_ms"].append(color_ms)
            timing["arrow_ms"].append(arrow_ms)
            timing["tracking_ms"].append(tracking_ms)
            timing["pipeline_ms"].append(elapsed_ms)
            timing["frame_wall_ms"].append((time.perf_counter() - last_frame_wall) * 1000.0)
            timing["frame_cpu_ms"].append(cpu_ms)
            last_frame_wall = time.perf_counter()
            if frames % 30 == 0:
                telemetry.append(read_board_sample())
            update_fps = result_updates / max(1e-9, time.perf_counter() - start_wall)
            payload = {
                "schema_version": 1, "frame_id": frames, "timestamp_ns": timestamp_ns,
                "source_fps": source_fps, "display_fps": source_fps,
                "model_update_fps": update_fps, "detector_updated": updated,
                "main_updated": bool(main_due), "specialist_updated": bool(specialist_due),
                "signals": signals, "directions": snapshot.get("directions", {}),
                "rejections": snapshot.get("rejections", []),
                "metrics": {"npu_ms": per_frame_npu, "decode_nms_ms": per_frame_decode,
                             "preprocess_ms": per_frame_pre, "detector_ms": per_frame_detector,
                             "color_ms": color_ms, "arrow_ms": arrow_ms, "tracking_ms": tracking_ms,
                             "pipeline_ms": elapsed_ms, "cpu_ms": cpu_ms},
            }
            jsonl.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
            draw_signals(frame, signals, updated=updated, main_updated=bool(main_due),
                         specialist_updated=bool(specialist_due), source_fps=source_fps,
                         update_fps=update_fps)
            writer.write(frame)
            rows_for_csv.append({"frame_id": frames, "timestamp_ns": timestamp_ns,
                                 "detector_updated": int(updated), "signal_count": len(signals),
                                 "npu_ms": per_frame_npu, "decode_nms_ms": per_frame_decode,
                                 "color_ms": color_ms, "arrow_ms": arrow_ms,
                                 "tracking_ms": tracking_ms, "pipeline_ms": elapsed_ms,
                                 "cpu_ms": cpu_ms})
            frames += 1
            if args.max_frames is not None and frames >= args.max_frames:
                break
    cap.release()
    writer.release()
    main_detector.close()
    if specialist_detector is not None:
        specialist_detector.close()
    duration_s = max(1e-9, time.perf_counter() - start_wall)
    summary = {
        "board_measured": platform.machine().lower() in ("aarch64", "arm64", "armv8l"),
        "platform": {"system": platform.system(), "machine": platform.machine(), "release": platform.release()},
        "input": str(input_path), "output_video": str(video_path), "jsonl": str(jsonl_path),
        "source_fps": source_fps, "output_fps": source_fps, "width": width, "height": height,
        "frames": frames, "duration_s": round(duration_s, 4),
        "main_updates": main_updates, "specialist_updates": specialist_updates,
        "result_updates": result_updates, "actual_result_update_fps": round(result_updates / duration_s, 4),
        "configured_main_hz": args.main_hz, "configured_specialist_hz": args.specialist_hz,
        "confidence_threshold": args.conf_threshold, "nms_threshold": args.nms_threshold,
        "main_model": {"path": str(main_path), "sha256": sha256(main_path)},
        "specialist_model": ({"path": str(specialist_path), "sha256": sha256(specialist_path)}
                             if specialist_path else None),
        "roi_source": "decoded_native_bgr_frame",
        "roi_note": "No interpolation is used; ROI is cropped before any classifier analysis.",
        "timings_ms": {name: percentiles(values) for name, values in timing.items()},
        "telemetry_samples": telemetry,
        "failure_count": failure_count,
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer_csv = csv.DictWriter(stream, fieldnames=list(rows_for_csv[0]) if rows_for_csv else ["frame_id"])
        writer_csv.writeheader()
        writer_csv.writerows(rows_for_csv)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
