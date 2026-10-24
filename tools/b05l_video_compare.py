"""B-05L compare per-frame video timelines from the three Specialists.

This script only reads B-05K/B-05L timeline JSON.  Without human video GT, a
"model-only" frame is a candidate difference, never a confirmed false positive
or miss.  Colors are the unchanged HSV observations, not color accuracy.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def runs(flags: list[bool], fps: float, min_length: int = 1) -> list[dict[str, Any]]:
    output = []
    start = None
    for index, value in enumerate(flags + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= min_length:
                output.append({"start_frame": start, "end_frame": index - 1,
                               "start_s": round(start / fps, 3), "end_s": round((index - 1) / fps, 3),
                               "frames": index - start})
            start = None
    return output


def summarize_model(frames: list[dict[str, Any]], label: str, fps: float) -> dict[str, Any]:
    any_flags = []
    trusted_flags = []
    color_counts = Counter()
    infer = []
    for frame in frames:
        detections = frame.get("detections", [])
        any_flags.append(bool(detections))
        trusted = [item for item in detections if float(item["confidence"]) >= 0.25]
        trusted_flags.append(bool(trusted))
        for item in detections:
            color_counts[str(item.get("color", "UNKNOWN"))] += 1
        if frame.get("inference_ms") is not None:
            infer.append(float(frame["inference_ms"]))
    trusted_runs = runs(trusted_flags, fps)
    any_runs = runs(any_flags, fps)
    first_any = next((i for i, value in enumerate(any_flags) if value), None)
    first_trusted = next((i for i, value in enumerate(trusted_flags) if value), None)
    first_stable = next((run for run in trusted_runs if run["frames"] >= 3), None)
    gaps = []
    previous_end = None
    for run in any_runs:
        if previous_end is not None:
            gaps.append({"start_s": round((previous_end + 1) / fps, 3),
                         "end_s": round(run["start_frame"] / fps, 3),
                         "seconds": round((run["start_frame"] - previous_end - 1) / fps, 3)})
        previous_end = run["end_frame"]
    return {
        "label": label,
        "frames_with_any_candidate": sum(any_flags),
        "frames_with_trusted_candidate": sum(trusted_flags),
        "first_any_candidate": None if first_any is None else {"frame": first_any, "seconds": round(first_any / fps, 3)},
        "first_trusted_candidate": None if first_trusted is None else {"frame": first_trusted, "seconds": round(first_trusted / fps, 3)},
        "first_stable_3frame_trusted": first_stable,
        "trusted_runs": trusted_runs,
        "any_candidate_runs": any_runs,
        "candidate_gaps": gaps,
        "longest_candidate_gap_s": max((gap["seconds"] for gap in gaps), default=None),
        "color_observation_counts": {color: color_counts.get(color, 0) for color in ("RED", "GREEN", "YELLOW", "UNKNOWN")},
        "mean_inference_ms": statistics.fmean(infer) if infer else None,
        "p95_inference_ms": float(np.percentile(infer, 95)) if infer else None,
    }


def pairwise(frames_a: list[dict[str, Any]], frames_b: list[dict[str, Any]], name_a: str, name_b: str, fps: float) -> dict[str, Any]:
    flags_a = []
    flags_b = []
    for frame_a, frame_b in zip(frames_a, frames_b):
        flags_a.append(any(float(item["confidence"]) >= 0.25 for item in frame_a.get("detections", [])))
        flags_b.append(any(float(item["confidence"]) >= 0.25 for item in frame_b.get("detections", [])))
    only_a = [a and not b for a, b in zip(flags_a, flags_b)]
    only_b = [b and not a for a, b in zip(flags_a, flags_b)]
    both = [a and b for a, b in zip(flags_a, flags_b)]
    neither = [not a and not b for a, b in zip(flags_a, flags_b)]
    return {
        "a": name_a,
        "b": name_b,
        "both": sum(both),
        "only_a": sum(only_a),
        "only_b": sum(only_b),
        "neither": sum(neither),
        "only_a_runs": runs(only_a, fps),
        "only_b_runs": runs(only_b, fps),
    }


def bbox_continuity(frames: list[dict[str, Any]]) -> dict[str, Any]:
    ious = []
    centers = []
    previous = []
    for frame in frames:
        current = [item["bbox"] for item in frame.get("detections", []) if float(item["confidence"]) >= 0.25]
        pairs = []
        used = set()
        for box in current:
            best = None
            best_iou = 0.0
            for index, old in enumerate(previous):
                if index in used:
                    continue
                overlap = iou(box, old)
                if overlap > best_iou:
                    best_iou, best = overlap, index
            if best is not None and best_iou >= 0.5:
                used.add(best)
                pairs.append((box, previous[best], best_iou))
        for box, old, overlap in pairs:
            ious.append(overlap)
            centers.append(abs((box[0] + box[2]) / 2 - (old[0] + old[2]) / 2))
        previous = current
    return {
        "matched_consecutive_pairs": len(ious),
        "mean_iou": float(np.mean(ious)) if ious else None,
        "p05_iou": float(np.percentile(ious, 5)) if ious else None,
        "mean_center_shift_px": float(np.mean(centers)) if centers else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timelines", type=Path, required=True, help="directory with <label>/<video>.json")
    parser.add_argument("--labels", nargs="+", default=["yolo11n", "v8n", "v8n_p2"])
    parser.add_argument("--baseline", default="yolo11n")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    videos = sorted({path.stem for path in (args.timelines / args.labels[0]).glob("*.json")})
    result = {"timelines": str(args.timelines.resolve()), "labels": args.labels, "videos": {}}
    for video in videos:
        data = {}
        for label in args.labels:
            path = args.timelines / label / f"{video}.json"
            if not path.exists():
                data[label] = None
                continue
            data[label] = json.loads(path.read_text(encoding="utf-8"))
        first = next((value for value in data.values() if value), None)
        if first is None:
            continue
        fps = float(first["summary"]["source_fps"])
        frames = {label: value["frames"] for label, value in data.items() if value}
        row = {
            "source_fps": fps,
            "decoded_frames": len(next(iter(frames.values()))),
            "duration_s": round(len(next(iter(frames.values()))) / fps, 3),
            "models": {label: summarize_model(items, label, fps) for label, items in frames.items()},
            "pairwise_vs_baseline": {},
            "bbox_continuity": {label: bbox_continuity(items) for label, items in frames.items()},
        }
        for label, items in frames.items():
            if label == args.baseline or label not in frames:
                continue
            row["pairwise_vs_baseline"][label] = pairwise(
                frames[args.baseline], items, args.baseline, label, fps
            )
        result["videos"][video] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({video: {
        "duration_s": row["duration_s"],
        "models": {label: {"first_trusted": value["first_trusted_candidate"],
                           "trusted_frames": value["frames_with_trusted_candidate"]}
                   for label, value in row["models"].items()},
    } for video, row in result["videos"].items()}, indent=2))


if __name__ == "__main__":
    main()
