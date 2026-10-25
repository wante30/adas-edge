"""B-05O: compare scheduled runs against the 30 Hz dense-inference reference.

The 30 Hz run is an offline upper bound, not RK3566 behaviour and not ground
truth.  Agreement numbers below answer "does downsampling lose boxes that
dense inference finds", nothing more.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.b05k_common import box_iou


def tl_by_frame(events: list[dict], source: str) -> dict[int, list[dict]]:
    table: dict[int, list[dict]] = {}
    for event in events:
        dets = [d for d in event["detections"] if d.get("source") == source]
        if dets:
            table[int(event["frame"])] = dets
    return table


def match(dets: list[dict], refs: list[dict], iou_thr: float = 0.5) -> tuple[int, int, int, list[dict]]:
    used: set[int] = set()
    tp = 0
    pairs = []
    for det in sorted(dets, key=lambda x: -x["confidence"]):
        best, best_iou = None, 0.0
        for index, ref in enumerate(refs):
            if index in used:
                continue
            overlap = box_iou(det["bbox"], ref["bbox"])
            if overlap > best_iou:
                best_iou, best = overlap, index
        if best is not None and best_iou >= iou_thr:
            used.add(best)
            tp += 1
            pairs.append((det, refs[best], best_iou))
    return tp, len(dets) - tp, len(refs) - tp, pairs


def build_tracks(table: dict[int, list[dict]], fps: float, gap_s: float = 0.5, iou_thr: float = 0.5) -> list[dict]:
    tracks: list[dict] = []
    active: list[dict] = []
    for frame in sorted(table):
        used: set[int] = set()
        for det in sorted(table[frame], key=lambda x: -x["confidence"]):
            best, best_iou = None, 0.0
            for index, track in enumerate(active):
                if index in used:
                    continue
                overlap = box_iou(det["bbox"], track["last_bbox"])
                if overlap > best_iou:
                    best_iou, best = overlap, index
            if best is not None and best_iou >= iou_thr:
                used.add(best)
                track = active[best]
                track["frames"][frame] = det["bbox"]
                track["last_bbox"] = det["bbox"]
                track["last_frame"] = frame
            else:
                used.add(len(active))
                active.append({"start_frame": frame, "last_frame": frame, "last_bbox": det["bbox"],
                               "frames": {frame: det["bbox"]}})
        still = []
        for track in active:
            (still if (frame - track["last_frame"]) / fps <= gap_s else tracks).append(track)
        active = still
    tracks.extend(active)
    for track in tracks:
        track["start_s"] = track["start_frame"] / fps
        track["end_s"] = track["last_frame"] / fps
        track["duration_s"] = track["end_s"] - track["start_s"]
    return tracks


def evaluate(sched: dict[str, Any], upper: dict[str, Any], source: str, frames_limit: int | None = None) -> dict[str, Any]:
    fps = sched["fps"]
    ref = tl_by_frame(upper["events"], source)
    sched_table = tl_by_frame(sched["events"], source)
    tp = fp = fn = 0
    per_frame = []
    for frame in sorted(sched_table):
        if frames_limit is not None and frame > frames_limit:
            continue
        refs = ref.get(frame, [])
        if not refs and not sched_table[frame]:
            continue
        a, b, c, _ = match(sched_table[frame], refs)
        tp += a
        fp += b
        fn += c
        if refs:
            per_frame.append(a / len(refs))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    tracks = build_tracks(ref, fps)
    hit, delays = 0, []
    for track in tracks:
        first = None
        for frame, bbox in track["frames"].items():
            for det in sched_table.get(frame, []):
                if box_iou(det["bbox"], bbox) >= 0.5:
                    first = frame
                    break
            if first is not None:
                break
        if first is None:
            continue
        hit += 1
        delays.append((first - track["start_frame"]) / fps)
    long_tracks = [t for t in tracks if t["duration_s"] >= 0.2]
    long_hit, long_delays = 0, []
    for track in long_tracks:
        first = None
        for frame, bbox in track["frames"].items():
            for det in sched_table.get(frame, []):
                if box_iou(det["bbox"], bbox) >= 0.5:
                    first = frame
                    break
            if first is not None:
                break
        if first is None:
            continue
        long_hit += 1
        long_delays.append((first - track["start_frame"]) / fps)
    return {
        "source": source,
        "frames_compared": len(sched_table),
        "box_tp": tp, "box_fp": fp, "box_fn": fn,
        "precision_vs_dense": precision, "recall_vs_dense": recall,
        "reference_tracks": len(tracks),
        "tracks_hit": hit, "tracks_missed": len(tracks) - hit,
        "delay_s": {"p50": float(np.percentile(delays, 50)) if delays else None,
                    "p95": float(np.percentile(delays, 95)) if delays else None,
                    "max": float(np.max(delays)) if delays else None},
        "long_tracks_ge_0.2s": len(long_tracks),
        "long_tracks_hit": long_hit,
        "long_track_miss_rate": (len(long_tracks) - long_hit) / len(long_tracks) if long_tracks else None,
        "long_delay_s": {"p50": float(np.percentile(long_delays, 50)) if long_delays else None,
                         "p95": float(np.percentile(long_delays, 95)) if long_delays else None,
                         "max": float(np.max(long_delays)) if long_delays else None},
    }


def window_stats(sched: dict[str, Any], upper: dict[str, Any], start: float, end: float) -> dict[str, Any]:
    fps = sched["fps"]
    output: dict[str, Any] = {"window": [start, end], "schemes": {}, "dense": {}}
    total_frames = int(round((end - start) * fps))
    for scheme, source in (("bdd_10hz", "bdd"), ("p2_10hz", "p2"), ("dual_10_5hz", "spec")):
        table = tl_by_frame(sched["events"], source)
        frames = [f for f in sorted(table) if start * fps <= f < end * fps]
        covered = len(frames)
        gaps, previous = [], None
        for frame in range(int(start * fps), int(end * fps)):
            if frame in table:
                if previous is not None:
                    gaps.append((frame - previous) / fps)
                previous = frame
        output["schemes"][scheme] = {
            "display_frames_with_tl": covered,
            "display_coverage": covered / total_frames if total_frames else None,
            "first_tl_s": frames[0] / fps if frames else None,
            "max_gap_s": max(gaps) if gaps else None,
        }
    for source in ("bdd", "p2", "spec"):
        table = tl_by_frame(upper["events"], source)
        frames = [f for f in sorted(table) if start * fps <= f < end * fps]
        output["dense"][source] = {"frames_with_tl": len(frames),
                                   "coverage": len(frames) / total_frames if total_frames else None,
                                   "first_tl_s": frames[0] / fps if frames else None}
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scheduled", type=Path, required=True)
    ap.add_argument("--upper", type=Path, required=True)
    ap.add_argument("--windows", type=float, nargs="*", default=[80.5, 82.0])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    sched = json.loads(args.scheduled.read_text(encoding="utf-8"))
    upper = json.loads(args.upper.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "video": sched["video"], "frames": sched["frames"], "fps": sched["fps"],
        "scheduled_rates": sched["rates"],
        "scheme_total_forward_per_second": sched["scheme_total_forward_per_second"],
        "reference": "30 Hz dense inference (offline upper bound, not ground truth)",
        "agreement": {},
    }
    for source in ("bdd", "p2", "spec"):
        result["agreement"][source] = evaluate(sched, upper, source)
    if len(args.windows) == 2:
        result["window"] = window_stats(sched, upper, args.windows[0], args.windows[1])
    result["key_events"] = sched.get("key_events", [])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "key_events"}, indent=2))


if __name__ == "__main__":
    main()
