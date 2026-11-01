"""Board replay through FrontPipeline with HazardFusion at 5 Hz."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from capi_detector import BDD_NAMES, CapiMtlDetector
from competition_ai.scenario import Lane
from tools.competition_console.pipeline import FrontPipeline

VIDEO = Path("/home/cat/adas-edge-rect/videos/video_20260929_113539.mp4")
MODEL = Path("/home/cat/models/b05n_p2_mtl_b2a_seg256/"
             "b05n_p2_mtl_b2a_seg256_int8.rknn")


def stats(values):
    values = sorted(values)
    if not values:
        return {}
    return {"n": len(values), "mean": round(sum(values) / len(values), 2),
            "p50": round(values[len(values) // 2], 2),
            "p95": round(values[min(len(values) - 1,
                                    int(len(values) * .95))], 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=VIDEO)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--hazard-hz", type=float, default=5.0)
    parser.add_argument("--sync", action="store_true")
    args = parser.parse_args()
    cap = cv2.VideoCapture(str(args.video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    stride = max(1, int(round(fps / 10.0)))
    detector = CapiMtlDetector(str(MODEL), (512, 288), class_names=BDD_NAMES)
    pipeline = FrontPipeline(
        detector, None, main_hz=10.0, hazards=True,
        hazards_async=not args.sync, hazard_hz=args.hazard_hz,
        main_every_request=True)
    counts = Counter()
    flags = Counter()
    timeline = {}
    samples = 0
    started = time.perf_counter()
    last = None
    try:
        while samples < args.samples:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if frame_id % stride:
                continue
            timestamp_ns = int(frame_id / fps * 1e9)
            pipeline.advance(frame, frame_id, timestamp_ns)
            last = pipeline.snapshot(Lane.LEFT, timestamp_ns)
            for event in last.hazards.events:
                key = (event["kind"], event.get("track_id"))
                entry = timeline.setdefault(key, {
                    "kind": event["kind"], "track_id": event.get("track_id"),
                    "triggered": None, "cleared": None,
                    "assessment": event.get("assessment"),
                    "target_class": event.get("target_class"),
                    "evidence": event.get("evidence"),
                })
                if event.get("state") == "TRIGGERED":
                    entry["triggered"] = event.get("media_s")
                    counts[event["kind"]] += 1
                elif event.get("state") == "CLEARED":
                    entry["cleared"] = event.get("media_s")
            flags["pedestrian"] += int(bool(last.hazards.pedestrian_hazard))
            flags["obstacle"] += int(bool(last.hazards.obstacle_hazard))
            samples += 1
    finally:
        cap.release()
        detector.close()
    elapsed = time.perf_counter() - started
    timings = list(getattr(pipeline.hazard_engine, "timings", []))
    print(json.dumps({
        "samples": samples,
        "wall_s": round(elapsed, 3),
        "throughput_fps": round(samples / max(elapsed, 1e-6), 2),
        "hazard_hz": args.hazard_hz,
        "async": not args.sync,
        "hazard_timing_ms": stats(timings),
        "event_counts": dict(counts),
        "flag_frames": dict(flags),
        "last_reason": last.hazards.reason if last else None,
        "timeline": list(timeline.values()),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
