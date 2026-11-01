"""Board replay: front video -> C API MTL -> RoadScene -> HazardFusion."""
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
from tools.competition_console.hazard_scene_adapter import SceneHazardEngine

VIDEO = Path("/home/cat/adas-edge-rect/videos/ch0_front.mp4")
MODEL = Path("/home/cat/models/b05n_p2_mtl_b2a_seg256/"
             "b05n_p2_mtl_b2a_seg256_int8.rknn")
OUT = ROOT / "results" / "board_video_events.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=VIDEO)
    parser.add_argument("--samples", type=int, default=120)
    args = parser.parse_args()
    video = args.video
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    stride = max(1, int(round(fps / 10.0)))
    detector = CapiMtlDetector(str(MODEL), (512, 288), class_names=BDD_NAMES)
    engine = SceneHazardEngine()
    counts = Counter()
    timeline = {}
    samples = 0
    started = time.perf_counter()
    last_events = []
    try:
        while samples < args.samples:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if frame_id % stride:
                continue
            media_s = frame_id / fps
            items, inference_ms = detector.detect(frame)
            masks = detector.last_masks.copy() if detector.last_masks is not None else None
            observation = engine.update(
                frame, items, frame_id, int(media_s * 1e9),
                masks=masks, letterbox=detector.last_letterbox,
                source_width=frame.shape[1], source_height=frame.shape[0])
            last_events = [dict(item) for item in observation.events]
            for event in observation.events:
                key = (event["kind"], event["track_id"])
                entry = timeline.setdefault(key, {
                    "kind": event["kind"], "track_id": event["track_id"],
                    "triggered_media_s": None, "cleared_media_s": None,
                    "assessment": event["assessment"],
                    "target_class": event["target_class"],
                    "evidence": event["evidence"],
                })
                if event["state"] == "TRIGGERED":
                    entry["triggered_media_s"] = event["media_s"]
                    counts[event["kind"]] += 1
                elif event["state"] == "CLEARED":
                    entry["cleared_media_s"] = event["media_s"]
            samples += 1
    finally:
        cap.release()
        detector.close()
    elapsed = time.perf_counter() - started
    payload = {
        "scope": "BOARD_RK3566_CAPI_MTL_ROADSCENE_HAZARDFUSION",
        "video": str(video),
        "model": str(MODEL),
        "samples": samples,
        "wall_s": round(elapsed, 3),
        "mean_ms": round(elapsed / max(1, samples) * 1000, 3),
        "hazard_mean_ms": round(
            sum(engine.timings) / max(1, len(engine.timings)), 3),
        "last_breakdown": {k: round(v, 2)
                           for k, v in engine.last_breakdown.items()},
        "event_counts": dict(counts),
        "timeline": list(timeline.values()),
        "last_events": last_events,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = OUT.with_name("board_video_events_" + video.stem + ".json")
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "samples", "wall_s", "mean_ms", "hazard_mean_ms",
        "last_breakdown", "event_counts")}, ensure_ascii=False))
    print("timeline", json.dumps(payload["timeline"], ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
