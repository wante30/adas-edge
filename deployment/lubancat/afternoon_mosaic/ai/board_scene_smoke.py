"""Board smoke: C API MTL -> SceneObservation -> RoadScene -> HazardFusion."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from capi_detector import BDD_NAMES, CapiMtlDetector
from tools.competition_console.hazard_scene_adapter import SceneHazardEngine

FRAME_DIR = Path("/home/cat/b05p_q4_frontvideo_replay_20261029")
MODEL = Path("/home/cat/models/b05n_p2_mtl_b2a_seg256/"
             "b05n_p2_mtl_b2a_seg256_int8.rknn")


def decode(payload: bytes):
    packed = np.frombuffer(payload, dtype=np.uint8).reshape(720, 1280, 2)
    return cv2.cvtColor(packed, cv2.COLOR_BGR5652BGR)


def main():
    detector = CapiMtlDetector(str(MODEL), (512, 288), class_names=BDD_NAMES)
    engine = SceneHazardEngine()
    paths = sorted(FRAME_DIR.glob("frame_*.rgb565"))[:32]
    events = {}
    timings = []
    started = time.perf_counter()
    for index, path in enumerate(paths):
        full = decode(path.read_bytes())
        frame = full[360:, 640:]
        items, inference_ms = detector.detect(frame)
        masks = detector.last_masks.copy() if detector.last_masks is not None else None
        letterbox = detector.last_letterbox
        timestamp_ns = int(index * 33_333_333)
        observation = engine.update(
            frame, items, index, timestamp_ns,
            masks=masks, letterbox=letterbox,
            source_width=frame.shape[1], source_height=frame.shape[0])
        if index < 3:
            print("DBG", index, "masks",
                  None if masks is None else np.unique(masks[0]).tolist(),
                  "road", observation.reason, "tracks", len(observation.tracks))
        for event in observation.events:
            events[event["kind"]] = events.get(event["kind"], 0) + 1
        timings.append({
            "frame": index,
            "detections": len(items),
            "inference_ms": round(inference_ms, 3),
            "hazard_ms": round(engine.timings[-1], 3),
            "breakdown": {key: round(value, 2)
                          for key, value in engine.last_breakdown.items()},
            "reason": observation.reason,
            "events": [event["kind"] for event in observation.events],
        })
    elapsed = time.perf_counter() - started
    print(json.dumps({
        "frames": len(paths),
        "wall_s": round(elapsed, 3),
        "mean_ms": round(elapsed / max(1, len(paths)) * 1000, 3),
        "event_counts": events,
        "last_tracks": observation.tracks[-3:],
        "timings": timings[-8:],
    }, ensure_ascii=False))
    detector.close()


if __name__ == "__main__":
    main()
