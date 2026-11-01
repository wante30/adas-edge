"""Synthetic 2x2 coordinate test: four road frames tiled, one inference."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/cat/competition_cockpit_port_20261018")
TEST = Path("/home/cat/fast_mosaic_20261101")
OUT = Path("/home/cat/fast_mosaic_20261101/results")
sys.path.insert(0, str(TEST))
sys.path.insert(0, str(ROOT))
os.environ["ADAS_P2CAPI_LIB"] = str(TEST / "libp2capi_fast.so")

import cv2
import numpy as np
from capi_detector import CapiDetector

MODEL = "/home/cat/cityscapes_arch_compare_20261027/models/bdd_yolov8n_512x288_raw_heads_int8_capi.rknn"
FRAMES = sorted(Path("/home/cat/b05n_mtl_b2a_bench_20261031/replay60").glob("frame_*.rgb565"))
COLORS = [(60, 227, 255), (69, 243, 176), (255, 188, 92), (255, 80, 100)]
NAMES = {0: "Q1", 1: "Q2", 2: "Q3", 3: "Q4"}


def load(path):
    raw = np.fromfile(path, dtype="<u2", count=1280 * 720)
    return cv2.cvtColor(raw.view(np.uint8).reshape(720, 1280, 2), cv2.COLOR_BGR5652BGR)


def quadrant(cx, cy):
    return (2 if cy >= 360 else 0) + (1 if cx >= 640 else 0)


def main():
    picks = [FRAMES[i] for i in (0, 15, 30, 45)]
    tiles = [load(path) for path in picks]
    tiles = [cv2.resize(t, (640, 360)) for t in tiles]
    mosaic = np.zeros((720, 1280, 3), dtype=np.uint8)
    mosaic[0:360, 0:640] = tiles[0]
    mosaic[0:360, 640:1280] = tiles[1]
    mosaic[360:720, 0:640] = tiles[2]
    mosaic[360:720, 640:1280] = tiles[3]

    det = CapiDetector(MODEL, (512, 288), class_names=(
        "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle",
        "bicycle", "traffic light", "traffic sign"))
    items, ms = det.detect(mosaic)
    boxes = []
    for item in items:
        x1, y1, x2, y2 = [float(v) for v in item["bbox"]]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        q = quadrant(cx, cy)
        boxes.append({
            "class_name": item["class_name"],
            "confidence": round(float(item["confidence"]), 4),
            "bbox": [round(v, 2) for v in (x1, y1, x2, y2)],
            "center": [round(cx, 2), round(cy, 2)],
            "quadrant": q,
            "quadrant_name": NAMES[q],
            "crosses_boundary": bool((x1 < 640 < x2) or (y1 < 360 < y2)),
        })
    canvas = mosaic.copy()
    cv2.line(canvas, (640, 0), (640, 720), (30, 80, 95), 2)
    cv2.line(canvas, (0, 360), (1280, 360), (30, 80, 95), 2)
    for q, (x, y) in enumerate(((10, 30), (650, 30), (10, 390), (650, 390))):
        cv2.putText(canvas, NAMES[q], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    COLORS[q], 2, cv2.LINE_AA)
    for b in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in b["bbox"]]
        color = COLORS[b["quadrant"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f'{NAMES[b["quadrant"]]} {b["class_name"]} {b["confidence"]:.2f}'
        cv2.putText(canvas, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"synthetic 2x2 coordinate test (one inference) infer={ms:.1f} ms",
                (14, 708), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 250, 255), 1, cv2.LINE_AA)
    png = OUT / "mosaic_synthetic_bdd_capi.png"
    cv2.imwrite(str(png), canvas)
    payload = {
        "source": "four replay road frames tiled 2x2; one inference",
        "input": "1280x720 -> 512x288",
        "inference_ms": round(float(ms), 2),
        "detections": len(boxes),
        "quadrant_counts": {NAMES[q]: sum(1 for b in boxes if b["quadrant"] == q) for q in range(4)},
        "frame_files": [p.name for p in picks],
        "boxes": boxes,
    }
    (OUT / "mosaic_synthetic_bdd_capi.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({k: payload[k] for k in ("detections", "quadrant_counts", "inference_ms")}, ensure_ascii=False))
    det.close()


if __name__ == "__main__":
    main()
