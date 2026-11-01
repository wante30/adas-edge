"""Benchmark BDD-YOLOv8n C API through the Python ctypes wrapper."""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path("/home/cat/competition_cockpit_port_20261018")
TEST = Path("/home/cat/fast_mosaic_20261101")
sys.path.insert(0, str(TEST))
sys.path.insert(0, str(ROOT))
os.environ["ADAS_P2CAPI_LIB"] = str(TEST / "libp2capi_fast.so")

import cv2
import numpy as np
from capi_detector import CapiDetector

MODEL = "/home/cat/cityscapes_arch_compare_20261027/models/bdd_yolov8n_512x288_raw_heads_int8_capi.rknn"
FRAMES = sorted(Path("/home/cat/b05n_mtl_b2a_bench_20261031/replay60").glob("frame_*.rgb565"))


def load(path):
    raw = np.fromfile(path, dtype="<u2", count=1280 * 720)
    return cv2.cvtColor(raw.view(np.uint8).reshape(720, 1280, 2), cv2.COLOR_BGR5652BGR)


def stats(values):
    values = sorted(values)
    return (round(values[len(values) // 2], 2),
            round(values[int(len(values) * 0.95)], 2),
            round(statistics.mean(values), 2))


def main():
    frames = [load(path) for path in FRAMES]
    det = CapiDetector(MODEL, (512, 288), class_names=(
        "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle",
        "bicycle", "traffic light", "traffic sign"))
    for _ in range(10):
        det.detect(frames[0])
    wall_times, rknn, get, decode = [], [], [], []
    counts = 0
    started = time.perf_counter()
    for _ in range(5):
        for frame in frames:
            t0 = time.perf_counter()
            items, _ = det.detect(frame)
            wall_times.append((time.perf_counter() - t0) * 1000)
            timing = det.last_timing or {}
            rknn.append(float(timing.get("rknn_call_ms", 0)))
            get.append(float(timing.get("decode_nms_ms", 0)))
            decode.append(float(timing.get("preprocess_ms", 0)))
            counts += len(items)
    wall = time.perf_counter() - started
    print("BDD_CAPI frames", len(wall_times), "fps", round(len(wall_times) / wall, 2),
          "wall_s", round(wall, 3), "detections", counts,
          "output_count", det.output_count, "scales", det.scale_count)
    print("BDD_CAPI wall_ms p50/p95/mean", stats(wall_times))
    print("BDD_CAPI rknn_ms p50/p95/mean", stats(rknn))
    print("BDD_CAPI preprocess_ms p50/p95/mean", stats(decode))
    print("BDD_CAPI decode_nms_ms p50/p95/mean", stats(get))
    print("BDD_CAPI last_timing", {k: round(v, 2) if isinstance(v, float) else v
                                   for k, v in (det.last_timing or {}).items()})
    det.close()


if __name__ == "__main__":
    main()
