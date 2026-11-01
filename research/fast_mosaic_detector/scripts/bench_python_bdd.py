"""Benchmark BDD-YOLOv8n Python RKNN Lite on the same 60-frame replay."""

from __future__ import annotations

import importlib.util
import statistics
import sys
import time
from pathlib import Path

ROOT = Path("/home/cat/competition_cockpit_port_20261018")
TEST = Path("/home/cat/fast_mosaic_20261101")
PROD = Path("/home/cat/qt_gpu_probe_ai_20261019")
sys.path.insert(0, str(PROD))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEST))

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location("ai_bench", TEST / "ai_bridge.py")
ai = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ai
spec.loader.exec_module(ai)
comp_spec = importlib.util.spec_from_file_location("comp_bench", TEST / "b05p_competition_bridge.py")
comp = importlib.util.module_from_spec(comp_spec)
sys.modules[comp_spec.name] = comp
comp_spec.loader.exec_module(comp)
comp._install_raw_head_adapter(ai, "B")
Detector = ai.RknnConsoleDetector

MODEL = "/home/cat/cityscapes_arch_compare_20261027/models/bdd_yolov8n_512x288_raw_heads_int8.rknn"
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
    det = Detector(MODEL, (512, 288), class_profile="bdd10")
    for _ in range(10):
        det.detect(frames[0])
    times = []
    detect_ms = []
    counts = 0
    started = time.perf_counter()
    for round_index in range(5):
        for frame in frames:
            t0 = time.perf_counter()
            items, ms = det.detect(frame)
            times.append((time.perf_counter() - t0) * 1000)
            detect_ms.append(float(ms))
            counts += len(items)
    wall = time.perf_counter() - started
    print("BDD_PYTHON frames", len(times), "fps", round(len(times) / wall, 2),
          "wall_s", round(wall, 3), "detections", counts)
    print("BDD_PYTHON wall_ms p50/p95/mean", stats(times))
    print("BDD_PYTHON infer_ms p50/p95/mean", stats(detect_ms))
    det.close()


if __name__ == "__main__":
    main()
