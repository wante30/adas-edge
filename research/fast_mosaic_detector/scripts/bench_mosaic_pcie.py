"""Measure FAST-MOSAIC throughput on the live FPGA PCIe PGV stream."""

from __future__ import annotations

import json
import os
import statistics
import struct
import sys
import time
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
PGV = Path("/dev/shm/pango-video/latest.pgv")
HEADER = struct.Struct("<4s7IQ")
FRAME_BYTES = 1280 * 720 * 2


def load_pgv():
    data = PGV.read_bytes()
    magic, version, width, height, frame_id, generation, crc, payload_bytes, stamp = HEADER.unpack_from(data)
    if magic != b"PGVF":
        raise RuntimeError("bad PGV magic")
    raw = np.frombuffer(data, dtype="<u2", count=1280 * 720, offset=HEADER.size)
    return cv2.cvtColor(raw.view(np.uint8).reshape(720, 1280, 2), cv2.COLOR_BGR5652BGR), frame_id


def stats(values):
    values = sorted(values)
    return (round(values[len(values) // 2], 2),
            round(values[int(len(values) * 0.95)], 2),
            round(statistics.mean(values), 2))


def main():
    det = CapiDetector(MODEL, (512, 288), class_names=(
        "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle",
        "bicycle", "traffic light", "traffic sign"))
    times, frames = [], []
    boxes = 0
    started = time.perf_counter()
    for _ in range(100):
        frame, frame_id = load_pgv()
        t0 = time.perf_counter()
        items, ms = det.detect(frame)
        times.append((time.perf_counter() - t0) * 1000)
        frames.append(frame_id)
        boxes += len(items)
    wall = time.perf_counter() - started
    result = {
        "source": "real FPGA PCIe /dev/shm/pango-video/latest.pgv",
        "frames": len(times),
        "fps": round(len(times) / wall, 2),
        "wall_s": round(wall, 3),
        "unique_pgv_frames": len(set(frames)),
        "detections": boxes,
        "wall_ms_p50_p95_mean": stats(times),
        "inference_ms_last": round(float(ms), 2),
        "note": "Qt/HMI running; one inference per full 1280x720 mosaic frame.",
    }
    (OUT / "mosaic_pcie_bench.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    det.close()


if __name__ == "__main__":
    main()
