import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path("/home/cat/competition_cockpit_port_20261018")
TEST = Path("/home/cat/fast_mosaic_20261101")
PROD = Path("/home/cat/qt_gpu_probe_ai_20261019")
sys.path.insert(0, str(PROD))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEST))

os.environ["ADAS_P2CAPI_LIB"] = str(TEST / "libp2capi_fast.so")
import capi_detector
from capi_detector import CapiDetector

# Load the production adapter (Python RKNN Lite path) from the repo snapshot.
spec = importlib.util.spec_from_file_location("ai_bdd_test", TEST / "ai_bridge.py")
ai = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ai
spec.loader.exec_module(ai)
comp_spec = importlib.util.spec_from_file_location("comp_bdd_test", TEST / "b05p_competition_bridge.py")
comp = importlib.util.module_from_spec(comp_spec)
sys.modules[comp_spec.name] = comp
comp_spec.loader.exec_module(comp)
comp._install_raw_head_adapter(ai, "B")
Adapter = ai.RknnConsoleDetector

import cv2
import numpy as np

MODEL = "/home/cat/cityscapes_arch_compare_20261027/models/bdd_yolov8n_512x288_raw_heads_int8.rknn"
CAPI = "/home/cat/cityscapes_arch_compare_20261027/models/bdd_yolov8n_512x288_raw_heads_int8_capi.rknn"
FRAME = "/home/cat/b05n_mtl_b2a_bench_20261031/replay60/frame_000.rgb565"
raw = np.fromfile(FRAME, dtype="<u2", count=1280 * 720)
full = cv2.cvtColor(raw.view(np.uint8).reshape(720, 1280, 2), cv2.COLOR_BGR5652BGR)
road = full[360:720, 640:1280]  # Q4

py = Adapter(MODEL, (512, 288), class_profile="bdd10")
cap = CapiDetector(CAPI, (512, 288), class_names=capi_detector.BDD_NAMES)


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)


def compare(label, frame):
    p_items, p_ms = py.detect(frame)
    c_items, c_ms = cap.detect(frame)
    print(label, "python", len(p_items), round(p_ms, 1), "capi", len(c_items), round(c_ms, 1))
    print("  py classes", sorted((x["class_name"], round(x["confidence"], 3)) for x in p_items))
    print("  c  classes", sorted((x["class_name"], round(x["confidence"], 3)) for x in c_items))
    matched = 0; ious = []
    used = set()
    for p in p_items:
        best = -1; best_iou = 0
        for j, c in enumerate(c_items):
            if j in used or c["class_name"] != p["class_name"]:
                continue
            v = iou(p["bbox"], c["bbox"])
            if v > best_iou:
                best_iou = v; best = j
        if best >= 0 and best_iou >= 0.5:
            matched += 1; used.add(best); ious.append(best_iou)
    print("  matched@0.5", matched, "/", len(p_items), "mean_iou", round(float(np.mean(ious)), 3) if ious else None)
    print("  capi output_count", cap.output_count, "scales", cap.scale_count, "timing", {k: round(v, 1) for k, v in cap.last_timing.items() if isinstance(v, float)})


compare("Q4", road)
compare("FULL", full)
py.close(); cap.close()
