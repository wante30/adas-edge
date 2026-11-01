import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/home/cat/competition_cockpit_port_20261018")
sys.path.insert(0, "/home/cat/b05n_p2_capi_integration_20261031")

import capi_detector

capi_detector.LIB_PATH = "/home/cat/b05n_p2_capi_integration_20261031/libp2capi.so.new"
from capi_detector import CapiMtlDetector

FRAME = "/home/cat/b05n_mtl_b2a_bench_20261031/replay60/frame_000.rgb565"
MODELS = {
    "MTL-FULL": "/home/cat/models/b05n_p2_mtl_b2a/b05n_p2_mtl_b2a_int8.rknn",
    "MTL-FAST": "/home/cat/models/b05n_p2_mtl_b2a_seg256/b05n_p2_mtl_b2a_seg256_int8.rknn",
}

a = np.fromfile(FRAME, dtype="<u2", count=1280 * 720)
frame = cv2.cvtColor(a.view(np.uint8).reshape(720, 1280, 2), cv2.COLOR_BGR5652BGR)

for name, path in MODELS.items():
    det = CapiMtlDetector(path, (512, 288), class_names=capi_detector.BDD_NAMES)
    times = []
    for i in range(5):
        started = time.perf_counter()
        items, ms = det.detect(frame)
        times.append((time.perf_counter() - started) * 1000)
    lane, style, drv = det.last_masks
    print(name, "mask", det.mask_w, "x", det.mask_h, "items", len(items))
    print("  classes", sorted((x["class_name"], round(x["confidence"], 3)) for x in items))
    print("  lane_hist", np.bincount(lane.ravel(), minlength=9).tolist())
    print("  style_fg", np.bincount(style[(lane != 8)].ravel(), minlength=3).tolist())
    print("  drivable_hist", np.bincount(drv.ravel(), minlength=3).tolist())
    print("  timing", {k: (round(v, 2) if isinstance(v, float) else v)
                       for k, v in det.last_timing.items()})
    print("  wall_ms", [round(t, 1) for t in times])
    det.close()
    import os
    print("  overlay", os.path.exists(capi_detector.OVERLAY_PATH),
          os.path.getsize(capi_detector.OVERLAY_PATH) if os.path.exists(capi_detector.OVERLAY_PATH) else 0)
