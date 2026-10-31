import shutil
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
a = np.fromfile(FRAME, dtype="<u2", count=1280 * 720)
full = cv2.cvtColor(a.view(np.uint8).reshape(720, 1280, 2), cv2.COLOR_BGR5652BGR)
road = full[360:720, 640:1280].copy()  # quadrant 3

for tag, path in (("full", "/home/cat/models/b05n_p2_mtl_b2a/b05n_p2_mtl_b2a_int8.rknn"),
                  ("fast", "/home/cat/models/b05n_p2_mtl_b2a_seg256/b05n_p2_mtl_b2a_seg256_int8.rknn")):
    det = CapiMtlDetector(path, (512, 288), class_names=capi_detector.BDD_NAMES)
    for _ in range(3):
        det.detect(road)
        time.sleep(0.15)
    time.sleep(0.6)
    shutil.copy(capi_detector.OVERLAY_PATH, f"/home/cat/b05n_p2_capi_integration_20261031/overlay_{tag}.png")
    shutil.copy(capi_detector.OVERLAY_META, f"/home/cat/b05n_p2_capi_integration_20261031/overlay_{tag}.json")
    overlay = cv2.imread(capi_detector.OVERLAY_PATH, cv2.IMREAD_UNCHANGED)
    print(tag, "mask", det.mask_w, det.mask_h, "overlay", overlay.shape,
          "timing", {k: round(v, 1) for k, v in det.last_timing.items() if isinstance(v, float)})
    det.close()
cv2.imwrite("/home/cat/b05n_p2_capi_integration_20261031/road_frame.png", road)
print("done")
