import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/home/cat/competition_cockpit_port_20261018")
sys.path.insert(0, "/home/cat/qt_gpu_probe_ai_20261019")
from capi_detector import CapiDetector
from tools.competition_console.rknn_detector import RknnConsoleDetector

P2 = "/home/cat/cityscapes_arch_compare_20261027/models/b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn"
CAPI = "/home/cat/cityscapes_arch_compare_20261027/models/b05n_p2_multiclass_capi.rknn"

a = np.fromfile("/home/cat/b05n_mtl_b2a_bench_20261031/replay60/frame_000.rgb565", dtype="<u2", count=1280 * 720)
frame = cv2.cvtColor(a.view(np.uint8).reshape(720, 1280, 2), cv2.COLOR_BGR5652BGR)

py = RknnConsoleDetector(P2, (512, 288), class_profile="bdd10")
cap = CapiDetector(CAPI, (512, 288), class_profile="bdd10")
for i in range(3):
    p_items, p_ms = py.detect(frame)
    c_items, c_ms = cap.detect(frame)
print("python", len(p_items), round(p_ms, 2), [(x["class_name"], round(x["confidence"], 3)) for x in p_items[:6]])
print("capi  ", len(c_items), round(c_ms, 2), [(x["class_name"], round(x["confidence"], 3)) for x in c_items[:6]])
print("classes py", sorted(set(x["class_name"] for x in p_items)))
print("classes c ", sorted(set(x["class_name"] for x in c_items)))
print("timing c", cap.last_timing)
py.close(); cap.close()
