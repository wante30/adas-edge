import sys

import cv2

sys.path.insert(0, "/home/cat/competition_cockpit_port_20261018")
sys.path.insert(0, "/home/cat/qt_gpu_probe_ai_20261019")

from capi_detector import CapiDetector

model = "/home/cat/cityscapes_arch_compare_20261027/models/b05n_p2_multiclass_capi.rknn"
det = CapiDetector(model, (512, 288), class_names=("pedestrian", "rider", "car", "truck",
                                                    "bus", "train", "motorcycle", "bicycle",
                                                    "traffic light", "traffic sign"))
for q in (0, 1, 3):
    frame = cv2.imread(f"/tmp/pgv_q{q}.png")
    items, ms = det.detect(frame)
    print("q", q, "n", len(items), "ms", round(ms, 1),
          sorted((i["class_name"], round(i["confidence"], 2)) for i in items)[:12])
det.close()
