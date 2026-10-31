import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path("/home/cat/competition_cockpit_port_20261018")
BRIDGE_DIR = Path("/home/cat/qt_gpu_probe_ai_20261019")
os.environ["ADAS_EDGE_ROOT"] = str(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BRIDGE_DIR))

import cv2
import numpy as np

from competition_ai.detector import BDD_NAMES
from tools.competition_console.rknn_detector import RknnConsoleDetector as BaseDetector

spec = importlib.util.spec_from_file_location("ai_tl", BRIDGE_DIR / "ai_bridge.py")
ai = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ai; spec.loader.exec_module(ai)
comp_spec = importlib.util.spec_from_file_location("comp_tl", BRIDGE_DIR / "b05p_competition_bridge.py")
comp = importlib.util.module_from_spec(comp_spec); sys.modules[comp_spec.name] = comp; comp_spec.loader.exec_module(comp)
comp._install_raw_head_adapter(ai, "B")
Adapter = ai.RknnConsoleDetector

P2 = "/home/cat/cityscapes_arch_compare_20261027/models/b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn"
CAPI = "/home/cat/cityscapes_arch_compare_20261027/models/b05n_p2_multiclass_capi.rknn"
a = np.fromfile("/home/cat/b05n_mtl_b2a_bench_20261031/replay60/frame_000.rgb565", dtype="<u2", count=1280 * 720)
frame = cv2.cvtColor(a.view(np.uint8).reshape(720, 1280, 2), cv2.COLOR_BGR5652BGR)

for name, path, cls in (("base-B", P2, BaseDetector), ("adapter-B", P2, Adapter),
                        ("adapter-capi", CAPI, Adapter)):
    det = cls(path, (512, 288), class_profile="bdd10")
    items, ms = det.detect(frame)
    print(name, "n=", len(items), "ms=", round(ms, 1),
          sorted((i["class_name"], round(i["confidence"], 3)) for i in items))
    det.close()
