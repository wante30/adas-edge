"""Independent C API lifecycle switch: BDD -> P2 -> MTL-FAST -> BDD."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/cat/competition_cockpit_port_20261018")
TEST = Path("/home/cat/fast_mosaic_20261101")
sys.path.insert(0, str(TEST))
sys.path.insert(0, str(ROOT))
os.environ["ADAS_P2CAPI_LIB"] = str(TEST / "libp2capi_fast.so")

import cv2
import numpy as np
from capi_detector import CapiDetector, CapiMtlDetector

MODELS = {
    "BDD": "/home/cat/cityscapes_arch_compare_20261027/models/bdd_yolov8n_512x288_raw_heads_int8_capi.rknn",
    "P2": "/home/cat/cityscapes_arch_compare_20261027/models/b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn",
    "MTL-FAST": "/home/cat/models/b05n_p2_mtl_b2a_seg256/b05n_p2_mtl_b2a_seg256_int8.rknn",
}
FRAME = Path("/home/cat/b05n_mtl_b2a_bench_20261031/replay60/frame_000.rgb565")


def load():
    raw = np.fromfile(FRAME, dtype="<u2", count=1280 * 720)
    return cv2.cvtColor(raw.view(np.uint8).reshape(720, 1280, 2), cv2.COLOR_BGR5652BGR)


def main():
    frame = load()
    report = []
    for cycle in range(2):
        for name in ("BDD", "P2", "MTL-FAST", "BDD"):
            cls = CapiMtlDetector if name == "MTL-FAST" else CapiDetector
            det = cls(MODELS[name], (512, 288))
            items, ms = det.detect(frame)
            report.append({
                "cycle": cycle, "model": name, "type": type(det).__name__,
                "output_count": det.output_count,
                "mask": [det.mask_w, det.mask_h] if name == "MTL-FAST" else None,
                "detections": len(items),
                "inference_ms": round(float(ms), 2),
            })
            det.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    Path("/home/cat/fast_mosaic_20261101/results/capi_lifecycle.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
