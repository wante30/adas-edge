import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, "/home/cat/competition_cockpit_port_20261018")
from competition_ai.detector import CompetitionDetector
from competition_ai.model_profiles import get_profile

ROOT = Path("/home/cat/b05h_frames")
OUT = Path("/home/cat/b05h_teachers.json")
BDD = "/home/cat/adas-edge-rect/rect_models/bdd_yolov8n_512x288_int8_kl.rknn"
SPEC = "/home/cat/adas-edge-rect/tl_check/tl_specialist_512x192_hybrid2.rknn"


def run(path, size, names, crop):
    det = CompetitionDetector(path, input_size=size, class_names=names,
                              conf_threshold=0.001, nms_threshold=0.45 if crop is None else 0.70)
    out = {}
    for p in sorted(ROOT.rglob("*.jpg")):
        im = cv2.imread(str(p))
        if im is None:
            continue
        h = im.shape[0]
        source = im if crop is None else im[:int(h * crop)]
        rows = []
        r = det.infer(source)
        for d in r.detections:
            if str(d.class_name) != "traffic light":
                continue
            rows.append({"bbox": [float(d.x1), float(d.y1), float(d.x2), float(d.y2)],
                         "confidence": float(d.confidence)})
        out[p.name] = rows
    det.close()
    return out


def main():
    result = {
        "bdd": run(BDD, (512, 288), get_profile("bdd10"), None),
        "specialist": run(SPEC, (512, 192), ("traffic light",), 0.60),
    }
    OUT.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    print("wrote", OUT, len(result["bdd"]), sum(len(v) for v in result["bdd"].values()), sum(len(v) for v in result["specialist"].values()))


if __name__ == "__main__":
    main()
