import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, "/home/cat/adas-edge-rect")
from competition_ai.detector import CompetitionDetector
from competition_ai.model_profiles import get_profile

IMAGES = Path("/home/cat/b05c_images")
OUT = Path("/home/cat/b05c_predictions.json")
BDD = "/home/cat/adas-edge-rect/rect_models/bdd_yolov8n_512x288_int8_kl.rknn"
SPEC = "/home/cat/adas-edge-rect/tl_check/tl_specialist_512x192_hybrid2.rknn"


def run(model_path, size, class_names, crop_fraction=None, conf=0.001, nms=0.45):
    detector = CompetitionDetector(model_path, input_size=size, class_names=class_names,
                                   conf_threshold=conf, nms_threshold=nms)
    result = {}
    for image_path in sorted(IMAGES.glob("*.jpg")):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        source = frame
        y_scale = 1.0
        if crop_fraction is not None:
            roi_h = int(round(height * crop_fraction))
            source = frame[:roi_h]
            y_scale = 1.0  # specialist ROI starts at y=0: keep source-pixel y
        frame_result = detector.infer(source, frame_id=0)
        boxes = []
        for det in frame_result.detections:
            name = str(det.class_name)
            if name != "traffic light":
                continue
            boxes.append({
                "confidence": float(det.confidence),
                "bbox": [float(det.x1), float(det.y1) * y_scale,
                         float(det.x2), float(det.y2) * y_scale],
            })
        result[image_path.name] = boxes
    detector.close()
    return result


def main():
    bdd = run(BDD, (512, 288), get_profile("bdd10"), None, 0.001, 0.45)
    spec = run(SPEC, (512, 192), ("traffic light",), 0.60, 0.001, 0.70)
    OUT.write_text(json.dumps({
        "source": "RK3566 B05C; BDD KL 512x288 conf=.001 nms=.45; "
                  "Specialist 512x192 top60%% ROI conf=.001 nms=.70",
        "bdd": bdd,
        "specialist": spec,
    }, separators=(",", ":")), encoding="utf-8")
    print("wrote", OUT, "images", len(bdd))


if __name__ == "__main__":
    main()


