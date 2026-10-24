import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, "/home/cat/adas-edge-rect")
from competition_ai.detector import CompetitionDetector

IMAGES = Path("/home/cat/b05c_images")
OUT = Path("/home/cat/b05d_roi.json")
MODEL = "/home/cat/adas-edge-rect/tl_check/tl_specialist_512x192_hybrid2.rknn"
VARIANTS = {
    "baseline_1280x432": (0, 0, 1280, 432),
    "center_960x432": (160, 0, 1120, 432),
    "center_800x432": (240, 0, 1040, 432),
}


def main():
    detector = CompetitionDetector(
        MODEL, input_size=(512, 192), class_names=("traffic light",),
        conf_threshold=0.001, nms_threshold=0.70,
    )
    all_output = {}
    for name, (x0, y0, x1, y1) in VARIANTS.items():
        per_image = {}
        for path in sorted(IMAGES.glob("*.jpg")):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            x1c, y1c = min(x1, w), min(y1, h)
            crop = frame[y0:y1c, x0:x1c]
            result = detector.infer(crop, frame_id=0)
            rows = []
            for det in result.detections:
                if str(det.class_name) != "traffic light":
                    continue
                rows.append({
                    "confidence": float(det.confidence),
                    "bbox": [float(det.x1) + x0, float(det.y1) + y0,
                             float(det.x2) + x0, float(det.y2) + y0],
                })
            per_image[path.name] = rows
        all_output[name] = per_image
        print(name, sum(len(v) for v in per_image.values()), flush=True)
    detector.close()
    OUT.write_text(json.dumps(all_output, separators=(",", ":")), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
