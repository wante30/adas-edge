"""Draw BDD50 false-positive examples grouped by non-traffic-light overlap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

COLOR = {"traffic sign": (0, 180, 255), "car": (255, 160, 0), "background/no_non_tl_gt": (0, 0, 255)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.analysis.read_text(encoding="utf-8"))
    gt_data = {}
    for path in args.gt.glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        gt_data[str(value["name"])] = value
    for label, model in data["models"].items():
        groups = {}
        for example in model["fp"]["examples"]:
            groups.setdefault(example["contained_in_category"], []).append(example)
        for category, examples in groups.items():
            items = []
            for example in sorted(examples, key=lambda row: -row["confidence"])[:args.limit]:
                image = cv2.imread(str(args.images / f"{example['image']}.jpg"))
                canvas = image.copy()
                x1, y1, x2, y2 = (int(round(value)) for value in example["bbox"])
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Draw the non-TL GT box that contains/overlaps this FP when available.
                for frame in gt_data[example["image"]].get("frames", []):
                    for obj in frame.get("objects", []):
                        if obj.get("category") == "traffic light":
                            continue
                        b = obj.get("box2d", {})
                        bx1, by1, bx2, by2 = (float(b.get(k, 0)) for k in ("x1", "y1", "x2", "y2"))
                        if bx1 <= x1 <= bx2 and by1 <= y1 <= by2:
                            cv2.rectangle(canvas, (int(bx1), int(by1)), (int(bx2), int(by2)),
                                          COLOR.get(str(obj.get("category")), (0, 255, 0)), 1)
                            break
                cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), (18, 24, 28), -1)
                caption = f"{example['image']} FP {example['confidence']:.2f} contained={category} frac={example['contained_fraction']:.2f}"
                cv2.putText(canvas, caption, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 248, 250), 1, cv2.LINE_AA)
                cv2.imwrite(str(args.output / f"{label}_{category.replace('/', '_')}_{len(items):02d}.jpg"), canvas,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                items.append(canvas)
            if items:
                width, height = 640, 360
                cols = 2
                rows = (len(items) + cols - 1) // cols
                sheet = np.full((rows * height, cols * (width + 6), 3), 28, np.uint8)
                for index, image in enumerate(items):
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                    row, col = divmod(index, cols)
                    sheet[row * height:(row + 1) * height, col * (width + 6):col * (width + 6) + width] = image
                cv2.imwrite(str(args.output / f"{label}_{category.replace('/', '_')}_sheet.jpg"), sheet,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(json.dumps({"output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
