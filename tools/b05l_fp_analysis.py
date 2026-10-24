"""B-05L BDD50 false-positive and false-negative breakdown.

False positives are predictions that cannot be one-to-one matched to a
traffic-light GT.  We then check whether the box overlaps another BDD GT class
(traffic sign, car, truck, bus, person, ...), which gives an objective hint for
sign/tail-light confusions without pretending it is a perfect attribution.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.b05k_common import size_group  # noqa: E402
from tools.b05l_eval import iou, match_image  # noqa: E402


def load_objects(gt_dir: Path, images_dir: Path, roi_top: float) -> dict[str, dict[str, list[dict[str, Any]]]]:
    output = {}
    for path in sorted(gt_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        name = str(data["name"])
        image = cv2.imread(str(images_dir / f"{name}.jpg"))
        height = image.shape[0]
        roi_height = int(round(height * roi_top))
        rows: dict[str, list[dict[str, Any]]] = {}
        for frame in data.get("frames", []):
            for obj in frame.get("objects", []):
                box = obj.get("box2d", {})
                x1, y1, x2, y2 = (float(box.get(k, 0)) for k in ("x1", "y1", "x2", "y2"))
                if x2 <= x1 or y2 <= y1:
                    continue
                category = str(obj.get("category", "unknown"))
                rows.setdefault(category, []).append({
                    "bbox": [x1, y1, x2, y2],
                    "color": str(obj.get("attributes", {}).get("trafficLightColor", "none")).lower(),
                    "occluded": bool(obj.get("attributes", {}).get("occluded", False)),
                    "truncated": bool(obj.get("attributes", {}).get("truncated", False)),
                    "in_roi": not (y1 >= roi_height),
                })
        output[name] = rows
    return output


def analyze(label: str, eval_data: dict[str, Any], objects: dict[str, dict[str, list[dict[str, Any]]]], conf: float) -> dict[str, Any]:
    predictions = eval_data["predictions_by_image"]
    fp_overlap_categories = Counter()
    fp_contained_categories = Counter()
    fp_conf = []
    fp_examples = []
    fn_sizes = Counter()
    fn_colors = Counter()
    fn_occluded = Counter()
    fn_truncated = Counter()
    fn_examples = []
    total_fp = 0
    total_fn = 0
    for image_name, categories in objects.items():
        targets = [item for item in categories.get("traffic light", []) if item["in_roi"]]
        pairs, tp, fp, fn = match_image(targets, predictions.get(image_name, []), conf=conf, iou_threshold=0.5)
        matched_gt = {index for _, index in pairs}
        matched_pred = {id(prediction) for prediction, _ in pairs}
        for prediction in predictions.get(image_name, []):
            if float(prediction["confidence"]) < conf or id(prediction) in matched_pred:
                continue
            total_fp += 1
            best_category, best_overlap = "background/no_non_tl_gt", 0.0
            contain_category, contain_fraction = "background/no_non_tl_gt", 0.0
            px1, py1, px2, py2 = prediction["bbox"]
            pred_area = max(1.0, (px2 - px1) * (py2 - py1))
            for category, items in categories.items():
                if category == "traffic light":
                    continue
                for item in items:
                    overlap = iou(prediction["bbox"], item["bbox"])
                    if overlap > best_overlap:
                        best_overlap, best_category = overlap, category
                    ox1, oy1, ox2, oy2 = item["bbox"]
                    inter = max(0.0, min(px2, ox2) - max(px1, ox1)) * max(0.0, min(py2, oy2) - max(py1, oy1))
                    fraction = inter / pred_area
                    center_inside = ox1 <= (px1 + px2) / 2 <= ox2 and oy1 <= (py1 + py2) / 2 <= oy2
                    if center_inside and fraction > contain_fraction:
                        contain_fraction, contain_category = fraction, category
            if best_overlap < 0.3:
                best_category = "background/no_non_tl_gt"
            if contain_fraction < 0.5:
                contain_category = "background/no_non_tl_gt"
            fp_overlap_categories[best_category] += 1
            fp_contained_categories[contain_category] += 1
            fp_conf.append(float(prediction["confidence"]))
            if len(fp_examples) < 500:
                fp_examples.append({"image": image_name, "bbox": prediction["bbox"],
                                    "confidence": float(prediction["confidence"]),
                                    "overlap_category": best_category, "overlap_iou": best_overlap,
                                    "contained_in_category": contain_category,
                                    "contained_fraction": contain_fraction})
        for index, target in enumerate(targets):
            if index in matched_gt:
                continue
            total_fn += 1
            fn_sizes[size_group(float(target["bbox"][2] - target["bbox"][0]))] += 1
            fn_colors[target["color"]] += 1
            fn_occluded[str(target["occluded"])] += 1
            fn_truncated[str(target["truncated"])] += 1
            if len(fn_examples) < 500:
                fn_examples.append({"image": image_name, "bbox": target["bbox"], "color": target["color"],
                                    "width_px": target["bbox"][2] - target["bbox"][0],
                                    "occluded": target["occluded"], "truncated": target["truncated"]})
    return {
        "label": label,
        "conf": conf,
        "fp": {
            "total": total_fp,
            "by_overlap_category": dict(fp_overlap_categories),
            "by_contained_category": dict(fp_contained_categories),
            "mean_confidence": sum(fp_conf) / len(fp_conf) if fp_conf else 0.0,
            "examples": fp_examples,
        },
        "fn": {
            "total": total_fn,
            "by_size_group": dict(fn_sizes),
            "by_color": dict(fn_colors),
            "by_occluded": dict(fn_occluded),
            "by_truncated": dict(fn_truncated),
            "examples": fn_examples,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", action="append", required=True, help="LABEL=eval.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    loaded = []
    for value in args.eval:
        label, path = value.split("=", 1)
        loaded.append((label, json.loads(Path(path).read_text(encoding="utf-8"))))
    first = loaded[0][1]
    objects = load_objects(Path(first["dataset"]["gt_dir"]), Path(first["dataset"]["images_dir"]),
                           float(first["dataset"]["roi_top_fraction"]))
    result = {"models": {label: analyze(label, data, objects, args.conf) for label, data in loaded}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({label: {"fp": value["fp"]["total"],
                              "fp_overlap_categories": value["fp"]["by_overlap_category"],
                              "fp_contained_categories": value["fp"]["by_contained_category"],
                              "fn": value["fn"]["total"], "fn_sizes": value["fn"]["by_size_group"]}
                      for label, value in result["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
