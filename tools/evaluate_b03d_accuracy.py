"""Evaluate the B-03C-R RKNN predictions against BDD box2d ground truth.

The input labels are one BDD-style JSON file per image, as supplied by the
documented HF mirror used when the official Detection 2020 download was
unreachable. Predictions are already in original 1280x720 coordinates.
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.coco_eval_helpers import (  # noqa: E402
    AREA_INDEX,
    CATEGORY_IDS,
    CLASSES,
    average_precision,
    average_precision_over_iou,
    box_area,
    operating_point,
    predictions_to_coco,
    size_bucket,
    strip_internal_fields,
)

BDD_MAP = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "pedestrian": "person",
    "person": "person",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "traffic light": "traffic light",
}
SKIPPED = {"rider", "traffic sign", "train", "trailer", "other vehicle", "other person"}
ALIASES = {"motor": "motorcycle", "bike": "bicycle"}


def _objects(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
        frames = payload["frames"]
        return list(frames[0].get("objects", [])) if frames else []
    if isinstance(payload, dict) and isinstance(payload.get("labels"), list):
        return list(payload["labels"])
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and isinstance(item.get("labels"), list):
                return list(item["labels"])
    raise ValueError("unsupported BDD label JSON shape")


def _objects_from_file(label_dir: Path, image_name: str) -> tuple[list[dict[str, Any]], str] | None:
    stem = Path(image_name).stem
    json_path = label_dir / (stem + ".json")
    xml_path = label_dir / (stem + ".xml")
    if json_path.is_file():
        return _objects(json.loads(json_path.read_text(encoding="utf-8"))), "json"
    if xml_path.is_file():
        root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
        objects = []
        for node in root.findall("object"):
            box = node.find("bndbox")
            if box is None:
                continue
            objects.append({
                "category": (node.findtext("name") or "").strip(),
                "box2d": {
                    "x1": float(box.findtext("xmin") or "nan"),
                    "y1": float(box.findtext("ymin") or "nan"),
                    "x2": float(box.findtext("xmax") or "nan"),
                    "y2": float(box.findtext("ymax") or "nan"),
                },
                "attributes": {},
            })
        return objects, "voc_xml"
    return None


def build_ground_truth(label_dir: Path, image_names: list[str], image_dir: Path) -> tuple[dict, dict]:
    import cv2

    images = []
    annotations = []
    counts: Counter[str] = Counter()
    image_counts: dict[str, set[int]] = defaultdict(set)
    colors: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    duplicate_boxes = 0
    outside_boxes = 0
    dimensions: set[tuple[int, int]] = set()
    label_formats: Counter[str] = Counter()
    for image_id, image_name in enumerate(image_names, start=1):
        image_path = image_dir / image_name
        frame = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if frame is None:
            raise FileNotFoundError(image_path)
        height, width = frame.shape[:2]
        dimensions.add((width, height))
        images.append({"id": image_id, "file_name": image_name, "width": width, "height": height})
        loaded = _objects_from_file(label_dir, image_name)
        if loaded is None:
            continue
        objects, label_format = loaded
        label_formats[label_format] += 1
        seen: set[tuple[str, float, float, float, float]] = set()
        for obj in objects:
            category = str(obj.get("category", ""))
            attrs = obj.get("attributes") or {}
            if category == "traffic light":
                colors[str(attrs.get("trafficLightColor", "unknown"))] += 1
            target = BDD_MAP.get(category)
            box = obj.get("box2d") or {}
            if target is None or not all(key in box for key in ("x1", "y1", "x2", "y2")):
                skipped[category] += 1
                continue
            x1, y1 = float(box["x1"]), float(box["y1"])
            x2, y2 = float(box["x2"]), float(box["y2"])
            if x2 <= x1 or y2 <= y1:
                skipped[category] += 1
                continue
            key = (target, x1, y1, x2, y2)
            duplicate_boxes += int(key in seen)
            seen.add(key)
            outside_boxes += int(x1 < 0 or y1 < 0 or x2 > width or y2 > height)
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": CATEGORY_IDS[target],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": box_area([x1, y1, x2, y2]),
                    "iscrowd": 0,
                    "bucket": size_bucket(box_area([x1, y1, x2, y2])),
                }
            )
            counts[target] += 1
            image_counts[target].add(image_id)
    categories = [{"id": CATEGORY_IDS[name], "name": name} for name in CLASSES]
    gt = {"images": images, "categories": categories, "annotations": annotations}
    stats = {
        "images_requested": len(image_names),
        "images_with_labels": sum(1 for image_name in image_names if _objects_from_file(label_dir, image_name) is not None),
        "label_formats": dict(sorted(label_formats.items())),
        "image_dimensions": [list(item) for item in sorted(dimensions)],
        "per_class_instances": {name: counts.get(name, 0) for name in CLASSES},
        "per_class_images": {name: len(image_counts.get(name, set())) for name in CLASSES},
        "skipped_category_counts": dict(sorted(skipped.items())),
        "traffic_light_color_counts": dict(sorted(colors.items())),
        "exact_duplicate_boxes": duplicate_boxes,
        "boxes_outside_image": outside_boxes,
        "source_note": "BDD100K Detection 2020 VOC conversion from Feng1909/bdd100k_voc_labels; original det_val.json was not directly reachable",
    }
    return gt, stats


def _model_names(model: dict[str, Any]) -> dict[int, str]:
    source = model.get("source_class_names_by_id") or model.get("class_names_by_id") or {}
    return {int(key): ALIASES.get(str(value), str(value)) for key, value in source.items()}


def _rows_for_model(model: dict[str, Any], image_names: list[str]) -> list[tuple[str, list[tuple]]]:
    by_image = model.get("detections", {}).get("by_image", {})
    rows = []
    for image_name in image_names:
        detections = []
        for item in by_image.get(image_name, []):
            box = item.get("bbox_xyxy")
            if not box or len(box) != 4:
                continue
            detections.append((
                int(item["class_id"]), float(box[0]), float(box[1]), float(box[2]), float(box[3]),
                float(item.get("confidence", 0.0)),
            ))
        rows.append((image_name, detections))
    return rows


def _rows_from_fp32(path: Path, image_names: list[str]) -> tuple[list[tuple[str, list[tuple]]], dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = {int(key): ALIASES.get(str(value), str(value)) for key, value in payload["source_class_names_by_id"].items()}
    rows = []
    by_image = payload.get("by_image", {})
    for image_name in image_names:
        values = []
        for item in by_image.get(image_name, []):
            box = item["bbox_xyxy"]
            values.append((int(item["class_id"]), *map(float, box), float(item["confidence"])))
        rows.append((image_name, values))
    return rows, names


def _micro(point: dict[str, Any]) -> dict[str, float | int]:
    total = {key: sum(int(point["classes"][name][key]) for name in CLASSES) for key in ("tp", "fp", "fn")}
    precision = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
    recall = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**total, "precision": precision, "recall": recall, "f1": f1}


def evaluate_rows(rows: list[tuple[str, list[tuple]]], model_names: dict[int, str], image_names: list[str], gt: dict, *, conf: float, score_floor: float, max_det: int) -> dict[str, Any]:
    detections, detection_stats = predictions_to_coco(rows, image_names, model_names, score_floor=score_floor)
    point = operating_point(detections, gt, conf=conf, iou_threshold=0.5)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = strip_internal_fields(gt)
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(detections) if detections else COCO()
    if not detections:
        coco_dt.dataset = {"images": coco_gt.dataset["images"], "categories": coco_gt.dataset["categories"], "annotations": []}
        coco_dt.createIndex()
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.catIds = [CATEGORY_IDS[name] for name in CLASSES]
    evaluator.params.maxDets = [1, 10, max_det]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    precision = evaluator.eval["precision"]
    per_class = {}
    for index, name in enumerate(CLASSES):
        per_class[name] = {
            "AP50": average_precision(precision, 0, AREA_INDEX["all"], index),
            "AP50_95": average_precision_over_iou(precision, range(10), AREA_INDEX["all"], index),
            **point["classes"][name],
        }
    return {
        "detections_at_score_floor": detection_stats,
        "AP50": float(evaluator.stats[1]),
        "AP50_95": float(evaluator.stats[0]),
        "operating_point": {"conf": conf, "iou_match": 0.5, **_micro(point)},
        "per_class": per_class,
        "classes": CLASSES,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fp32-reference", type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=100)
    args = parser.parse_args()
    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    image_names = list(payload["test_set"]["images"])
    gt, gt_stats = build_ground_truth(args.ground_truth_dir, image_names, args.images_dir)
    result: dict[str, Any] = {
        "protocol": {
            "images": len(image_names),
            "image_names_source": str(args.predictions),
            "conf": args.conf,
            "score_floor": args.score_floor,
            "nms_iou": payload.get("protocol", {}).get("nms_iou"),
            "match_iou": 0.5,
            "max_det": args.max_det,
            "class_mapping": "BDD names to seven classes; rider, train and traffic sign excluded",
        },
        "ground_truth": gt_stats,
        "models": {},
    }
    for model in payload["models"]:
        name = str(model["model"])
        result["models"][name] = evaluate_rows(_rows_for_model(model, image_names), _model_names(model), image_names, gt, conf=args.conf, score_floor=args.score_floor, max_det=args.max_det)
    if args.fp32_reference:
        rows, names = _rows_from_fp32(args.fp32_reference, image_names)
        result["models"]["BDD-YOLOv8n-FP32-reference"] = evaluate_rows(rows, names, image_names, gt, conf=args.conf, score_floor=0.0, max_det=args.max_det)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "ground_truth": gt_stats, "models": {key: value["operating_point"] for key, value in result["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
