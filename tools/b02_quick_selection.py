"""Summarize cached BDD predictions at fixed confidence thresholds."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.coco_eval_helpers import (
    CLASSES,
    CATEGORY_IDS,
    MODEL_MAP,
    build_ground_truth,
    iou,
    operating_point,
    predictions_to_coco,
    read_bdd_rows,
)

DEFAULT_THRESHOLDS = (0.25, 0.35, 0.45, 0.55)
GROUPS = {
    "all_target_classes": tuple(CLASSES),
    "vehicles": ("car", "truck", "bus", "motorcycle"),
    "pedestrian": ("person",),
    "traffic_light": ("traffic light",),
}


def aggregate_counts(class_metrics: dict[str, dict], classes: Iterable[str]) -> dict[str, int | float]:
    """Aggregate per-class TP/FP/FN, then calculate micro precision/recall/F1."""
    counts = {
        key: sum(int(class_metrics[name][key]) for name in classes)
        for key in ("tp", "fp", "fn")
    }
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def load_prediction_dump(path: Path) -> tuple[list[tuple[str, list[tuple]]], dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [(entry["image"], [tuple(row) for row in entry["detections"]]) for entry in payload["per_image"]]
    names = {int(key): value for key, value in payload["model_names"].items()}
    return rows, names


def evaluate_dump(
    predictions: list[tuple[str, list[tuple]]],
    model_names: dict[int, str],
    coco_gt: dict,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
) -> dict:
    detections, _ = predictions_to_coco(predictions, [name for name, _ in predictions], model_names, score_floor=0.0)
    result = {}
    for threshold in thresholds:
        point = operating_point(detections, coco_gt, conf=threshold, iou_threshold=0.5)
        per_class = point["classes"]
        result[f"{threshold:.2f}"] = {
            group: aggregate_counts(per_class, classes)
            for group, classes in GROUPS.items()
        }
    return result


def high_confidence_false_positives(
    predictions: list[tuple[str, list[tuple]]],
    model_names: dict[int, str],
    coco_gt: dict,
    threshold: float = 0.25,
    limit: int = 20,
) -> list[dict]:
    """Return the highest-scoring unmatched boxes for focused visual inspection."""
    image_ids = {image["file_name"]: image["id"] for image in coco_gt["images"]}
    truth = defaultdict(list)
    for annotation in coco_gt["annotations"]:
        x, y, width, height = annotation["bbox"]
        truth[(annotation["image_id"], annotation["category_id"])].append(
            [x, y, x + width, y + height]
        )

    false_positives = []
    for image_name, rows in predictions:
        image_id = image_ids[image_name]
        grouped = defaultdict(list)
        for row in rows:
            model_class, x1, y1, x2, y2, score = row
            target = MODEL_MAP.get(model_names.get(int(model_class), ""))
            if target is not None and score >= threshold:
                grouped[CATEGORY_IDS[target]].append((float(score), [x1, y1, x2, y2]))

        for category_id, candidates in grouped.items():
            ground_truth = truth[(image_id, category_id)]
            used = set()
            for score, box in sorted(candidates, key=lambda item: -item[0]):
                overlaps = [
                    (iou(box, target), index)
                    for index, target in enumerate(ground_truth)
                    if index not in used
                ]
                best_iou, best_index = max(overlaps, default=(0.0, -1))
                max_iou = max((iou(box, target) for target in ground_truth), default=0.0)
                if best_iou >= 0.5:
                    used.add(best_index)
                    continue
                false_positives.append(
                    {
                        "image": image_name,
                        "class": next(name for name, value in CATEGORY_IDS.items() if value == category_id),
                        "score": score,
                        "box_xyxy": [round(float(value), 1) for value in box],
                        "max_iou_same_class": round(max_iou, 3),
                        "same_class_ground_truth": len(ground_truth),
                    }
                )
    return sorted(false_positives, key=lambda entry: -entry["score"])[:limit]


def parse_prediction_arg(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bdd-root", type=Path, required=True)
    parser.add_argument("--predictions", action="append", type=parse_prediction_arg, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    loaded = {label: load_prediction_dump(path) for label, path in args.predictions}
    image_names = [name for name, _ in next(iter(loaded.values()))[0]]
    for label, (rows, _) in loaded.items():
        if [name for name, _ in rows] != image_names:
            raise ValueError(f"prediction dump {label!r} does not contain the same ordered images")
    ground_truth, stats = build_ground_truth(
        read_bdd_rows(args.bdd_root / "bdd100k.csv"), image_names, args.width, args.height
    )
    report = {
        "images": len(image_names),
        "ground_truth": stats,
        "match_iou": 0.5,
        "thresholds": list(DEFAULT_THRESHOLDS),
        "groups": {name: list(classes) for name, classes in GROUPS.items()},
        "models": {
            label: {
                "threshold_metrics": evaluate_dump(rows, model_names, ground_truth),
                "high_confidence_false_positives": high_confidence_false_positives(
                    rows, model_names, ground_truth
                ),
            }
            for label, (rows, model_names) in loaded.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
