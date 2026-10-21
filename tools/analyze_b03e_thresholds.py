"""Analyze cached B-03D predictions at several confidence thresholds.

This deliberately performs no model inference.  It reuses the low-confidence
RKNN cache and the exact B-03D VOC ground-truth subset so threshold changes are
measured with identical boxes, class mapping, IoU matching, and NMS output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.coco_eval_helpers import operating_point, predictions_to_coco  # noqa: E402
from tools.evaluate_b03d_accuracy import (  # noqa: E402
    _model_names,
    _rows_for_model,
    _rows_from_fp32,
    build_ground_truth,
)


def micro(point: dict[str, Any]) -> dict[str, float | int]:
    total = {
        key: sum(int(point["classes"][name][key]) for name in point["classes"])
        for key in ("tp", "fp", "fn")
    }
    precision = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
    recall = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**total, "precision": precision, "recall": recall, "f1": f1}


def evaluate(
    rows: list[tuple[str, list[tuple]]],
    names: dict[int, str],
    image_names: list[str],
    gt: dict,
    confs: list[float],
    score_floor: float,
) -> dict[str, Any]:
    detections, detection_stats = predictions_to_coco(
        rows, image_names, names, score_floor=score_floor
    )
    results: dict[str, Any] = {
        "score_floor": score_floor,
        "detections_at_score_floor": detection_stats,
        "thresholds": {},
    }
    for conf in confs:
        point = operating_point(detections, gt, conf=conf, iou_threshold=0.5)
        results["thresholds"][str(conf)] = {
            "conf": conf,
            "overall": micro(point),
            "classes": point["classes"],
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fp32-reference", type=Path)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--confs", type=float, nargs="+", default=[0.25, 0.35, 0.45, 0.55])
    args = parser.parse_args()

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    image_names = list(payload["test_set"]["images"])
    gt, gt_stats = build_ground_truth(args.ground_truth_dir, image_names, args.images_dir)
    result: dict[str, Any] = {
        "protocol": {
            "images": len(image_names),
            "prediction_cache": str(args.predictions),
            "score_floor": args.score_floor,
            "thresholds": args.confs,
            "nms_iou": payload.get("protocol", {}).get("nms_iou"),
            "match_iou": 0.5,
            "gt_source": str(args.ground_truth_dir),
            "note": "Historical B-03D holdout; threshold sweep is a screening result, not an independent test.",
        },
        "ground_truth": gt_stats,
        "models": {},
    }
    for model in payload["models"]:
        result["models"][str(model["model"])] = evaluate(
            _rows_for_model(model, image_names),
            _model_names(model),
            image_names,
            gt,
            args.confs,
            args.score_floor,
        )

    if args.fp32_reference:
        rows, names = _rows_from_fp32(args.fp32_reference, image_names)
        result["models"]["BDD-YOLOv8n-FP32-reference"] = evaluate(
            rows, names, image_names, gt, [0.25], args.score_floor
        )
        result["models"]["BDD-YOLOv8n-FP32-reference"]["note"] = (
            "Reference cache was generated at conf=0.25; no low-confidence FP32 cache, "
            "so this is not a complete AP/threshold sweep."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
