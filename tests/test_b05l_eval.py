"""Unit tests for the B-05L one-to-one evaluator and curve accounting."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.b05l_eval import average_precision, confidence_curves, fp_budget, match_image, ranked_matching  # noqa: E402


def test_one_prediction_cannot_match_two_gt() -> None:
    gt = [
        {"bbox": [0, 0, 10, 10], "width_px": 10},
        {"bbox": [8, 0, 18, 10], "width_px": 10},
    ]
    predictions = [{"bbox": [0, 0, 10, 10], "confidence": 0.9}]
    pairs, tp, fp, fn = match_image(gt, predictions, conf=0.25, iou_threshold=0.5)
    assert len(pairs) == 1
    assert (tp, fp, fn) == (1, 0, 1)


def test_duplicate_predictions_count_as_false_positives() -> None:
    gt = [{"bbox": [0, 0, 10, 10], "width_px": 10}]
    predictions = [
        {"bbox": [0, 0, 10, 10], "confidence": 0.9},
        {"bbox": [0, 0, 10, 10], "confidence": 0.8},
    ]
    _, tp, fp, fn = match_image(gt, predictions, conf=0.25, iou_threshold=0.5)
    assert (tp, fp, fn) == (1, 1, 0)


def test_ranked_matching_is_global_by_confidence() -> None:
    gt = {"a": [{"bbox": [0, 0, 10, 10]}], "b": [{"bbox": [0, 0, 10, 10]}]}
    predictions = {
        "a": [{"bbox": [0, 0, 10, 10], "confidence": 0.5}],
        "b": [{"bbox": [0, 0, 10, 10], "confidence": 0.9}],
    }
    rows = ranked_matching(gt, predictions, floor=0.001, iou_threshold=0.5)
    assert [row["image"] for row in rows] == ["b", "a"]
    assert all(row["tp"] == 1 for row in rows)
    ap, _ = average_precision(rows, total_gt=2)
    assert abs(ap - 1.0) < 1e-9


def test_confidence_curve_counts_predictions_at_threshold() -> None:
    rows = [
        {"confidence": 0.9, "tp": 1, "bbox": [0, 0, 1, 1]},
        {"confidence": 0.6, "tp": 0, "bbox": [0, 0, 1, 1]},
    ]
    curve = confidence_curves(rows, total_gt=2, images=1)
    point = next(item for item in curve["thresholds"] if abs(item["conf"] - 0.6) < 1e-9)
    assert point["tp"] == 1
    assert point["fp"] == 1
    assert abs(point["recall"] - 0.5) < 1e-9
    assert abs(point["fp_per_image"] - 1.0) < 1e-9


def test_fp_budget_reports_threshold_and_recall() -> None:
    rows = [
        {"confidence": 0.9, "tp": 1},
        {"confidence": 0.8, "tp": 0},
        {"confidence": 0.7, "tp": 1},
    ]
    result = fp_budget(rows, total_gt=2, images=1, budgets=[0.0, 1.0])
    assert result[0]["threshold"] == 0.9
    assert abs(result[0]["recall"] - 0.5) < 1e-9
    assert result[1]["threshold"] == 0.7
    assert abs(result[1]["recall"] - 1.0) < 1e-9
