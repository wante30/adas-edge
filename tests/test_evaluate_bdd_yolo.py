"""Behavior checks for the frozen-BDD-holdout evaluator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_tool():
    path = Path(__file__).parents[1] / "tools" / "evaluate_bdd_yolo.py"
    spec = importlib.util.spec_from_file_location("evaluate_bdd_yolo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_imgsz_accepts_square_and_16_9() -> None:
    tool = _load_tool()

    assert tool.parse_imgsz("320") == 320
    assert tool.parse_imgsz(416) == 416
    # Ultralytics expects (height, width) for a rectangular input.
    assert tool.parse_imgsz("512x288") == (288, 512)
    assert tool.parse_imgsz("512×288") == (288, 512)
    for invalid in ("512x300", "0x288", "8x8"):
        with pytest.raises(ValueError):
            tool.parse_imgsz(invalid)


def test_size_buckets_follow_coco_area_thresholds() -> None:
    tool = _load_tool()

    assert tool.bucket_of([0, 0, 20, 20]) == "small"
    assert tool.bucket_of([0, 0, 50, 50]) == "medium"
    assert tool.bucket_of([0, 0, 200, 200]) == "large"
    # A 100x10 sliver has 1000 px, below the 1024 px small threshold.
    assert tool.bucket_of([0, 0, 100, 10]) == "small"


def test_match_class_is_one_to_one_and_reports_fp() -> None:
    tool = _load_tool()
    truth = [[0, 0, 10, 10], [100, 100, 110, 110]]
    preds = [[0, 0, 10, 10], [101, 100, 110, 110], [500, 500, 510, 510]]

    false_positives, true_positives, matched = tool.match_class(preds, truth, 0.5)

    assert true_positives == 2
    assert false_positives == 1
    assert matched == {0, 1}


def test_evaluate_reports_macro_recall_and_small_object_recall() -> None:
    tool = _load_tool()
    predictions = [{"car": [[0, 0, 20, 20]], "person": [], "traffic light": []}]
    ground_truth = [
        {
            "car": [[0, 0, 20, 20], [0, 0, 200, 200]],
            "person": [[10, 10, 20, 20]],
            "bicycle": [[30, 30, 40, 40]],
        }
    ]

    report = tool.evaluate(predictions, ground_truth)

    assert report["classes"]["car"]["tp"] == 1
    assert report["classes"]["car"]["fn"] == 1
    assert report["classes"]["person"]["fn"] == 1
    # Only car recalls anything (1 of 2 GT boxes), averaged over all 7 classes.
    assert report["classes"]["car"]["recall"] == pytest.approx(0.5)
    assert report["macro_recall"] == pytest.approx(0.5 / 7)
    # Small GT: car 20x20 (matched), person 10x10, bicycle 10x10. Large GT: car 200x200.
    assert report["size_buckets"]["small"]["gt"] == 3
    assert report["size_buckets"]["small"]["matched"] == 1
    assert report["size_buckets"]["large"]["gt"] == 1
    assert report["size_buckets"]["large"]["recall"] == 0.0
    assert report["per_class_size_recall"]["bicycle/small"] == 0.0


def test_evaluate_handles_empty_ground_truth_without_dividing_by_zero() -> None:
    tool = _load_tool()

    report = tool.evaluate([{"car": [[0, 0, 10, 10]]}], [{}])

    assert report["classes"]["car"]["fp"] == 1
    assert report["classes"]["car"]["recall"] == 0.0
    assert report["macro_recall"] == 0.0
    assert all(bucket["recall"] == 0.0 for bucket in report["size_buckets"].values())


def test_render_table_includes_macro_and_buckets() -> None:
    tool = _load_tool()
    report = tool.evaluate([{"car": [[0, 0, 10, 10]]}], [{"car": [[0, 0, 10, 10]]}])
    report.update(
        {
            "label": "BASE-512x288",
            "model": "yolo11n.pt",
            "size": "512x288",
            "geometry": "fixed",
            "observed_input_shapes": [[3, 288, 512]],
            "images": 1,
        }
    )

    table = tool.render_table(report)

    assert "macro" in table
    assert "size bucket" in table
    assert "traffic light" in table
    # the geometry the network actually received is never implicit
    assert "observed network input: 512x288 (3ch)" in table
    assert "geometry: fixed" in table
