"""Behavior checks for the COCO-style evaluation helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]


def _load_helpers():
    path = REPO / "tools" / "coco_eval_helpers.py"
    spec = importlib.util.spec_from_file_location("coco_eval_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rows(entries: list[tuple[str, str, float, float, float, float]]) -> list[dict]:
    return [
        {"image_path": f"images/100k/val/{name}", "class_name": klass, "xmin": str(x1), "ymin": str(y1), "xmax": str(x2), "ymax": str(y2)}
        for name, klass, x1, y1, x2, y2 in entries
    ]


def test_categories_are_dense_and_one_based() -> None:
    helpers = _load_helpers()

    assert list(helpers.CATEGORY_IDS.values()) == list(range(1, 8))
    assert helpers.CATEGORY_IDS["car"] == 1
    assert len(helpers.CLASSES) == 7


def test_bdd_mapping_drops_rider_and_signs_but_keeps_the_seven() -> None:
    helpers = _load_helpers()

    assert helpers.BDD_MAP["pedestrian"] == "person"
    assert helpers.BDD_MAP["traffic light"] == "traffic light"
    for dropped in ("rider", "traffic sign", "train", "trailer", "other vehicle", "other person"):
        assert dropped not in helpers.BDD_MAP


def test_build_ground_truth_only_keeps_named_images_and_drops_unmapped() -> None:
    helpers = _load_helpers()
    rows = _rows(
        [
            ("a.jpg", "car", 0, 0, 100, 100),
            ("a.jpg", "rider", 0, 0, 10, 10),
            ("a.jpg", "traffic light", 10, 10, 14, 20),
            ("outside.jpg", "car", 0, 0, 10, 10),
            ("a.jpg", "car", 0, 0, 0, 0),
        ]
    )

    coco_gt, stats = helpers.build_ground_truth(rows, ["a.jpg", "b.jpg"], width=1280, height=720)

    assert [image["id"] for image in coco_gt["images"]] == [1, 2]
    assert coco_gt["images"][0]["file_name"] == "a.jpg"
    assert stats["per_class_instances"]["car"] == 1
    assert stats["per_class_instances"]["traffic light"] == 1
    assert stats["dropped_rows"] == 2  # rider, plus the zero-area car row
    assert stats["rows_outside_holdout"] == 1
    annotation = coco_gt["annotations"][0]
    assert annotation["bbox"] == [0.0, 0.0, 100.0, 100.0]
    assert annotation["category_id"] == helpers.CATEGORY_IDS["car"]
    assert annotation["bucket"] == "large"  # 100x100 = 10000 px, above the 96x96 medium ceiling


def test_size_bucket_matches_coco_area_thresholds() -> None:
    helpers = _load_helpers()

    assert helpers.size_bucket(32 * 32 - 1) == "small"
    assert helpers.size_bucket(32 * 32) == "medium"
    assert helpers.size_bucket(96 * 96 - 1) == "medium"
    assert helpers.size_bucket(96 * 96) == "large"


def test_strip_internal_fields_removes_bucket_only() -> None:
    helpers = _load_helpers()
    rows = _rows([("a.jpg", "car", 1, 2, 11, 12)])
    coco_gt, _ = helpers.build_ground_truth(rows, ["a.jpg"], width=1280, height=720)

    stripped = helpers.strip_internal_fields(coco_gt)

    assert "bucket" not in stripped["annotations"][0]
    assert stripped["annotations"][0]["bbox"] == [1.0, 2.0, 10.0, 10.0]
    # the source dict is left alone
    assert "bucket" in coco_gt["annotations"][0]


def test_predictions_to_coco_maps_names_applies_floor_and_counts_drops() -> None:
    helpers = _load_helpers()
    model_names = {0: "person", 2: "car", 3: "motorcycle", 4: "airplane", 9: "traffic light"}
    per_image = [
        (
            "a.jpg",
            [
                (2, 0.0, 0.0, 10.0, 10.0, 0.9),
                (2, 0.0, 0.0, 10.0, 10.0, 0.0005),   # below floor
                (4, 0.0, 0.0, 10.0, 10.0, 0.8),      # unmapped class
                (9, 5.0, 5.0, 5.0, 9.0, 0.7),        # degenerate
                (0, 1.0, 1.0, 5.0, 5.0, 0.6),
            ],
        )
    ]

    detections, stats = helpers.predictions_to_coco(per_image, ["a.jpg"], model_names, score_floor=0.001)

    assert [d["category_id"] for d in detections] == [helpers.CATEGORY_IDS["car"], helpers.CATEGORY_IDS["person"]]
    assert all(d["image_id"] == 1 for d in detections)
    assert stats["below_score_floor"] == 1
    assert stats["dropped_unmapped_or_degenerate"] == 2
    assert stats["per_class_detections"]["car"] == 1  # the second car row is under the floor
    assert stats["per_class_detections"]["person"] == 1


def test_operating_point_is_one_to_one_and_never_a_map() -> None:
    helpers = _load_helpers()
    rows = _rows([("a.jpg", "car", 0, 0, 10, 10), ("a.jpg", "car", 100, 100, 110, 110)])
    coco_gt, _ = helpers.build_ground_truth(rows, ["a.jpg"], width=1280, height=720)
    detections = [
        {"image_id": 1, "category_id": helpers.CATEGORY_IDS["car"], "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": helpers.CATEGORY_IDS["car"], "bbox": [0, 0, 10, 10], "score": 0.8},
    ]

    point = helpers.operating_point(detections, coco_gt, conf=0.25)

    # the second overlapping prediction is a false positive, not a second true positive
    assert point["classes"]["car"]["tp"] == 1
    assert point["classes"]["car"]["fp"] == 1
    assert point["classes"]["car"]["fn"] == 1
    assert point["classes"]["car"]["recall"] == pytest.approx(0.5)
    assert point["macro_recall"] == pytest.approx(0.5 / 7)


def test_operating_point_conf_floor_can_remove_every_prediction() -> None:
    helpers = _load_helpers()
    rows = _rows([("a.jpg", "traffic light", 0, 0, 8, 20)])
    coco_gt, _ = helpers.build_ground_truth(rows, ["a.jpg"], width=1280, height=720)
    detections = [
        {"image_id": 1, "category_id": helpers.CATEGORY_IDS["traffic light"], "bbox": [0, 0, 8, 20], "score": 0.07}
    ]

    kept = helpers.operating_point(detections, coco_gt, conf=0.25)
    loose = helpers.operating_point(detections, coco_gt, conf=0.05)

    assert kept["classes"]["traffic light"]["tp"] == 0
    assert kept["classes"]["traffic light"]["recall"] == 0.0
    assert loose["classes"]["traffic light"]["tp"] == 1
    assert loose["classes"]["traffic light"]["recall"] == 1.0


def test_operating_point_buckets_by_ground_truth_size() -> None:
    helpers = _load_helpers()
    rows = _rows([("a.jpg", "traffic light", 0, 0, 8, 20)])
    coco_gt, _ = helpers.build_ground_truth(rows, ["a.jpg"], width=1280, height=720)
    detections = [
        {"image_id": 1, "category_id": helpers.CATEGORY_IDS["traffic light"], "bbox": [0, 0, 8, 20], "score": 0.9}
    ]

    point = helpers.operating_point(detections, coco_gt, conf=0.25)

    assert point["size_buckets"]["small"]["gt"] == 1
    assert point["size_buckets"]["small"]["matched"] == 1
    assert point["size_buckets"]["medium"]["gt"] == 0


def test_average_precision_ignores_unset_recall_levels() -> None:
    numpy = pytest.importorskip("numpy")
    helpers = _load_helpers()
    # shape [T, R, K, A, M]; -1 marks a recall level with no detections
    precision = numpy.full((10, 3, 1, 4, 1), -1.0)
    precision[0, :, 0, 0, 0] = [1.0, 0.5, -1.0]

    assert helpers.average_precision(precision, 0, 0) == pytest.approx(0.75)
    assert helpers.average_precision(precision, 1, 0) == 0.0


def test_average_precision_over_iou_averages_the_requested_thresholds() -> None:
    numpy = pytest.importorskip("numpy")
    helpers = _load_helpers()
    precision = numpy.full((10, 2, 1, 4, 1), -1.0)
    precision[0, :, 0, 0, 0] = 1.0
    precision[1, :, 0, 0, 0] = 0.0

    assert helpers.average_precision_over_iou(precision, range(2), 0) == pytest.approx(0.5)
    assert helpers.average_precision_over_iou(precision, range(1), 0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        helpers.average_precision_over_iou(precision, range(0), 0)
