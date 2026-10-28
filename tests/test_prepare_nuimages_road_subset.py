"""Behavior checks for the nuImages road-subset preparation helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_tool():
    path = Path(__file__).parents[1] / "tools" / "prepare_nuimages_road_subset.py"
    spec = importlib.util.spec_from_file_location("prepare_nuimages_road_subset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves ``cls.__module__`` through sys.modules, so the module
    # has to be registered before exec_module runs the @dataclass decorators.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_coco_mode_labels_use_original_coco_ids() -> None:
    tool = _load_tool()

    # pedestrian.adult / bicycle / bus.rigid / truck keep their COCO ids so the
    # pretrained 80-class head survives the fine-tune untouched.
    assert tool.class_index("person", "coco80") == 0
    assert tool.class_index("bicycle", "coco80") == 1
    assert tool.class_index("car", "coco80") == 2
    assert tool.class_index("motorcycle", "coco80") == 3
    assert tool.class_index("bus", "coco80") == 5
    assert tool.class_index("truck", "coco80") == 7
    assert tool.class_index("traffic light", "coco80") == 9
    assert len(tool.COCO_NAMES) == 80


def test_compact_mode_densifies_indices() -> None:
    tool = _load_tool()

    assert [tool.class_index(name, "compact7") for name in tool.TARGET_CLASSES] == list(range(7))


def test_clean_boxes_maps_nuimages_categories_and_drops_noise() -> None:
    tool = _load_tool()
    boxes = [
        [400, 200, 800, 400],   # vehicle.car -> 16
        [0, 0, 1, 1],           # degenerate, dropped
        [900, 400, 300, 200],   # overhangs the right/bottom edge, clipped
        [10, 10, 40, 40],       # movable_object.barrier -> unmapped, dropped
    ]
    categories = [16, 14, 20, 8]

    labels = tool.clean_boxes(boxes, categories, 1600, 900, "coco80")

    assert len(labels) == 2
    assert [label[0] for label in labels] == [2, 3]
    assert all(0.0 <= value <= 1.0 for label in labels for value in label[1:])
    # The clipped motorcycle box must stay inside the frame.
    _, cx, cy, w, h = labels[1]
    assert cx + w / 2 <= 1.0 and cy + h / 2 <= 1.0


def test_clean_boxes_rejects_non_positive_frame() -> None:
    tool = _load_tool()

    assert tool.clean_boxes([[0, 0, 10, 10]], [16], 0, 900, "coco80") == []


def _record(tool, name: str, targets: list[str]) -> "object":
    labels = []
    for index, target in enumerate(targets):
        labels.append((tool.class_index(target, "coco80"), 0.5, 0.5, 0.1, 0.1 + index * 1e-6))
    return tool.Record(name=name, labels=labels)


def test_select_balanced_meets_rare_floors_before_common_frames() -> None:
    tool = _load_tool()
    records = []
    for index in range(4000):
        records.append(_record(tool, f"car_{index}.jpg", ["car"]))
    for index in range(50):
        records.append(_record(tool, f"bicycle_{index}.jpg", ["bicycle", "car"]))
    for index in range(50):
        records.append(_record(tool, f"bus_{index}.jpg", ["bus"]))

    chosen = tool.select_balanced(
        records,
        limit=200,
        seed=7,
        min_instances={1: 40, 5: 30, 3: 0, 7: 0},
        rarity_order=[1, 5, 3, 7, 0, 2],
        monotone_cap=0.35,
    )

    names = {record.name for record in chosen}
    assert len(chosen) == 200
    bicycle_frames = sum(1 for name in names if name.startswith("bicycle_"))
    bus_frames = sum(1 for name in names if name.startswith("bus_"))
    assert bicycle_frames >= 40
    assert bus_frames >= 30
    # car-only frames must not be allowed to crowd out the rare classes
    assert sum(1 for name in names if name.startswith("car_")) <= 200 - bicycle_frames - bus_frames


def test_select_balanced_rejects_impossible_limit() -> None:
    tool = _load_tool()
    records = [_record(tool, "a.jpg", ["car"])]

    with pytest.raises(ValueError):
        tool.select_balanced(records, limit=5, seed=1, min_instances={}, rarity_order=[2], monotone_cap=0.5)


def test_camera_of_reads_the_encoding_in_the_filename() -> None:
    tool = _load_tool()

    assert tool.camera_of("n003-2018-01-03-12-03-23+0800__CAM_FRONT__1514952221396202.jpg") == "CAM_FRONT"
    assert tool.camera_of("") == "unknown"
