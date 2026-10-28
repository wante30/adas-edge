"""Behavior checks for the frozen-holdout comparison matrix assembly."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_tool():
    path = Path(__file__).parents[1] / "tools" / "run_road_matrix.py"
    spec = importlib.util.spec_from_file_location("run_road_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_report(label: str, recall: float, model: str = "yolo11n.pt", size: str = "320") -> dict:
    classes = {
        name: {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": recall, "f1": recall}
        for name in ["car", "truck", "bus", "person", "motorcycle", "bicycle", "traffic light"]
    }
    return {
        "label": label,
        "model": model,
        "size": size,
        "images": 1000,
        "conf": 0.25,
        "iou_match": 0.5,
        "classes": classes,
        "macro_recall": recall,
        "macro_f1": recall,
        "size_buckets": {
            bucket: {"gt": 10, "matched": int(recall * 10), "recall": recall} for bucket in ("small", "medium", "large")
        },
        "per_class_size_recall": {"car/small": recall},
        "geometry": "fixed",
        "observed_input_shapes": [[3, 288, 512]],
        "wall_ms_per_image": 15.0,
        "detections": 1234,
    }


def test_default_matrix_covers_the_five_planned_configurations() -> None:
    tool = _load_tool()

    entries = tool.default_matrix(Path("yolo11n.pt"), Path("road"))

    assert [entry["label"] for entry in entries] == [
        "BASE-320",
        "BASE-416",
        "BASE-512x288",
        "Road-FT-320",
        "Road-FT-512x288",
    ]
    by_label = {entry["label"]: entry for entry in entries}
    # the COCO baselines share one checkpoint; the road pilots use their own
    assert by_label["BASE-320"]["model"] == by_label["BASE-512x288"]["model"]
    assert by_label["Road-FT-512x288"]["model"] != by_label["Road-FT-320"]["model"]
    assert by_label["BASE-512x288"]["imgsz"] == "512x288"
    # every row is measured with the frozen deployment canvas, not minrect
    assert {entry["geometry"] for entry in entries} == {"fixed"}


def test_assemble_writes_a_matrix_with_per_class_and_bucket_sections(tmp_path: Path) -> None:
    tool = _load_tool()
    reports = [
        ("BASE-320", _fake_report("BASE-320", 0.10)),
        ("Road-FT-512x288", _fake_report("Road-FT-512x288", 0.30, model="road-ft-512x288.pt", size="512x288")),
    ]

    matrix = tool.assemble(reports, tmp_path)

    assert matrix["macro_recall"] == {"BASE-320": 0.10, "Road-FT-512x288": 0.30}
    assert matrix["labels"] == ["BASE-320", "Road-FT-512x288"]
    assert matrix["holdout_images"] == 1000
    assert matrix["models"]["Road-FT-512x288"] == "road-ft-512x288.pt"

    assert matrix["geometry"]["Road-FT-512x288"] == "fixed"
    assert matrix["observed_input_shapes"]["BASE-320"] == [[3, 288, 512]]
    assert matrix["wall_ms_per_image"]["BASE-320"] == 15.0

    markdown = (tmp_path / "matrix.md").read_text(encoding="utf-8")
    assert "BASE-320 R / F1" in markdown
    assert "traffic light" in markdown
    assert "small-object recall" in markdown or "Small-object recall" in markdown
    assert "measured network input" in markdown
    assert "pixel budget" in markdown
    assert "0.300" in markdown
    assert (tmp_path / "matrix.json").exists()
