"""Behavior checks for narrowing the pretrained COCO head to seven classes."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]


def _load_tool():
    path = REPO / "tools" / "transfer_coco_head.py"
    spec = importlib.util.spec_from_file_location("transfer_coco_head", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_all_seven_adas_classes_have_a_coco_source_row() -> None:
    tool = _load_tool()

    assert set(tool.DEFAULT_CLASSES) == set(tool.COCO_SOURCES)
    assert tool.selected_rows() == [0, 1, 2, 3, 5, 7, 9]


@pytest.mark.skipif(
    not (REPO / "yolo11n.pt").exists(),
    reason="yolo11n.pt is required for the head-transfer checks",
)
def test_narrow_head_preserves_the_pretrained_class_logits() -> None:
    tool = _load_tool()
    from ultralytics import YOLO

    source = YOLO(str(REPO / "yolo11n.pt")).model
    target = tool.build_narrow_head(source)

    assert target.model[-1].nc == 7
    assert target.model[-1].no == 7 + target.model[-1].reg_max * 4
    assert [target.names[i] for i in range(7)] == list(tool.DEFAULT_CLASSES)

    verification = tool.verify_narrow_head(source, target)
    assert verification["all_match"] is True, verification
    assert all(entry["logits"] == 7 for entry in verification["branches"].values())
    # the class trunk is shared bit for bit; only the terminal 1x1 conv is narrowed
    assert all(entry["trunk_max_abs_diff"] == 0.0 for entry in verification["branches"].values())
    assert all(entry["max_abs_diff"] <= 1e-5 for entry in verification["branches"].values())


@pytest.mark.skipif(
    not (REPO / "yolo11n.pt").exists(),
    reason="yolo11n.pt is required for the head-transfer checks",
)
def test_only_the_terminal_class_convs_change() -> None:
    tool = _load_tool()
    from ultralytics import YOLO

    source = YOLO(str(REPO / "yolo11n.pt")).model
    target = tool.build_narrow_head(source)

    report = tool.untouched_tensors(source, target)

    assert report["unexpected_differences"] == []
    # three branches x (weight, bias) at the terminal 1x1 conv
    assert len(report["narrowed_tensors"]) == 6
    assert all(".cv3." in key and key.endswith((".2.weight", ".2.bias")) for key in report["narrowed_tensors"])
    # the whole backbone, neck, box branch and class trunk remain untouched
    assert report["identical_tensors"] > 200


@pytest.mark.skipif(
    not (REPO / "yolo11n.pt").exists(),
    reason="yolo11n.pt is required for the head-transfer checks",
)
def test_narrowed_checkpoint_reloads_as_a_seven_class_model(tmp_path: Path) -> None:
    tool = _load_tool()
    from ultralytics import YOLO

    out = tmp_path / "yolo11n-road7.pt"
    summary = tool.build_checkpoint(REPO / "yolo11n.pt", out, verify=True)

    assert out.exists()
    assert summary["verification"]["all_match"] is True
    assert summary["state_dict"]["unexpected_differences"] == []

    reloaded = YOLO(str(out))
    assert reloaded.model.model[-1].nc == 7
    assert [reloaded.names[index] for index in range(7)] == list(tool.DEFAULT_CLASSES)
