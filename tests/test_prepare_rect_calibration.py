"""Behavior checks for the rectangular RKNN calibration generator.

The point of the tool is that calibration samples must be the *deployed* canvas in
the *deployed* channel order, and must never come from the frozen holdout. Each of
those three claims is asserted here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parents[1]


def _load_tool():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    path = REPO / "tools" / "prepare_rect_calibration.py"
    spec = importlib.util.spec_from_file_location("prepare_rect_calibration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_frames(directory: Path, count: int, size: tuple[int, int] = (1280, 720)) -> None:
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        # a distinct colour per frame so sampling order is observable
        frame[:, :, 0] = (index * 37) % 256
        frame[:, :, 1] = (index * 53) % 256
        frame[:, :, 2] = (index * 71) % 256
        cv2.imwrite(str(directory / f"frame_{index:03d}.jpg"), frame)


def test_guard_refuses_the_frozen_holdout(tmp_path: Path) -> None:
    tool = _load_tool()
    holdout = tmp_path / "holdout"
    _write_frames(holdout, 1)
    nested = holdout / "images" / "100k" / "val"
    _write_frames(nested, 1)

    for source in (holdout, nested):
        with pytest.raises(SystemExit):
            tool.guard_source(source, holdout)


def test_guard_allows_a_source_outside_the_holdout(tmp_path: Path) -> None:
    tool = _load_tool()
    holdout = tmp_path / "holdout"
    holdout.mkdir()
    other = tmp_path / "road_frames"
    _write_frames(other, 1)

    tool.guard_source(other, holdout)  # must not raise


def test_generator_writes_exact_canvas_npy_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()
    frames = tmp_path / "frames"
    _write_frames(frames, 6)
    out = tmp_path / "calib"
    manifest = tmp_path / "calib.txt"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_rect_calibration.py",
            "--source", str(frames),
            "--width", "512",
            "--height", "288",
            "--limit", "4",
            "--stride", "2",
            "--output-dir", str(out),
            "--manifest", str(manifest),
            "--frozen-holdout", str(tmp_path / "unused_holdout"),
        ],
    )
    tool.main()

    arrays = sorted(out.glob("*.npy"))
    # 6 frames at stride 2 -> indices 0, 2, 4
    assert len(arrays) == 3
    assert len(list(out.glob("*.png"))) == 3
    for path in arrays:
        array = np.load(path)
        # NCHW, because RKNN's .npy loader rejects NHWC outright
        assert array.shape == (1, 3, 288, 512)
        assert array.dtype == np.uint8

    lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 3
    assert all(line.endswith(".npy") for line in lines)


def test_generator_rejects_an_illegal_canvas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()
    frames = tmp_path / "frames"
    _write_frames(frames, 2)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_rect_calibration.py",
            "--source", str(frames),
            "--width", "640",
            "--height", "360",  # not a multiple of 32
            "--output-dir", str(tmp_path / "calib"),
            "--manifest", str(tmp_path / "calib.txt"),
            "--frozen-holdout", str(tmp_path / "unused_holdout"),
        ],
    )
    with pytest.raises(SystemExit):
        tool.main()


def test_generator_report_records_channel_order_and_no_holdout_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()
    frames = tmp_path / "frames"
    _write_frames(frames, 3)
    report_path = tmp_path / "report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_rect_calibration.py",
            "--source", str(frames),
            "--width", "320",
            "--height", "192",
            "--limit", "3",
            "--stride", "1",
            "--output-dir", str(tmp_path / "calib"),
            "--manifest", str(tmp_path / "calib.txt"),
            "--report", str(report_path),
            "--frozen-holdout", str(tmp_path / "unused_holdout"),
        ],
    )
    tool.main()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["canvas"] == "320x192"
    assert report["canvas_pixels"] == 320 * 192
    assert report["channel_order"] == "RGB"
    assert report["layout"] == "NCHW"
    assert report["array_shape"] == [1, 3, 192, 320]
    assert report["expected_by_toolkit"] == "(1, 3, 192, 320)"
    assert report["samples"] == 3
    assert report["frozen_holdout_used"] is False


def test_stride_sampling_is_deterministic(tmp_path: Path) -> None:
    tool = _load_tool()
    frames = tmp_path / "frames"
    _write_frames(frames, 6)

    first = list(tool.iter_source_frames(frames, 2))
    second = list(tool.iter_source_frames(frames, 2))

    assert len(first) == 3
    assert len(second) == 3
    assert [name for name, _ in first] == ["frame_000.jpg", "frame_002.jpg", "frame_004.jpg"]
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(first, second))
