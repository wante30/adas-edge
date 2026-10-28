import builtins
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from tools.run_competition_console import _summarize_ms, digest
from tools.competition_console.navigation_reader import NavigationReader
from tools.competition_console.gtk_display import (
    choose_display_backend,
    fit_canvas,
    prepare_display_rgb,
    screen_to_canvas,
)
from tools.competition_console.view import NativeView, font_bbox, font_text_width
from tools.competition_console.rknn_detector import RknnConsoleDetector


class LegacyFont:
    def getsize(self, text):
        return len(text) * 10, 20

    def getoffset(self, text):
        return 0, 2


def test_pillow7_font_measurement_falls_back_to_legacy_methods():
    font = LegacyFont()

    assert font_text_width(font, "检测") == 20
    assert font_bbox(font, "检测", "mm") == (-10, -8, 10, 12)


def test_text_tile_draws_without_pillow8_anchor_argument(monkeypatch):
    view = NativeView()
    draw_calls = []

    class LegacyDraw:
        def text(self, xy, value, font, fill):
            draw_calls.append((xy, value, font, fill))

    font = LegacyFont()
    monkeypatch.setattr(view, "font", lambda size, numeric=False: font)
    monkeypatch.setattr("tools.competition_console.view.ImageDraw.Draw", lambda image: LegacyDraw())

    tile, left, top = view.text_tile("测试", 20, "white", False, "mm")

    assert tile.size == (20, 20)
    assert (left, top) == (-10, -8)
    assert draw_calls == [((0, -2), "测试", font, "white")]


def test_gtk_canvas_mapping_accounts_for_letterbox():
    transform = fit_canvas((1600, 900), (1024, 600))

    assert transform == (0.64, 0.0, 12.0, 1024.0, 576.0)
    assert screen_to_canvas(512, 300, transform) == (800, 450)
    assert screen_to_canvas(512, 8, transform) is None


def test_linux_without_opencv_gui_uses_gtk():
    assert choose_display_backend("auto", platform_name="linux", gui_backend="NONE") == "gtk"
    assert choose_display_backend("auto", platform_name="win32", gui_backend="NONE") == "opencv"
    assert choose_display_backend("opencv", platform_name="linux", gui_backend="NONE") == "opencv"


def test_gtk_rgb_input_skips_channel_conversion_and_bgr_input_matches():
    rgb = np.zeros((4, 6, 3), np.uint8)
    rgb[:, :, 0] = 210
    rgb[:, :, 2] = 35

    direct = prepare_display_rgb(rgb, (6, 4), input_rgb=True)
    converted = prepare_display_rgb(rgb[:, :, ::-1].copy(), (6, 4), input_rgb=False)

    assert direct is rgb
    assert np.array_equal(direct, converted)


def test_gtk_rgb_input_resizes_without_channel_swapping():
    rgb = np.zeros((4, 6, 3), np.uint8)
    rgb[:, :, 0] = 210
    rgb[:, :, 2] = 35

    result = prepare_display_rgb(rgb, (3, 2), input_rgb=True)

    assert result.shape == (2, 3, 3)
    assert result[:, :, 0].mean() > result[:, :, 2].mean()


def test_rknn_detector_adapter_preserves_console_contract(monkeypatch):
    calls = {}

    class FakeDetector:
        def __init__(self, model_path, input_size, conf_threshold, nms_threshold,
                     class_names=None, class_profile=None, use_rga_preprocess=False):
            calls["init"] = (model_path, input_size, conf_threshold, nms_threshold,
                             class_names, class_profile, use_rga_preprocess)
            calls["closed"] = False

        def infer(self, frame, stream_id, frame_id, timestamp_ns):
            calls["frame"] = (frame, stream_id, frame_id, timestamp_ns)
            item = SimpleNamespace(
                class_id=0,
                class_name="person",
                confidence=0.91,
                x1=10,
                y1=20,
                x2=30,
                y2=40,
            )
            return SimpleNamespace(inference_ms=12.5, detections=[item])

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr("competition_ai.detector.CompetitionDetector", FakeDetector)
    frame = object()
    detector = RknnConsoleDetector("traffic.rknn", (512, 192), specialist=True)

    detections, elapsed = detector.detect(frame)

    assert calls["init"] == ("traffic.rknn", (512, 192), 0.05, 0.70, None, None, False)
    assert calls["frame"] == (frame, 0, 0, None)
    assert detections == [{
        "class_name": "traffic light",
        "confidence": 0.91,
        "bbox": [10.0, 20.0, 30.0, 40.0],
        "source": "TL_SPECIALIST_RKNN",
    }]
    assert elapsed == 12.5
    detector.close()
    assert calls["closed"]
    RknnConsoleDetector("main.rknn", (512, 288))
    assert calls["init"] == ("main.rknn", (512, 288), 0.25, 0.45, None, None, True)


def test_navigation_reader_fails_closed_if_ocr_runtime_is_missing(monkeypatch):
    original_import = builtins.__import__
    reader = NavigationReader()
    monkeypatch.setattr(
        NavigationReader,
        "_arrow",
        staticmethod(lambda image: (12, 0.9, "left", (2, 2, 20, 40), np.ones((12, 32, 3), np.uint8))),
    )

    def import_without_rapidocr(name, *args, **kwargs):
        if name == "rapidocr":
            raise ModuleNotFoundError("No module named 'rapidocr'", name="rapidocr")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_rapidocr)
    observation = reader.update(np.zeros((540, 960, 3), np.uint8), 0, 0)

    assert observation.valid is False
    assert observation.direction == "none"
    assert observation.reason == "OCR_ENGINE_UNAVAILABLE:rapidocr"


def test_digest_works_without_python311_file_digest(tmp_path, monkeypatch):
    content = b"competition console portability"
    path = tmp_path / "payload.bin"
    path.write_bytes(content)
    monkeypatch.delattr(hashlib, "file_digest", raising=False)

    assert digest(path) == hashlib.sha256(content).hexdigest()


def test_timing_summary_reports_empty_and_populated_windows():
    assert _summarize_ms([]) == {
        "samples": 0,
        "mean": None,
        "median": None,
        "p95": None,
        "max": None,
    }
    summary = _summarize_ms([1.0, 2.0, 3.0])

    assert summary["samples"] == 3
    assert summary["mean"] == 2.0
    assert summary["median"] == 2.0
    assert summary["p95"] == pytest.approx(2.9)
    assert summary["max"] == 3.0
