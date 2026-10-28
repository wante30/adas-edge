"""Behavior checks for the single-stream RKNN profiling helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_profiler_module():
    path = Path(__file__).parents[1] / "tools" / "profile_rknn_single_stream.py"
    spec = importlib.util.spec_from_file_location("profile_rknn_single_stream", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_letterbox_returns_static_nhwc_input_and_scale_metadata() -> None:
    profiler = _load_profiler_module()
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    result, scale, pad_x, pad_y = profiler.letterbox_bgr(image, 320)

    assert result.shape == (320, 320, 3)
    assert result.dtype == np.uint8
    assert scale == 1 / 6
    assert pad_x == 0
    assert pad_y == 70


def test_summary_includes_each_latency_statistic() -> None:
    profiler = _load_profiler_module()

    summary = profiler.summarize([1.0, 2.0, 3.0, 4.0])

    assert summary == {"mean": 2.5, "median": 2.5, "p95": 3.85, "std": 1.118}


def test_nms_delegates_to_opencv(monkeypatch) -> None:
    profiler = _load_profiler_module()
    import cv2

    calls = {}

    def fake_nms_boxes(boxes, scores, score_threshold, nms_threshold):
        calls["boxes"] = boxes
        calls["scores"] = scores
        calls["score_threshold"] = score_threshold
        calls["nms_threshold"] = nms_threshold
        return np.asarray([[1], [0]], dtype=np.int32)

    monkeypatch.setattr(cv2.dnn, "NMSBoxes", fake_nms_boxes)
    boxes = np.asarray([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=np.float32)
    result = profiler._nms(boxes, np.asarray([0.8, 0.7], dtype=np.float32), 0.45)

    assert result == [1, 0]
    assert calls["score_threshold"] == 0.0
    assert calls["nms_threshold"] == 0.45


def test_gstreamer_pipeline_contains_mpp_decoder() -> None:
    profiler = _load_profiler_module()
    pipeline = profiler.gstreamer_pipeline("clip.mp4")
    assert "filesrc location=clip.mp4" in pipeline
    assert "mppvideodec" in pipeline
    assert "video/x-raw,format=BGR" in pipeline


def test_raw_decode_handles_filtered_candidate_vectors() -> None:
    profiler = _load_profiler_module()
    regression = np.zeros((1, 64, 1, 1), dtype=np.float32)
    classes = np.full((1, 80, 1, 1), -10.0, dtype=np.float32)
    classes[0, 2, 0, 0] = 1.0
    empty_classes = np.full((1, 80, 1, 1), -10.0, dtype=np.float32)

    result = profiler.decode_yolo11_raw(
        [regression, classes, regression, empty_classes, regression, empty_classes],
        input_size=8,
        confidence_threshold=0.25,
        iou_threshold=0.45,
        scale=1.0,
        pad_x=0,
        pad_y=0,
    )

    assert len(result) == 1
    assert result[0][5] == 2.0


def test_normalize_input_shape_is_pure_about_square_and_rectangular() -> None:
    profiler = _load_profiler_module()

    assert profiler.normalize_input_shape(320) == (320, 320)
    assert profiler.normalize_input_shape((512, 288)) == (512, 288)
    assert profiler.normalize_input_shape([512, 288]) == (512, 288)
    # The decoder stays a pure function: grid divisibility is asserted there, and
    # the 32-pixel model policy lives in validate_model_shape. Tiny shapes must
    # remain legal so unit-scale tensors can be decoded.
    assert profiler.normalize_input_shape(8) == (8, 8)
    for invalid in (0, -32, (512,), (512, 288, 3), (0, 288)):
        try:
            profiler.normalize_input_shape(invalid)
        except ValueError:
            continue
        raise AssertionError(f"{invalid!r} must be rejected")


def test_validate_model_shape_enforces_rknn_multiple_of_32() -> None:
    profiler = _load_profiler_module()

    assert profiler.validate_model_shape(320) == (320, 320)
    assert profiler.validate_model_shape("512x288") == (512, 288)
    assert profiler.validate_model_shape("512X288") == (512, 288)
    assert profiler.validate_model_shape("512x288") == (512, 288)
    for invalid in (300, "512x300", "0x288", "-16x288", "512x288x3", "abc"):
        try:
            profiler.validate_model_shape(invalid)
        except ValueError:
            continue
        raise AssertionError(f"{invalid!r} must be rejected")


def test_letterbox_supports_non_square_model_input() -> None:
    profiler = _load_profiler_module()
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    result, scale, pad_x, pad_y = profiler.letterbox_bgr(image, (512, 288))

    assert result.shape == (288, 512, 3)
    assert scale == 512 / 1920
    assert pad_x == 0
    assert pad_y == (288 - round(1080 * 512 / 1920)) // 2
