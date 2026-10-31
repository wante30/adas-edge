"""Unit tests for the B-05N multiclass decode/evaluation helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.b05n_tl_eval import decode_class, prepare_full  # noqa: E402


def test_multiclass_decode_uses_class_eight_channel() -> None:
    raw = np.zeros((1, 14, 2), dtype=np.float32)
    # Two anchors: TL score 0.9 and 0.1; box xywh in network pixels.
    raw[0, 12, 0] = 0.9
    raw[0, 12, 1] = 0.1
    raw[0, :4, 0] = [256, 144, 20, 30]
    raw[0, :4, 1] = [100, 100, 10, 10]
    boxes = decode_class(raw, 8, 0.25, 0.5)
    assert len(boxes) == 1
    assert boxes[0]["confidence"] > 0.89
    assert len(boxes[0]["bbox"]) == 4


def test_full_frame_input_is_288x512() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    tensor, transform = prepare_full(frame, 512, 288)
    assert tensor.shape == (1, 3, 288, 512)
    assert tensor.dtype == np.float32
    assert abs(transform["scale_x"] - 0.4) < 1e-6
    assert abs(transform["scale_y"] - 0.4) < 1e-6
