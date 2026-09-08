from datetime import datetime, timezone

import numpy as np

from adas_edge.contracts import Frame, PixelFormat


def test_rgb888_frame_converts_to_bgr_without_losing_metadata() -> None:
    frame = Frame(
        camera_id="front",
        frame_id=7,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        pixels=np.array([[[255, 0, 0]]], dtype=np.uint8),
        pixel_format=PixelFormat.RGB888,
    )

    bgr = frame.as_bgr()

    assert bgr.tolist() == [[[0, 0, 255]]]
    assert frame.camera_id == "front"
    assert frame.frame_id == 7


def test_nv12_frame_converts_to_three_channel_bgr() -> None:
    frame = Frame(
        camera_id="front",
        frame_id=1,
        timestamp=datetime.now(timezone.utc),
        pixels=np.full((3, 2), 128, dtype=np.uint8),
        pixel_format=PixelFormat.NV12,
    )

    assert frame.as_bgr().shape == (2, 2, 3)
