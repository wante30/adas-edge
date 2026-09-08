from datetime import datetime, timezone

import cv2
import numpy as np

from adas_edge.video.local import LocalFileSource


def test_local_image_source_emits_bgr_frame_with_camera_metadata(tmp_path) -> None:
    image_path = tmp_path / "one.png"
    expected = np.full((3, 4, 3), (11, 22, 33), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), expected)

    source = LocalFileSource(image_path, camera_id="left")
    frame = source.read()

    assert frame is not None
    assert frame.camera_id == "left"
    assert frame.frame_id == 0
    assert frame.width == 4
    assert frame.height == 3
    assert np.array_equal(frame.as_bgr(), expected)
    assert source.read() is None


def test_local_directory_source_uses_sorted_images(tmp_path) -> None:
    for name, value in [("b.png", 2), ("a.png", 1)]:
        assert cv2.imwrite(str(tmp_path / name), np.full((2, 2, 3), value, dtype=np.uint8))

    source = LocalFileSource(tmp_path)
    first = source.read()
    second = source.read()

    assert first is not None and second is not None
    assert int(first.pixels[0, 0, 0]) == 1
    assert int(second.pixels[0, 0, 0]) == 2
