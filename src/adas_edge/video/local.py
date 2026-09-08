from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import cv2

from adas_edge.contracts import Frame, PixelFormat
from adas_edge.video.base import FrameSource


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class LocalFileSource(FrameSource):
    """Reads an image, video file, or a sorted directory of images."""

    def __init__(self, path: str | Path, camera_id: str = "front") -> None:
        self.path = Path(path)
        self.camera_id = camera_id
        self._frame_id = 0
        self._capture: cv2.VideoCapture | None = None
        self._images: list[Path] | None = None
        self._image_index = 0

        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if self.path.is_dir():
            self._images = sorted(
                item for item in self.path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES
            )
            if not self._images:
                raise ValueError(f"No supported images in {self.path}")
        elif self.path.suffix.lower() in IMAGE_SUFFIXES:
            self._images = [self.path]
        else:
            self._capture = cv2.VideoCapture(str(self.path))
            if not self._capture.isOpened():
                raise RuntimeError(f"Could not open video: {self.path}")

    def read(self) -> Frame | None:
        if self._images is not None:
            if self._image_index >= len(self._images):
                return None
            pixels = cv2.imread(str(self._images[self._image_index]), cv2.IMREAD_COLOR)
            self._image_index += 1
            if pixels is None:
                raise RuntimeError(f"Could not decode image: {self._images[self._image_index - 1]}")
        else:
            assert self._capture is not None
            ok, pixels = self._capture.read()
            if not ok:
                return None
        frame = Frame(
            camera_id=self.camera_id,
            frame_id=self._frame_id,
            timestamp=datetime.now(timezone.utc),
            pixels=pixels,
            pixel_format=PixelFormat.BGR888,
        )
        self._frame_id += 1
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
