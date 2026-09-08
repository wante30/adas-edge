from __future__ import annotations

from datetime import datetime, timezone

import cv2

from adas_edge.contracts import Frame, PixelFormat
from adas_edge.video.base import FrameSource


class USBCameraSource(FrameSource):
    """OpenCV/V4L2/DirectShow USB-camera adapter."""

    def __init__(self, device_index: int = 0, camera_id: str = "front") -> None:
        self.camera_id = camera_id
        self._frame_id = 0
        self._capture = cv2.VideoCapture(device_index)
        if not self._capture.isOpened():
            self._capture.release()
            raise RuntimeError(f"Could not open USB camera {device_index}")

    def read(self) -> Frame | None:
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
        self._capture.release()
