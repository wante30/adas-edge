from __future__ import annotations

from datetime import datetime, timezone
import socket

import cv2
import numpy as np

from adas_edge.contracts import Frame, PixelFormat
from adas_edge.video.base import FrameSource


class UDPVideoSource(FrameSource):
    """Prototype JPEG-datagram receiver; FPGA packet framing is a later protocol decision."""

    def __init__(self, host: str, port: int, camera_id: str = "front", timeout_s: float = 1.0) -> None:
        self.camera_id = camera_id
        self._frame_id = 0
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((host, port))
        self._socket.settimeout(timeout_s)

    def read(self) -> Frame | None:
        try:
            payload, _ = self._socket.recvfrom(65_507)
        except TimeoutError:
            return None
        pixels = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if pixels is None:
            raise ValueError("UDP payload is not a complete JPEG image")
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
        self._socket.close()
