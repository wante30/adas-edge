"""Stable data contracts between acquisition, perception, decision, and transport."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import cv2
import numpy as np


class PixelFormat(str, Enum):
    BGR888 = "BGR888"
    RGB888 = "RGB888"
    NV12 = "NV12"


@dataclass(frozen=True)
class Frame:
    camera_id: str
    frame_id: int
    timestamp: datetime
    pixels: np.ndarray
    pixel_format: PixelFormat

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        if self.pixel_format is PixelFormat.NV12:
            return int(self.pixels.shape[0] * 2 // 3)
        return int(self.pixels.shape[0])

    def as_bgr(self) -> np.ndarray:
        if self.pixel_format is PixelFormat.BGR888:
            return self.pixels
        if self.pixel_format is PixelFormat.RGB888:
            return cv2.cvtColor(self.pixels, cv2.COLOR_RGB2BGR)
        if self.pixel_format is PixelFormat.NV12:
            return cv2.cvtColor(self.pixels, cv2.COLOR_YUV2BGR_NV12)
        raise ValueError(f"Unsupported pixel format: {self.pixel_format}")

    def metadata(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp.isoformat(),
            "width": self.width,
            "height": self.height,
            "pixel_format": self.pixel_format.value,
        }


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
