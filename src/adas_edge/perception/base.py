from __future__ import annotations

from abc import ABC, abstractmethod

from adas_edge.contracts import Detection, Frame


class Detector(ABC):
    """Replaceable object detector; implementations own their runtime."""

    model_name: str

    @abstractmethod
    def detect(self, frame: Frame) -> list[Detection]:
        """Run object detection on one frame in BGR/RGB/NV12 form."""
