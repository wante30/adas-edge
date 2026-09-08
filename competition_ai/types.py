from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class FrameResult:
    stream_id: int | str
    frame_id: int
    timestamp_ns: int | None
    inference_ms: float
    detections: list[Detection]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["detections"] = [d.to_dict() for d in self.detections]
        return data
