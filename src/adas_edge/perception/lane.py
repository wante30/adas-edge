from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from adas_edge.contracts import Frame


@dataclass(frozen=True)
class LaneState:
    lane_type: str | None
    confidence: float


class LanePerception(ABC):
    @abstractmethod
    def infer(self, frame: Frame) -> LaneState:
        """Return independent lane semantics, not a detector-specific tensor."""


class MockLanePerception(LanePerception):
    def __init__(self, lane_type: str | None = None) -> None:
        self._lane_type = lane_type

    def infer(self, frame: Frame) -> LaneState:
        return LaneState(lane_type=self._lane_type, confidence=0.0)
