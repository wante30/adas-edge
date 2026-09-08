from __future__ import annotations

from abc import ABC, abstractmethod

from adas_edge.contracts import Frame


class FrameSource(ABC):
    """Transport-independent producer of timestamped video frames."""

    @abstractmethod
    def read(self) -> Frame | None:
        """Return one frame, or None after finite input is exhausted."""

    @abstractmethod
    def close(self) -> None:
        """Release source resources."""
