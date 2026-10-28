"""Small front-camera lane candidate extractor for PC/RK3566 replay.

This is deliberately a CPU-friendly geometric adapter. It produces lane
*candidates* for the HMI and carries an experimental marker; it is not a lane
segmentation or road-model accuracy claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class LaneEstimate:
    lanes: dict[str, dict[str, Any]]
    coverage: float
    source: str = "OPENCV_HOUGH_EXPERIMENTAL"

    def as_metadata(self) -> dict[str, Any]:
        return {
            "left": dict(self.lanes.get("left", {})),
            "right": dict(self.lanes.get("right", {})),
            "coverage": round(float(self.coverage), 4),
            "source": self.source,
            "validation": "EXPERIMENTAL",
        }


class FrontLaneCandidate:
    """Track two forward lane boundaries with a bounded EMA."""

    def __init__(self, ema: float = 0.35, max_width: int = 320, update_stride: int = 1, max_missed: int = 6) -> None:
        self.ema = float(np.clip(ema, 0.0, 1.0))
        self.max_width = max(160, int(max_width))
        self.update_stride = max(1, int(update_stride))
        self.max_missed = max(1, int(max_missed))
        self._calls = 0
        self._last_estimate = LaneEstimate({}, 0.0, self.source_name)
        self._lines: dict[str, tuple[float, float] | None] = {"left": None, "right": None}
        self._missed = {"left": 0, "right": 0}

    def reset(self) -> None:
        self._lines = {"left": None, "right": None}
        self._missed = {"left": 0, "right": 0}
        self._calls = 0
        self._last_estimate = LaneEstimate({}, 0.0, self.source_name)

    @staticmethod
    def _candidate_lines(frame: np.ndarray) -> list[tuple[float, float, float]]:
        height, width = frame.shape[:2]
        y0 = int(height * 0.48)
        roi = frame[y0:, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # White/yellow road markings plus edges. The mask avoids most tree and
        # vehicle texture while retaining the painted lines.
        white = (hsv[:, :, 1] < 85) & (hsv[:, :, 2] > 125)
        yellow = (hsv[:, :, 0] >= 12) & (hsv[:, :, 0] <= 42) & (hsv[:, :, 1] > 65) & (hsv[:, :, 2] > 100)
        color = ((white | yellow).astype(np.uint8) * 255)
        edges = cv2.Canny(gray, 60, 150)
        mask = cv2.bitwise_or(edges, color)
        polygon = np.zeros_like(mask)
        cv2.fillPoly(
            polygon,
            [np.array([
                (int(width * 0.04), roi.shape[0] - 1),
                (int(width * 0.96), roi.shape[0] - 1),
                (int(width * 0.60), int(roi.shape[0] * 0.04)),
                (int(width * 0.40), int(roi.shape[0] * 0.04)),
            ], dtype=np.int32)],
            255,
        )
        mask = cv2.bitwise_and(mask, polygon)
        lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=max(18, int(width * 0.018)), minLineLength=max(24, int(width * 0.06)), maxLineGap=max(20, int(width * 0.035)))
        candidates: list[tuple[float, float, float]] = []
        if lines is None:
            return candidates
        # OpenCV may return either ``(N, 1, 4)`` or ``(N, 4)`` depending on
        # the build; flattening the leading dimensions keeps this runtime
        # portable between the PC wheel and RK3566's OpenCV build.
        for raw in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = (float(value) for value in raw)
            dx, dy = x2 - x1, y2 - y1
            if abs(dx) < 3 or abs(dy) < 6:
                continue
            slope = dy / dx
            if abs(slope) < 0.25 or abs(slope) > 8.0:
                continue
            # x_at_bottom in full-frame coordinates, plus line length.
            x_bottom = x1 + (roi.shape[0] - 1 - y1) * dx / dy
            if not (-0.25 * width <= x_bottom <= 1.25 * width):
                continue
            candidates.append((slope, x_bottom, float(np.hypot(dx, dy))))
        return candidates

    @staticmethod
    def _world_points(line: tuple[float, float], width: int, height: int) -> list[dict[str, float]]:
        slope, x_bottom = line
        # Image y samples are mapped to display-only ground distances. This is
        # a stable visualization mapping until camera calibration is available.
        zs = (4.0, 8.0, 16.0, 30.0, 48.0)
        ys = (height * 0.96, height * 0.88, height * 0.76, height * 0.63, height * 0.55)
        points = []
        for y, z in zip(ys, zs):
            # ``slope`` is dy/dx from the image line.  Convert to dx/dy for
            # sampling x at each image y; otherwise a steep lane line would
            # fan out in the wrong direction in the pseudo-world view.
            dx_per_dy = 1.0 / slope if abs(slope) > 1e-6 else 0.0
            x = x_bottom + (y - height * 0.96) * dx_per_dy
            lateral = (x / max(1.0, width) - 0.5) * 10.0
            points.append({"x": round(float(lateral), 3), "z": float(z)})
        return points

    def update(self, frame: np.ndarray) -> LaneEstimate:
        self._calls += 1
        if self._calls > 1 and (self._calls - 1) % self.update_stride:
            return self._last_estimate
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            self._last_estimate = LaneEstimate({}, 0.0, self.source_name)
            return self._last_estimate
        # Hough cost grows quickly with image area. The detector already uses
        # a 512x288 canvas, so a bounded 320px lane canvas keeps the CPU
        # adapter useful on both the PC and RK3566 without changing the
        # pseudo-world contract.
        if frame.shape[1] > self.max_width:
            scale = self.max_width / float(frame.shape[1])
            frame = cv2.resize(frame, (self.max_width, max(1, int(round(frame.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
        height, width = frame.shape[:2]
        candidates = self._candidate_lines(frame)
        selected: dict[str, tuple[float, float, float] | None] = {"left": None, "right": None}
        for item in candidates:
            slope, x_bottom, length = item
            if slope < 0 and x_bottom < width * 0.55:
                if selected["left"] is None or length > selected["left"][2]:
                    selected["left"] = item
            elif slope > 0 and x_bottom > width * 0.45:
                if selected["right"] is None or length > selected["right"][2]:
                    selected["right"] = item

        for side, item in selected.items():
            if item is None:
                self._missed[side] += 1
                if self._missed[side] > self.max_missed:
                    self._lines[side] = None
                continue
            current = (float(item[0]), float(item[1]))
            previous = self._lines[side]
            self._lines[side] = current if previous is None else tuple(self.ema * current[i] + (1.0 - self.ema) * previous[i] for i in range(2))
            self._missed[side] = 0

        lanes: dict[str, dict[str, Any]] = {}
        for side, line in self._lines.items():
            if line is not None:
                lanes[side] = {
                    "points": self._world_points(line, width, height),
                    "style": "unknown",
                    "confidence": round(0.55 if self._missed[side] == 0 else 0.35, 3),
                    "source": self.source_name,
                }
        self._last_estimate = LaneEstimate(lanes, len(lanes) / 2.0, self.source_name)
        return self._last_estimate

    @property
    def source_name(self) -> str:
        return "OPENCV_HOUGH_EXPERIMENTAL"
