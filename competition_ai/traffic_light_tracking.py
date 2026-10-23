"""B-05B multi-light traffic-light colour and temporal tracking.

The Specialist model only supplies a traffic-light box.  This module keeps the
colour decision separate from detection, requires a lamp-like bright/saturated
blob, stabilises a colour for two observations, retains each physical light
through detector gaps, and returns one record per light for A-line integration.

No state here triggers a vehicle decision; callers must not use UNKNOWN or an
unassociated light as a red-light decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .traffic_light import TrafficLightState

DEFAULT_TTL_MS = 1200
CONFIRM_UPDATES = 2
MIN_TRACK_IOU = 0.30
COLOR_RATIO_THRESHOLD = 0.06
MIN_CONFIDENCE = 0.25


def _bbox(detection: Any) -> tuple[float, float, float, float] | None:
    raw = detection.get("bbox") or detection.get("bbox_xyxy") if isinstance(detection, dict) else None
    if raw is None or len(raw) != 4:
        return None
    try:
        box = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    return box if all(np.isfinite(box)) else None


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class LampColor:
    color: TrafficLightState
    confidence: float
    glow_ratio: float
    valid: bool


def classify_lamp_color(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> LampColor:
    """Classify one lamp ROI with a bright/saturated blob gate.

    Blue road signs are deliberately outside the red/yellow/green hue ranges.
    A dim, white, multi-colour or low-area ROI returns UNKNOWN.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
        return LampColor(TrafficLightState.UNKNOWN, 0.0, 0.0, False)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (max(0, min(limit, int(round(value)))) for value, limit in zip(bbox, (width, height, width, height)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return LampColor(TrafficLightState.UNKNOWN, 0.0, 0.0, False)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return LampColor(TrafficLightState.UNKNOWN, 0.0, 0.0, False)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    valid = (saturation >= 90) & (value >= 130)
    total = int(valid.sum())
    if total < 3:
        return LampColor(TrafficLightState.UNKNOWN, 0.0, 0.0, False)
    masks = {
        TrafficLightState.RED: (((hue <= 10) | (hue >= 170)) & valid),
        TrafficLightState.YELLOW: ((hue >= 14) & (hue <= 40) & valid),
        TrafficLightState.GREEN: ((hue >= 40) & (hue <= 95) & valid),
    }
    ratios = {}
    blobs = {}
    for color, mask in masks.items():
        mask_u8 = mask.astype(np.uint8)
        count = int(mask_u8.sum())
        ratios[color] = count / max(1, total)
        if count:
            components, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
            blobs[color] = int(stats[1:, cv2.CC_STAT_AREA].max()) if components > 1 else 0

    ordered = sorted(ratios.items(), key=lambda item: item[1], reverse=True)
    color, ratio = ordered[0]
    second = ordered[1][1] if len(ordered) > 1 else 0.0
    # Require both a fraction of the ROI and one coherent glowing blob. This
    # rejects a few saturated leaves or sign pixels around a dim housing.
    min_blob = max(3, int(roi.shape[0] * roi.shape[1] * 0.012))
    if ratio < COLOR_RATIO_THRESHOLD or ratio - second < 0.04:
        return LampColor(TrafficLightState.UNKNOWN, round(float(ratio), 3), ratio, False)
    if blobs.get(color, 0) < min_blob:
        return LampColor(TrafficLightState.UNKNOWN, round(float(ratio), 3), ratio, False)
    return LampColor(color, round(float(ratio), 3), ratio, True)


@dataclass
class TrafficLightTrack:
    track_id: int
    bbox: tuple[float, float, float, float]
    detection_confidence: float
    color: TrafficLightState = TrafficLightState.UNKNOWN
    color_confidence: float = 0.0
    candidate: TrafficLightState = TrafficLightState.UNKNOWN
    candidate_count: int = 0
    last_seen_ns: int = 0

    def update(self, bbox, confidence: float, observation: LampColor, now_ns: int) -> None:
        self.bbox = bbox
        self.detection_confidence = confidence
        self.last_seen_ns = now_ns
        if observation.color == TrafficLightState.UNKNOWN:
            self.color = TrafficLightState.UNKNOWN
            self.color_confidence = 0.0
            self.candidate = TrafficLightState.UNKNOWN
            self.candidate_count = 0
            return
        self.color_confidence = observation.confidence
        if observation.color == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = observation.color
            self.candidate_count = 1
        if self.candidate_count >= CONFIRM_UPDATES:
            self.color = observation.color


class TrafficLightTracker:
    """Associate detections to physical lights and keep per-light state."""

    def __init__(self, ttl_ms: int = DEFAULT_TTL_MS, confirm_updates: int = CONFIRM_UPDATES):
        self.ttl_ms = max(1, int(ttl_ms))
        self.confirm_updates = max(1, int(confirm_updates))
        self._tracks: list[TrafficLightTrack] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, frame, detections, *, source_id: str, source_frame_id: int, timestamp_ns: int) -> list[dict[str, Any]]:
        now_ns = int(timestamp_ns)
        candidates = []
        for item in detections or ():
            if not isinstance(item, dict):
                continue
            if str(item.get("class_name", "")).replace("_", " ").lower() != "traffic light":
                continue
            bbox = _bbox(item)
            if bbox is None:
                continue
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(confidence) or confidence < MIN_CONFIDENCE:
                continue
            candidates.append((bbox, confidence))

        used: set[int] = set()
        for bbox, confidence in candidates:
            best_index, best_iou = None, MIN_TRACK_IOU
            for index, track in enumerate(self._tracks):
                if index in used:
                    continue
                overlap = _iou(bbox, track.bbox)
                if overlap > best_iou:
                    best_index, best_iou = index, overlap
            observation = classify_lamp_color(frame, bbox)
            if best_index is None:
                track = TrafficLightTrack(self._next_id, bbox, confidence, last_seen_ns=now_ns)
                self._next_id += 1
                track.update(bbox, confidence, observation, now_ns)
                self._tracks.append(track)
                used.add(len(self._tracks) - 1)
            else:
                self._tracks[best_index].update(bbox, confidence, observation, now_ns)
                used.add(best_index)

        cutoff = now_ns - self.ttl_ms * 1_000_000
        self._tracks = [track for track in self._tracks if track.last_seen_ns >= cutoff]
        records = []
        for track in self._tracks:
            age_ms = max(0.0, (now_ns - track.last_seen_ns) / 1_000_000)
            records.append({
                "source_id": str(source_id),
                "source_frame_id": int(source_frame_id),
                "timestamp_ns": int(track.last_seen_ns),
                "bbox": [round(float(value), 3) for value in track.bbox],
                "confidence": round(float(track.detection_confidence), 4),
                "color": track.color.value,
                "color_confidence": round(float(track.color_confidence), 4),
                "valid": bool(track.color != TrafficLightState.UNKNOWN and age_ms <= self.ttl_ms),
                "ttl_ms": self.ttl_ms,
                "track_id": track.track_id,
                "age_ms": round(age_ms, 1),
            })
        return records
