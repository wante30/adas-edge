"""B-05E low-rate traffic-light detection with tracked high-rate colour.

Detection supplies a physical light at ~1 Hz.  Between detections a small LK
optical-flow tracker updates the ROI on every real video frame, and the colour
classifier observes that tracked ROI.  The track does not pretend an old bbox
is a new detection: detection and colour have separate ages and TTLs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .traffic_light import TrafficLightState

DEFAULT_DETECTION_TTL_MS = 1200
DEFAULT_COLOR_TTL_MS = 500
CONFIRM_UPDATES = 2
MIN_TRACK_IOU = 0.30
MIN_CONFIDENCE = 0.25
COLOR_RATIO_THRESHOLD = 0.06


def _bbox(detection: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(detection, dict):
        return None
    raw = detection.get("bbox") or detection.get("bbox_xyxy")
    if raw is None or len(raw) != 4:
        return None
    try:
        box = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    return box if all(np.isfinite(box)) else None


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.0


@dataclass(frozen=True)
class LampColor:
    color: TrafficLightState
    confidence: float
    glow_ratio: float
    valid: bool


def classify_lamp_color(frame: np.ndarray, bbox) -> LampColor:
    """Classify a lamp ROI; reject dim/white/blue/mixed regions as UNKNOWN."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
        return LampColor(TrafficLightState.UNKNOWN, 0.0, 0.0, False)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (max(0, min(limit, int(round(v)))) for v, limit in zip(bbox, (w, h, w, h)))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return LampColor(TrafficLightState.UNKNOWN, 0.0, 0.0, False)
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    valid = (sat >= 90) & (val >= 130)
    total = int(valid.sum())
    if total < 3:
        return LampColor(TrafficLightState.UNKNOWN, 0.0, 0.0, False)
    masks = {
        TrafficLightState.RED: ((hue <= 10) | (hue >= 170)) & valid,
        TrafficLightState.YELLOW: (hue >= 14) & (hue <= 40) & valid,
        TrafficLightState.GREEN: (hue >= 40) & (hue <= 95) & valid,
    }
    ratios, blobs = {}, {}
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
    min_blob = max(3, int(roi.shape[0] * roi.shape[1] * 0.012))
    if ratio < COLOR_RATIO_THRESHOLD or ratio - second < 0.04:
        return LampColor(TrafficLightState.UNKNOWN, round(float(ratio), 3), ratio, False)
    if blobs.get(color, 0) < min_blob:
        return LampColor(TrafficLightState.UNKNOWN, round(float(ratio), 3), ratio, False)
    return LampColor(color, round(float(ratio), 3), ratio, True)


def track_bbox_lk(prev_gray, gray, bbox, *, max_corners=40, min_points=6,
                  fb_max=1.5, max_shift_fraction=0.30):
    """Track one ROI with pyramidal LK and forward-backward validation.

    Returns the shifted bbox, or None when the ROI has too few features, flow
    is inconsistent, or displacement is implausibly large.
    """
    if prev_gray is None or gray is None:
        return None
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
    x2, y2 = max(x1 + 4, min(w - 1, x2)), max(y1 + 4, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    mask = np.zeros_like(prev_gray)
    mask[y1:y2, x1:x2] = 255
    points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_corners,
                                     qualityLevel=0.01, minDistance=2, mask=mask)
    if points is None or len(points) < min_points:
        return None
    p1, status1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None,
                                              winSize=(15, 15), maxLevel=2)
    p0, status2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None,
                                              winSize=(15, 15), maxLevel=2)
    good = (status1.reshape(-1) == 1) & (status2.reshape(-1) == 1)
    if int(good.sum()) < min_points:
        return None
    old = points.reshape(-1, 2)[good]
    new = p1.reshape(-1, 2)[good]
    back = p0.reshape(-1, 2)[good]
    keep = np.linalg.norm(back - old, axis=1) <= fb_max
    if int(keep.sum()) < min_points:
        return None
    displacement = np.median(new[keep] - old[keep], axis=0)
    box_w, box_h = max(1.0, bbox[2] - bbox[0]), max(1.0, bbox[3] - bbox[1])
    if abs(displacement[0]) > box_w * max_shift_fraction or abs(displacement[1]) > box_h * max_shift_fraction:
        return None
    x1s, y1s = bbox[0] + float(displacement[0]), bbox[1] + float(displacement[1])
    x2s, y2s = bbox[2] + float(displacement[0]), bbox[3] + float(displacement[1])
    if x2s <= 0 or y2s <= 0 or x1s >= w or y1s >= h:
        return None
    clipped = (max(0.0, x1s), max(0.0, y1s), min(float(w), x2s), min(float(h), y2s))
    if (clipped[2] - clipped[0]) * (clipped[3] - clipped[1]) < 0.60 * box_w * box_h:
        return None
    return clipped


@dataclass
class TrafficLightTrack:
    track_id: int
    bbox: tuple[float, float, float, float]
    detection_confidence: float
    detection_frame_id: int
    detection_timestamp_ns: int
    color: TrafficLightState = TrafficLightState.UNKNOWN
    color_confidence: float = 0.0
    candidate: TrafficLightState = TrafficLightState.UNKNOWN
    candidate_count: int = 0
    color_frame_id: int = 0
    color_timestamp_ns: int = 0
    last_track_frame_id: int = 0
    track_valid: bool = True
    lost: bool = False

    def observe_color(self, observation: LampColor, frame_id: int, timestamp_ns: int,
                      confirm_updates: int) -> None:
        if observation.color == TrafficLightState.UNKNOWN:
            self.color = TrafficLightState.UNKNOWN
            self.color_confidence = 0.0
            self.candidate = TrafficLightState.UNKNOWN
            self.candidate_count = 0
            self.color_frame_id = 0
            self.color_timestamp_ns = 0
            return
        self.color_confidence = observation.confidence
        self.color_frame_id = int(frame_id)
        self.color_timestamp_ns = int(timestamp_ns)
        if observation.color == self.candidate:
            self.candidate_count += 1
        else:
            self.color = TrafficLightState.UNKNOWN
            self.candidate = observation.color
            self.candidate_count = 1
        if self.candidate_count >= max(1, int(confirm_updates)):
            self.color = observation.color


class TrafficLightTracker:
    """Low-rate detection + per-frame ROI tracking + colour evidence.

    ``update_detections`` is called at Specialist cadence.  ``update_frame`` is
    called for every new real video frame.  Duplicate/out-of-order frame ids and
    generation changes never refresh age or colour evidence.
    """

    def __init__(self, detection_ttl_ms: int = DEFAULT_DETECTION_TTL_MS,
                 color_ttl_ms: int = DEFAULT_COLOR_TTL_MS,
                 confirm_updates: int = CONFIRM_UPDATES, ttl_ms: int | None = None):
        self.detection_ttl_ms = max(1, int(ttl_ms if ttl_ms is not None else detection_ttl_ms))
        self.color_ttl_ms = max(1, int(color_ttl_ms))
        self.confirm_updates = max(1, int(confirm_updates))
        self._tracks: list[TrafficLightTrack] = []
        self._next_id = 1
        self._last_frame_id = -1
        self._last_timestamp_ns = -1
        self._generation = None
        self._prev_gray = None
        self._prev_frame_id = -1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._last_frame_id = -1
        self._last_timestamp_ns = -1
        self._prev_gray = None
        self._prev_frame_id = -1

    def _begin_frame(self, frame_id: int, timestamp_ns: int, generation) -> bool:
        if generation is not None and self._generation is not None and generation != self._generation:
            self.reset()
        self._generation = generation
        if frame_id <= self._last_frame_id or timestamp_ns <= self._last_timestamp_ns:
            return False
        self._last_frame_id = int(frame_id)
        self._last_timestamp_ns = int(timestamp_ns)
        return True

    def _match(self, bbox):
        best, index = MIN_TRACK_IOU, None
        for i, track in enumerate(self._tracks):
            overlap = _iou(bbox, track.bbox)
            if overlap > best:
                best, index = overlap, i
        return index

    def _prune(self, timestamp_ns):
        cutoff = timestamp_ns - self.detection_ttl_ms * 1_000_000
        self._tracks = [t for t in self._tracks if t.detection_timestamp_ns >= cutoff]

    def _records(self, timestamp_ns):
        records = []
        for track in self._tracks:
            detection_age = max(0.0, (timestamp_ns - track.detection_timestamp_ns) / 1_000_000)
            color_age = ((timestamp_ns - track.color_timestamp_ns) / 1_000_000
                         if track.color_timestamp_ns else 1e9)
            color_valid = (track.track_valid and not track.lost
                           and track.color != TrafficLightState.UNKNOWN
                           and color_age <= self.color_ttl_ms)
            detection_valid = detection_age <= self.detection_ttl_ms and not track.lost
            records.append({
                "source_frame_id": track.last_track_frame_id or track.detection_frame_id,
                "timestamp_ns": int(track.color_timestamp_ns or track.detection_timestamp_ns),
                "bbox": [round(float(v), 3) for v in track.bbox],
                "confidence": round(float(track.detection_confidence), 4),
                "color": track.color.value,
                "color_confidence": round(float(track.color_confidence), 4),
                "valid": bool(color_valid),
                "track_valid": bool(track.track_valid and not track.lost),
                "color_valid": bool(color_valid),
                "ttl_ms": self.detection_ttl_ms,
                "color_ttl_ms": self.color_ttl_ms,
                "detection_frame_id": track.detection_frame_id,
                "color_frame_id": track.color_frame_id,
                "detection_age_ms": round(detection_age, 1),
                "color_age_ms": round(color_age, 1),
                "detection_valid": bool(detection_valid),
                "track_id": track.track_id,
            })
        return records

    def update_detections(self, frame, detections, *, source_id="", source_frame_id=0,
                          timestamp_ns=0, generation=None):
        if not self._begin_frame(source_frame_id, timestamp_ns, generation):
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
            observation = classify_lamp_color(frame, bbox)
            index = self._match(bbox)
            if index is None:
                track = TrafficLightTrack(
                    self._next_id, bbox, confidence, int(source_frame_id), int(timestamp_ns),
                    last_track_frame_id=int(source_frame_id),
                )
                self._next_id += 1
                track.observe_color(observation, source_frame_id, timestamp_ns, self.confirm_updates)
                self._tracks.append(track)
            else:
                track = self._tracks[index]
                track.bbox = bbox
                track.detection_confidence = confidence
                track.detection_frame_id = int(source_frame_id)
                track.detection_timestamp_ns = int(timestamp_ns)
                track.last_track_frame_id = int(source_frame_id)
                track.track_valid = True
                track.lost = False
                track.observe_color(observation, source_frame_id, timestamp_ns, self.confirm_updates)
        self._prune(timestamp_ns)
        self._prev_gray = gray
        self._prev_frame_id = int(source_frame_id)
        return self._records(timestamp_ns)

    def update_frame(self, frame, *, source_id="", source_frame_id=0,
                     timestamp_ns=0, generation=None):
        if not self._begin_frame(source_frame_id, timestamp_ns, generation):
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for track in self._tracks:
            if track.last_track_frame_id == source_frame_id:
                continue
            shifted = track_bbox_lk(self._prev_gray, gray, track.bbox)
            if shifted is None:
                track.track_valid = False
                track.lost = True
                track.color = TrafficLightState.UNKNOWN
                track.color_confidence = 0.0
                track.candidate = TrafficLightState.UNKNOWN
                track.candidate_count = 0
                track.color_frame_id = 0
                track.color_timestamp_ns = 0
                continue
            track.bbox = shifted
            track.last_track_frame_id = int(source_frame_id)
            track.track_valid = True
            track.lost = False
            observation = classify_lamp_color(frame, shifted)
            track.observe_color(observation, source_frame_id, timestamp_ns, self.confirm_updates)
        self._prune(timestamp_ns)
        self._prev_gray = gray
        self._prev_frame_id = int(source_frame_id)
        return self._records(timestamp_ns)

    def current(self, timestamp_ns: int):
        """Return current records without consuming a video frame."""
        self._prune(int(timestamp_ns))
        return self._records(int(timestamp_ns))

    def update(self, frame, detections, *, source_id="", source_frame_id=0,
               timestamp_ns=0, generation=None):
        """Backward-compatible detection-only entry point."""
        return self.update_detections(
            frame, detections, source_id=source_id, source_frame_id=source_frame_id,
            timestamp_ns=timestamp_ns, generation=generation,
        )
