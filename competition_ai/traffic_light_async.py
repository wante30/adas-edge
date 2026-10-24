"""B-05F asynchronous traffic-light tracker.

The video loop can be at frame 105 while the 1 Hz Specialist returns a result
for frame 100.  This module caches recent source frames, propagates the bbox
from the detection frame to the current frame with LK, and only then associates
it.  It does not pretend the inference completion time is the capture time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .traffic_light import TrafficLightState
from .traffic_light_tracking import (
    LampColor,
    classify_lamp_color,
    track_bbox_lk,
    _bbox,
    _iou,
)

DETECTION_TTL_MS = 1200
COLOR_TTL_MS = 500
CONFIRM_UPDATES = 2
MIN_CONFIDENCE = 0.25
MAX_CACHE_FRAMES = 60


@dataclass
class AsyncTrack:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    source_id: str
    generation: int | None
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

    def observe(self, observed: LampColor, frame_id: int, timestamp_ns: int,
                confirm: int) -> None:
        if observed.color == TrafficLightState.UNKNOWN:
            self.color = TrafficLightState.UNKNOWN
            self.color_confidence = 0.0
            self.candidate = TrafficLightState.UNKNOWN
            self.candidate_count = 0
            self.color_frame_id = 0
            self.color_timestamp_ns = 0
            return
        self.color_confidence = observed.confidence
        self.color_frame_id = int(frame_id)
        self.color_timestamp_ns = int(timestamp_ns)
        if observed.color == self.candidate:
            self.candidate_count += 1
        else:
            self.color = TrafficLightState.UNKNOWN
            self.candidate = observed.color
            self.candidate_count = 1
        if self.candidate_count >= max(1, int(confirm)):
            self.color = observed.color


class AsyncTrafficLightTracker:
    """Frame-cached tracker accepting late Specialist detections."""

    def __init__(self, detection_ttl_ms: int = DETECTION_TTL_MS,
                 color_ttl_ms: int = COLOR_TTL_MS,
                 confirm_updates: int = CONFIRM_UPDATES,
                 max_cache_frames: int = MAX_CACHE_FRAMES):
        self.detection_ttl_ms = int(detection_ttl_ms)
        self.color_ttl_ms = int(color_ttl_ms)
        self.confirm_updates = max(1, int(confirm_updates))
        self.max_cache_frames = max(2, int(max_cache_frames))
        self._frames: dict[int, np.ndarray] = {}
        self._bgr_frames: dict[int, np.ndarray] = {}
        self._timestamps: dict[int, int] = {}
        self._frame_order: list[int] = []
        self._last_current_frame = -1
        self._last_timestamp_ns = -1
        self._generation = None
        self._source_id = ""
        self._tracks: list[AsyncTrack] = []
        self._next_id = 1

    def reset(self) -> None:
        self._frames.clear()
        self._bgr_frames.clear()
        self._timestamps.clear()
        self._frame_order.clear()
        self._last_current_frame = -1
        self._last_timestamp_ns = -1
        self._generation = None
        self._source_id = ""
        self._tracks.clear()
        self._next_id = 1

    def push_frame(self, frame, *, source_frame_id: int, timestamp_ns: int,
                   generation=None, source_id="") -> bool:
        """Cache a real source frame.  Duplicate/out-of-order frames are refused."""
        if generation is not None and self._generation is not None and generation != self._generation:
            self.reset()
        if source_id and self._source_id and source_id != self._source_id:
            self.reset()
        frame_id = int(source_frame_id)
        if frame_id <= self._last_current_frame:
            return False
        if int(timestamp_ns) <= self._last_timestamp_ns:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._generation = generation
        self._source_id = str(source_id)
        self._last_current_frame = int(frame_id)
        self._last_timestamp_ns = int(timestamp_ns)
        self._frames[frame_id] = gray
        self._bgr_frames[frame_id] = frame
        self._timestamps[frame_id] = int(timestamp_ns)
        self._frame_order.append(frame_id)
        while len(self._frame_order) > self.max_cache_frames:
            old = self._frame_order.pop(0)
            self._frames.pop(old, None)
            self._bgr_frames.pop(old, None)
            self._timestamps.pop(old, None)
        return True

    def _match(self, bbox, used: set[int]):
        best, index = 0.30, None
        for i, track in enumerate(self._tracks):
            if i in used:
                continue
            overlap = _iou(bbox, track.bbox)
            if overlap > best:
                best, index = overlap, i
        return index

    def _advance_to_current(self, bbox, detection_frame_id: int, current_frame_id: int):
        if detection_frame_id == current_frame_id:
            return bbox
        if detection_frame_id not in self._frames:
            return None
        if detection_frame_id > current_frame_id:
            return None
        current = bbox
        previous = self._frames[detection_frame_id]
        # Walk cached source frames in order; no full-image search, only ROI LK.
        for frame_id in self._frame_order:
            if frame_id <= detection_frame_id or frame_id > current_frame_id:
                continue
            current_frame = self._frames[frame_id]
            shifted = track_bbox_lk(previous, current_frame, current)
            if shifted is None:
                return None
            current = shifted
            previous = current_frame
        return current

    def _prune(self, now_ns: int) -> None:
        cutoff = now_ns - self.detection_ttl_ms * 1_000_000
        self._tracks = [t for t in self._tracks if t.detection_timestamp_ns >= cutoff]

    def _records(self, now_ns: int):
        output = []
        for track in self._tracks:
            detection_age = max(0.0, (now_ns - track.detection_timestamp_ns) / 1_000_000)
            color_age = ((now_ns - track.color_timestamp_ns) / 1_000_000
                         if track.color_timestamp_ns else 1e9)
            color_valid = (track.track_valid and not track.lost
                           and track.color != TrafficLightState.UNKNOWN
                           and color_age <= self.color_ttl_ms)
            effective = track.color.value if color_valid else "UNKNOWN"
            output.append({
                "source_id": track.source_id,
                "generation": track.generation,
                "track_id": track.track_id,
                "bbox": [round(float(v), 3) for v in track.bbox],
                "confidence": round(float(track.confidence), 4),
                "detection_frame_id": track.detection_frame_id,
                "color_frame_id": track.color_frame_id,
                "detection_timestamp_ns": track.detection_timestamp_ns,
                "color_timestamp_ns": track.color_timestamp_ns,
                "detection_age_ms": round(detection_age, 1),
                "color_age_ms": round(color_age, 1),
                "track_valid": bool(track.track_valid and not track.lost),
                "color_valid": bool(color_valid),
                "effective_color": effective,
                "debug_color": track.color.value,
                "color_confidence": round(float(track.color_confidence), 4),
                "ttl_ms": self.detection_ttl_ms,
                "color_ttl_ms": self.color_ttl_ms,
                "source_frame_id": track.last_track_frame_id or track.detection_frame_id,
            })
        return output

    def update_frame(self, frame, *, source_frame_id: int, timestamp_ns: int,
                     generation=None, source_id=""):
        if not self.push_frame(frame, source_frame_id=source_frame_id,
                               timestamp_ns=timestamp_ns, generation=generation,
                               source_id=source_id):
            return []
        current = self._frames[int(source_frame_id)]
        for track in self._tracks:
            if track.last_track_frame_id == source_frame_id:
                continue
            previous = self._frames.get(track.last_track_frame_id)
            if previous is None:
                track.track_valid = False
                track.lost = True
                track.color = TrafficLightState.UNKNOWN
                continue
            shifted = track_bbox_lk(previous, current, track.bbox)
            if shifted is None:
                track.track_valid = False
                track.lost = True
                track.color = TrafficLightState.UNKNOWN
                track.candidate = TrafficLightState.UNKNOWN
                track.candidate_count = 0
                track.color_frame_id = 0
                track.color_timestamp_ns = 0
                continue
            track.bbox = shifted
            track.last_track_frame_id = int(source_frame_id)
            track.track_valid = True
            track.lost = False
            track.observe(classify_lamp_color(frame, shifted), source_frame_id,
                          timestamp_ns, self.confirm_updates)
        self._prune(int(timestamp_ns))
        return self._records(int(timestamp_ns))

    def update_detections(self, detections, *, detection_frame_id: int,
                          detection_timestamp_ns: int, current_frame_id: int,
                          current_timestamp_ns: int, generation=None,
                          source_id=""):
        """Associate a late detection with the current frame safely."""
        if generation is not None and self._generation is not None and generation != self._generation:
            self.reset()
        if source_id and self._source_id and source_id != self._source_id:
            self.reset()
        detection_frame_id = int(detection_frame_id)
        current_frame_id = int(current_frame_id)
        if detection_frame_id > current_frame_id:
            return {"rejected": "detection_frame_in_future", "records": []}
        if current_frame_id not in self._frames:
            return {"rejected": "current_frame_not_cached", "records": []}
        if detection_timestamp_ns <= 0 or current_timestamp_ns - detection_timestamp_ns > self.detection_ttl_ms * 1_000_000:
            return {"rejected": "detection_expired", "records": []}
        current_bgr = self._bgr_frames[current_frame_id]
        used: set[int] = set()
        rejected = []
        for item in detections or ():
            if not isinstance(item, dict) or str(item.get("class_name", "")).replace("_", " ").lower() != "traffic light":
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
            advanced = self._advance_to_current(bbox, detection_frame_id, current_frame_id)
            if advanced is None:
                rejected.append({"reason": "tracking_from_detection_failed", "detection_frame_id": detection_frame_id})
                continue
            index = self._match(advanced, used)
            if index is None:
                track = AsyncTrack(
                    self._next_id, advanced, confidence, str(source_id), generation,
                    detection_frame_id, int(detection_timestamp_ns),
                    last_track_frame_id=current_frame_id,
                )
                self._next_id += 1
                track.observe(classify_lamp_color(current_bgr, advanced), current_frame_id,
                              current_timestamp_ns, self.confirm_updates)
                self._tracks.append(track)
            else:
                track = self._tracks[index]
                if detection_frame_id < track.detection_frame_id:
                    rejected.append({"reason": "older_detection_than_track", "detection_frame_id": detection_frame_id})
                    continue
                track.bbox = advanced
                track.confidence = confidence
                track.detection_frame_id = detection_frame_id
                track.detection_timestamp_ns = int(detection_timestamp_ns)
                track.last_track_frame_id = current_frame_id
                track.track_valid = True
                track.lost = False
                track.observe(classify_lamp_color(current_bgr, advanced), current_frame_id,
                              current_timestamp_ns, self.confirm_updates)
            used.add(len(self._tracks) - 1 if index is None else index)
        self._prune(int(current_timestamp_ns))
        return {"rejected": rejected, "records": self._records(int(current_timestamp_ns))}
