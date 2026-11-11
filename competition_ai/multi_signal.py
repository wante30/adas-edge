"""Independent Q4 lamp observations. No control permissions are produced here.

Times use capture CLOCK_MONOTONIC nanoseconds; a new publication never renews
an old observation. Backends consume an original-resolution BGR ROI.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Protocol

import cv2
import numpy as np

from .traffic_light import classify_traffic_light

COLORS = {"RED", "YELLOW", "GREEN", "UNKNOWN"}
SHAPES = {"ROUND", "LEFT_ARROW", "STRAIGHT_ARROW", "RIGHT_ARROW", "UTURN_ARROW", "UNKNOWN"}
DIRECTIONS = ("left", "straight", "right")


@dataclass(frozen=True)
class Attributes:
    color: str = "UNKNOWN"
    shape: str = "UNKNOWN"
    color_confidence: float = 0.0
    shape_confidence: float = 0.0
    reason: str = "UNOBSERVED"
    backend: str = "arm_hsv_contour_v1"
    evidence: dict = field(default_factory=dict)


class AttributeBackend(Protocol):
    def classify(self, roi: np.ndarray) -> Attributes: ...


class HSVBackend:
    """Reuse existing HSV color logic, with an additional mixed-glow veto."""

    def classify(self, roi):
        if roi.size == 0 or min(roi.shape[:2]) < 4:
            return Attributes(reason="INSUFFICIENT_PIXELS")
        # Bound native work without upsampling tiny lamps into false detail.
        if max(roi.shape[:2]) > 192:
            return Attributes(reason="ROI_TOO_LARGE")
        h, w = roi.shape[:2]
        result = classify_traffic_light(roi, {"bbox": [0, 0, w, h], "confidence": 1.0})
        color = result.value
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        valid = (sat >= 90) & (val >= 130)
        masks = {
            "RED": valid & ((hue <= 10) | (hue >= 170)),
            "YELLOW": valid & (hue >= 14) & (hue < 40),
            "GREEN": valid & (hue >= 40) & (hue <= 95),
        }
        counts = {key: int(mask.sum()) for key, mask in masks.items()}
        total = sum(counts.values())
        dominant = max(counts, key=counts.get)
        ratio = counts[dominant] / max(1, total)
        evidence = dict(glow_pixels=total, native_roi_size=[w, h], color_counts=counts)
        if total < 4 or ratio < .8 or color != dominant:
            return Attributes(reason="MIXED_OR_DIM_GLOW", evidence=evidence)
        return Attributes(color=color, color_confidence=ratio,
                          reason="COLOR_ONLY", evidence=evidence)


@dataclass(frozen=True)
class Candidate:
    bbox: tuple
    confidence: float
    frame_id: int
    timestamp_ns: int
    detector: str = "main"


def iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / area if area > 0 else 0.


def q4_box(bbox, *, space, width, height):
    """Explicit coordinate spaces; reject mosaic seam/other-camera boxes.

    q4_native uses original Q4 pixels. q4_display is the 2x front display.
    mosaic uses the complete four-camera image, with physical Q4 bottom right.
    """
    try:
        box = tuple(float(x) for x in bbox)
        if len(box) != 4 or not all(math.isfinite(x) for x in box):
            return None
    except (TypeError, ValueError):
        return None
    if space == "mosaic":
        if box[0] < width or box[1] < height or box[2] > 2*width or box[3] > 2*height:
            return None
        box = (box[0]-width, box[1]-height, box[2]-width, box[3]-height)
    elif space == "q4_display":
        box = tuple(x/2 for x in box)
    elif space != "q4_native":
        return None
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        return None
    return box


def candidates_from_rows(rows, *, frame_id, timestamp_ns, space, width, height, detector="main"):
    result = []
    for row in rows:
        if not isinstance(row, dict) or row.get("class_name") != "traffic light":
            continue
        box = q4_box(row.get("bbox", row.get("bbox_xyxy")), space=space, width=width, height=height)
        try:
            confidence = float(row.get("confidence", 0))
        except (TypeError, ValueError):
            continue
        if box is not None and math.isfinite(confidence) and .25 <= confidence <= 1:
            result.append(Candidate(box, confidence, int(frame_id), int(timestamp_ns), detector))
    return result


@dataclass
class _Track:
    track_id: int
    bbox: tuple
    timestamp_ns: int
    frame_id: int
    confidence: float
    attributes: Attributes
    detectors: tuple
    pending: tuple = ("UNKNOWN", "UNKNOWN")
    hits: int = 0
    color: str = "UNKNOWN"
    shape: str = "UNKNOWN"


def unknown_directions(reason):
    return {direction: dict(color="UNKNOWN", confidence=0., reason=reason,
                            signal_ids=[], valid=False) for direction in DIRECTIONS}


class MultiSignalTracker:
    """Small bounded IoU tracker, conservative on ambiguous identity crossings.

    Only exact-frame detections may be classified. Missed detections remain
    visible for debugging until TTL, but immediately lose direction validity.
    """

    def __init__(self, backend=None, ttl_ms=1200, confirm_updates=2, max_signals=8):
        self.backend = backend or HSVBackend()
        self.ttl_ns = int(ttl_ms * 1e6)
        self.confirm_updates = max(1, confirm_updates)
        self.max_signals = max_signals
        self.tracks = []
        self.next_id = 1
        self.context = None
        self.last_frame_id = -1
        self.last_timestamp_ns = -1
        self.rejections = []

    def update(self, frame, candidates, *, frame_id, timestamp_ns, context):
        context = tuple(context)
        if context != self.context or timestamp_ns < self.last_timestamp_ns or frame_id < self.last_frame_id:
            self.tracks = []
            self.last_frame_id = -1
            self.last_timestamp_ns = -1
            self.context = context
        if frame_id <= self.last_frame_id or timestamp_ns <= self.last_timestamp_ns:
            return False
        self.last_frame_id, self.last_timestamp_ns = frame_id, timestamp_ns
        self.tracks = [t for t in self.tracks if 0 <= timestamp_ns-t.timestamp_ns < self.ttl_ns]
        self.rejections = []
        unique = []
        for candidate in sorted(candidates, key=lambda c: c.confidence, reverse=True):
            if candidate.frame_id != frame_id or candidate.timestamp_ns != timestamp_ns:
                self.rejections.append("DETECTION_FRAME_MISMATCH")
                continue
            box = q4_box(candidate.bbox, space="q4_native", width=frame.shape[1], height=frame.shape[0])
            if box is None or not math.isfinite(candidate.confidence) or not .25 <= candidate.confidence <= 1:
                self.rejections.append("INVALID_CANDIDATE")
                continue
            duplicate = next((entry for entry in unique if iou(entry[0].bbox, box) >= .5), None)
            if duplicate is not None:
                duplicate[1].add(candidate.detector)
            elif len(unique) < self.max_signals:
                unique.append([candidate, {candidate.detector}])
            else:
                self.rejections.append("CANDIDATE_LIMIT")
        used = set()
        old_tracks = list(self.tracks)
        for candidate, detectors in unique:
            matches = [(iou(candidate.bbox, t.bbox), t) for t in old_tracks
                       if t.track_id not in used and iou(candidate.bbox, t.bbox) >= .3]
            matches.sort(key=lambda pair: pair[0], reverse=True)
            ambiguous = len(matches) > 1 and matches[0][0]-matches[1][0] < .15
            if ambiguous:
                self.rejections.append("AMBIGUOUS_TRACK_IDENTITY")
                for _, old in matches:
                    old.color = old.shape = "UNKNOWN"
                    old.hits = 0
                continue
            x1, y1 = (int(math.floor(x)) for x in candidate.bbox[:2])
            x2, y2 = (int(math.ceil(x)) for x in candidate.bbox[2:])
            attributes = self.backend.classify(frame[y1:y2, x1:x2])
            if attributes.color not in COLORS or attributes.shape not in SHAPES:
                attributes = Attributes(reason="INVALID_BACKEND_RESULT")
            if matches:
                track = matches[0][1]
            else:
                if len(self.tracks) >= self.max_signals:
                    self.rejections.append("TRACK_LIMIT")
                    continue
                track = _Track(self.next_id, candidate.bbox, timestamp_ns, frame_id,
                               candidate.confidence, attributes, tuple(detectors))
                self.next_id += 1
                self.tracks.append(track)
            used.add(track.track_id)
            track.bbox, track.timestamp_ns, track.frame_id = candidate.bbox, timestamp_ns, frame_id
            track.confidence, track.attributes, track.detectors = candidate.confidence, attributes, tuple(sorted(detectors))
            observed = (attributes.color, attributes.shape)
            track.hits = track.hits+1 if track.pending == observed else 1
            track.pending = observed
            # A changed/unknown observation invalidates immediately; confirmation
            # requires separate source frames, never repeated reads of a cache.
            track.color = attributes.color if track.hits >= self.confirm_updates else "UNKNOWN"
            track.shape = attributes.shape if track.hits >= self.confirm_updates else "UNKNOWN"
        for track in self.tracks:
            if track.track_id not in used:
                track.hits = 0
                track.color = track.shape = "UNKNOWN"
        return True

    def snapshot(self, now_ns, *, context=None, available=True, reason=""):
        matching = context is None or tuple(context) == self.context
        signals = []
        for track in self.tracks if matching else ():
            age = now_ns-track.timestamp_ns
            fresh = available and 0 <= age < self.ttl_ns and track.frame_id == self.last_frame_id
            row = dict(track_id=track.track_id, source_id="Q4", bbox=list(track.bbox),
                       bbox_space="q4_native", frame_id=track.frame_id,
                       timestamp_ns=track.timestamp_ns, expires_at_ns=track.timestamp_ns+self.ttl_ns,
                       age_ms=age/1e6, valid=fresh, color=track.color if fresh else "UNKNOWN",
                       shape=track.shape if fresh else "UNKNOWN", confidence=track.confidence,
                       color_confidence=track.attributes.color_confidence,
                       shape_confidence=track.attributes.shape_confidence,
                       detectors=list(track.detectors), backend=track.attributes.backend,
                       evidence=track.attributes.evidence, association="UNASSOCIATED",
                       ego_relevant="UNKNOWN", direction=None,
                       reason=(reason or "SOURCE_UNAVAILABLE") if not available else
                              "EXPIRED" if age >= self.ttl_ns else
                              "MISSED_DETECTION" if not fresh else
                              "STABILIZING" if track.hits < self.confirm_updates else track.attributes.reason)
            signals.append(row)
        why = reason or ("NO_ASSOCIATION_EVIDENCE" if matching else "CONTEXT_CHANGED")
        return dict(schema_version=1, source_id="Q4", context=list(self.context or ()) if matching else list(context or ()),
                    frame_id=self.last_frame_id if matching else -1, timestamp_ns=self.last_timestamp_ns if matching else 0,
                    published_at_ns=now_ns, ttl_ms=self.ttl_ns/1e6,
                    signals=signals, directions=unknown_directions(why), rejections=list(self.rejections),
                    debug_only=True)
