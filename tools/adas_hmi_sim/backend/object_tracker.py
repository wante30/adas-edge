"""Small class/IoU tracker with EMA world-position filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schemas import BBox, SceneObject, Velocity, WorldPoint


def iou(a: BBox, b: BBox) -> float:
    width = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    height = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    intersection = width * height
    union = a.width * a.height + b.width * b.height - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass
class Track:
    id: int
    class_name: str
    confidence: float
    bbox: BBox
    world: WorldPoint
    velocity: Velocity
    timestamp_ms: int
    age: int = 1
    missed: int = 0


class ObjectTracker:
    """Expiry counts detector updates, not the 30 Hz WebSocket clock."""

    def __init__(self, iou_threshold: float = 0.18, alpha: float = 0.35, max_missed: int = 4):
        if not 0 < alpha <= 1:
            raise ValueError("EMA alpha must be in (0, 1]")
        self.iou_threshold = iou_threshold
        self.alpha = alpha
        self.max_missed = max_missed
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    @property
    def count(self) -> int:
        return len(self._tracks)

    def update(self, measurements: list[dict[str, Any]], timestamp_ms: int) -> list[SceneObject]:
        candidates = []
        for index, item in enumerate(measurements):
            for track_id, track in self._tracks.items():
                if item["class_name"] == track.class_name:
                    score = iou(item["bbox"], track.bbox)
                    if score >= self.iou_threshold:
                        candidates.append((score, index, track_id))
        matched_inputs: set[int] = set()
        matched_tracks: set[int] = set()
        for _, index, track_id in sorted(candidates, reverse=True):
            if index in matched_inputs or track_id in matched_tracks:
                continue
            item, track = measurements[index], self._tracks[track_id]
            new = item["world"]
            filtered = WorldPoint(
                self.alpha * new.x + (1 - self.alpha) * track.world.x,
                self.alpha * new.z + (1 - self.alpha) * track.world.z,
            )
            dt = max(0.001, (timestamp_ms - track.timestamp_ms) / 1000)
            track.velocity = Velocity((filtered.x - track.world.x) / dt, (filtered.z - track.world.z) / dt)
            track.world, track.bbox = filtered, item["bbox"]
            track.confidence = item["confidence"]
            track.timestamp_ms, track.missed = timestamp_ms, 0
            track.age += 1
            matched_inputs.add(index)
            matched_tracks.add(track_id)
        for track_id in list(self._tracks):
            if track_id not in matched_tracks:
                track = self._tracks[track_id]
                track.missed += 1
                if track.missed > self.max_missed:
                    del self._tracks[track_id]
        for index, item in enumerate(measurements):
            if index not in matched_inputs:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = Track(track_id, item["class_name"], item["confidence"], item["bbox"], item["world"], Velocity(), timestamp_ms)
        return self.snapshot()

    def snapshot(self) -> list[SceneObject]:
        result = []
        for track in self._tracks.values():
            fade_in = min(1.0, 0.5 + track.age * 0.25)
            fade_out = max(0.0, 1.0 - track.missed / (self.max_missed + 1))
            result.append(SceneObject(track.id, track.class_name, track.confidence, track.bbox, track.world, track.velocity, fade_in * fade_out, track.age))
        return result
