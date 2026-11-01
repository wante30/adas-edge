"""Board adapter from RKNN C API rows/masks to the shared HazardFusion."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque
from threading import RLock
import os
import time

import numpy as np

from competition_ai.hazard_fusion import (
    COLLISION as FRONT_COLLISION_RISK,
    CROSSWALK_RISK,
    CUT_IN as VEHICLE_CUT_IN_CANDIDATE,
    NEAR_LEAD as NEAR_LEAD_VEHICLE,
    ONCOMING_CONFLICT as ONCOMING_PATH_CONFLICT,
    PEDESTRIAN_CROSSING as PEDESTRIAN_CROSSING_PATH,
    PEDESTRIAN_ENTRY as PEDESTRIAN_IN_PATH,
    TWO_WHEELER_CONFLICT as TWO_WHEELER_PATH_CONFLICT,
    HazardFusion,
)
from competition_ai.hazard_events import paired_red_lamps
from competition_ai.image_motion import BackgroundMotion
from competition_ai.road_scene import RoadSceneStabilizer
from competition_ai.scene_observation import Detection, SceneObservation
from .byte_tracker_lite import ByteTracker

PEDESTRIAN_KINDS = {
    PEDESTRIAN_IN_PATH, PEDESTRIAN_CROSSING_PATH, CROSSWALK_RISK,
}
OBSTACLE_KINDS = {
    NEAR_LEAD_VEHICLE, FRONT_COLLISION_RISK, VEHICLE_CUT_IN_CANDIDATE,
    ONCOMING_PATH_CONFLICT, TWO_WHEELER_PATH_CONFLICT,
}
SEVERE = {"warning", "critical"}


@dataclass(frozen=True)
class BoardHazardObservation:
    timestamp_ns: int | None = None
    events: tuple = ()
    tracks: tuple = ()
    reason: str = "warming_up"
    source_id: str = "rk3566-capi"
    generation: int = 1
    frame_id: int = -1
    media_s: float = 0.0
    road: dict = field(default_factory=dict)

    @property
    def pedestrian_hazard(self):
        return any(item.get("kind") in PEDESTRIAN_KINDS
                   and item.get("severity") in SEVERE
                   for item in self.events)

    @property
    def obstacle_hazard(self):
        return any(item.get("kind") in OBSTACLE_KINDS
                   and item.get("severity") in SEVERE
                   for item in self.events)


class SceneHazardEngine:
    """Stateful board hazard engine with the same public update/current API."""

    def __init__(self, *, ttl_s=1.5, brake_lights=True, source_id="rk3566-capi"):
        self.ttl_s = float(ttl_s)
        self.brake_lights = bool(brake_lights)
        self.source_id = source_id
        self._lock = RLock()
        self.fusion = HazardFusion()
        self.tracker = ByteTracker(high_thresh=0.25, low_thresh=0.10, new_track_thresh=0.30, track_buffer=30)
        self.road = RoadSceneStabilizer()
        self.motion = BackgroundMotion()
        self.timings = []
        self.last_breakdown = {}
        self.last_observation = BoardHazardObservation()
        self.generation = 1
        self.last_frame_id = None
        self.last_media_s = None
        self.road_interval_s = 0.50
        self.road_enabled = os.environ.get("ADAS_DISABLE_ROAD_SCENE", "0") != "1"
        self.breakdowns = deque(maxlen=3600)
        self.last_road_scene = None
        self.last_road_media_s = -1.0

    def reset(self):
        with self._lock:
            self.fusion = HazardFusion()
            self.tracker = ByteTracker(high_thresh=0.25, low_thresh=0.10, new_track_thresh=0.30, track_buffer=30)
            self.road = RoadSceneStabilizer()
            self.motion = BackgroundMotion()
            self.timings.clear()
            self.last_observation = BoardHazardObservation(
                reason="reset", source_id=self.source_id,
                generation=self.generation)
            self.last_frame_id = None
            self.last_media_s = None
            self.last_road_scene = None
            self.last_road_media_s = -1.0
            self.breakdowns.clear()

    def _tracked_rows(self, detections, width, height):
        boxes = []
        for item in detections:
            box = item.get("bbox")
            if box is None or len(box) != 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append({
                "className": str(item.get("class_name", "")),
                "confidence": float(item.get("confidence", 0.0)),
                "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
            })
        return self.tracker.update(boxes)

    def update(self, frame, detections, frame_id, timestamp_ns, *,
               masks=None, letterbox=None, source_width=None,
               source_height=None, traffic_state="UNKNOWN",
               traffic_lights=()):
        started = time.perf_counter()
        height, width = frame.shape[:2]
        width = int(source_width or width)
        height = int(source_height or height)
        media_s = float(timestamp_ns) / 1e9
        if self.last_frame_id is not None and int(frame_id) < self.last_frame_id:
            self.reset()
        if self.last_media_s is not None and media_s < self.last_media_s - 0.05:
            self.reset()
        self.last_frame_id = int(frame_id)
        self.last_media_s = media_s

        t_track = time.perf_counter()
        rows = self._tracked_rows(detections, width, height)
        t_road0 = time.perf_counter()
        row_dicts = []
        for box in rows:
            row_dicts.append({
                "class_name": box["className"],
                "confidence": box["confidence"],
                "bbox": [box["x"], box["y"],
                         box["x"] + box["width"], box["y"] + box["height"]],
                "trackId": int(box["trackId"]),
            })
        scale, pad_x, pad_y = (letterbox if letterbox is not None
                               else (1.0, 0, 0))
        meta = None
        if masks is not None:
            meta = {"mask_width": int(masks.shape[2]),
                    "mask_height": int(masks.shape[1])}
        observation = SceneObservation(
            source_id=self.source_id, generation=self.generation,
            frame_id=int(frame_id), media_s=media_s,
            detections=[Detection(row["class_name"], row["confidence"],
                                  tuple(row["bbox"]), row["trackId"])
                        for row in row_dicts],
            lane_cat=(masks[0] if masks is not None else None),
            lane_style=(masks[1] if masks is not None else None),
            drivable=(masks[2] if masks is not None else None),
            letterbox={"scale": float(scale), "pad_x": float(pad_x),
                       "pad_y": float(pad_y),
                       "mask_width": float(meta["mask_width"]) if meta else 0.0,
                       "mask_height": float(meta["mask_height"]) if meta else 0.0},
            source_width=width, source_height=height,
            traffic_light_state=traffic_state)
        if self.road_enabled and (self.last_road_scene is None
                or media_s - self.last_road_media_s >= self.road_interval_s):
            road_scene = self.road.update(observation)
            self.last_road_scene = road_scene
            self.last_road_media_s = media_s
        elif self.road_enabled:
            road_scene = self.last_road_scene
        else:
            road_scene = None
        t_motion0 = time.perf_counter()
        engine_items = [{"class_name": row["class_name"],
                         "confidence": row["confidence"],
                         "bbox": row["bbox"]} for row in row_dicts]
        motion = self.motion.update(frame, engine_items, int(timestamp_ns))
        t_brake0 = time.perf_counter()
        brake_scores = {}
        if self.brake_lights:
            for row in row_dicts:
                if row["class_name"] not in ("car", "truck", "bus"):
                    continue
                box = row["bbox"]
                if box[3] < height * 0.50 or box[3] - box[1] < height * 0.08:
                    continue
                brake_scores[row["trackId"]] = paired_red_lamps(frame, box)
        t_fusion0 = time.perf_counter()
        fusion = self.fusion.update(
            observation, frame_size=(width, height), motion=motion,
            road_scene=road_scene, brake_scores=brake_scores,
            traffic_state=traffic_state, traffic_lights=traffic_lights)
        events = []
        for event in fusion.events:
            if event.state == "CLEARED" or not event.primary:
                continue
            events.append({
                "kind": event.kind,
                "severity": event.severity,
                "assessment": event.assessment,
                "state": event.state,
                "track_id": event.track_id,
                "bbox": None if event.bbox is None else list(event.bbox),
                "target_class": event.target_class,
                "source": event.evidence.get("rule", "IMAGE_PERCEPTION"),
                "evidence": event.evidence,
                "frame_id": event.frame_id,
                "media_s": event.media_s,
                "generation": event.generation,
            })
        self.last_observation = BoardHazardObservation(
            timestamp_ns=int(timestamp_ns), events=tuple(events),
            tracks=tuple(fusion.tracks), reason=fusion.reason,
            source_id=self.source_id, generation=self.generation,
            frame_id=int(frame_id), media_s=media_s,
            road=(road_scene.as_dict() if road_scene is not None else {}))
        t_end = time.perf_counter()
        self.last_breakdown = {
            "track_ms": (t_road0 - t_track) * 1000.0,
            "road_ms": (t_motion0 - t_road0) * 1000.0,
            "motion_ms": (t_brake0 - t_motion0) * 1000.0,
            "brake_ms": (t_fusion0 - t_brake0) * 1000.0,
            "fusion_ms": (t_end - t_fusion0) * 1000.0,
        }
        self.breakdowns.append(dict(self.last_breakdown))
        self.timings.append((t_end - started) * 1000.0)
        if len(self.timings) > 3600:
            del self.timings[:-3600]
        return self.last_observation

    def current(self, now_ns):
        observed = self.last_observation
        if observed.timestamp_ns is None:
            return observed
        if now_ns - observed.timestamp_ns > self.ttl_s * 1e9:
            return BoardHazardObservation(
                reason="observation_expired", source_id=self.source_id,
                generation=self.generation)
        return observed
