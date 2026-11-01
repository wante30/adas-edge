"""Platform-neutral hazard fusion over SceneObservation.

The engine consumes the shared observation contract (source identity, frame
identity, media clock, detections with tracker IDs). It never imports Qt,
ONNX Runtime, RKNN, FFmpeg or BAR1, so the Windows producer and the RK3566
board can run the same rules.

First-stage rules:

* ``PEDESTRIAN_IN_PATH`` -- an established person track moves from outside
  the ego corridor into it, confirmed over multiple source frames.
* ``NEAR_LEAD_VEHICLE`` -- the lead in-corridor vehicle is close in image
  space. This is proximity only, with no metric distance.
* ``FRONT_COLLISION_RISK`` -- the lead in-corridor vehicle grows in image
  scale faster than the background, confirmed over multiple frames.

Image expansion is a collision cue, not a calibrated distance or speed.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import math
from typing import Callable, Iterable, Sequence

import numpy as np

from .image_tracks import ImageTracker

SCHEMA = "hazard-event/1"
VEHICLES = ("car", "truck", "bus")
PERSONS = ("person",)
TWO_WHEELERS = ("motorcycle", "bicycle")
RIDERS = ("rider",)
CLASS_ALIASES = {"motor": "motorcycle", "bike": "bicycle"}

DEFAULT_EGO_CORRIDOR = (
    (0.485, 0.480),
    (0.515, 0.480),
    (0.630, 0.880),
    (0.370, 0.880),
)

TRIGGERED = "TRIGGERED"
ACTIVE = "ACTIVE"
CLEARED = "CLEARED"

NEAR_LEAD = "NEAR_LEAD_VEHICLE"
COLLISION = "FRONT_COLLISION_RISK"
PEDESTRIAN_ENTRY = "PEDESTRIAN_IN_PATH"
PEDESTRIAN_CROSSING = "PEDESTRIAN_CROSSING_PATH"
CUT_IN = "VEHICLE_CUT_IN_CANDIDATE"
ONCOMING_APPROACH = "ONCOMING_VEHICLE_APPROACH"
ONCOMING_CONFLICT = "ONCOMING_PATH_CONFLICT"
TWO_WHEELER_CONFLICT = "TWO_WHEELER_PATH_CONFLICT"
CROSSWALK_RISK = "CROSSWALK_PEDESTRIAN_RISK"
LEAD_BRAKING = "LEAD_BRAKING_RISK"
RED_OBSERVED = "TRAFFIC_LIGHT_RED_OBSERVED"
RED_CONFLICT = "EGO_RED_LIGHT_CONFLICT_CANDIDATE"

OBSERVATION = "observation"
CANDIDATE = "candidate"
WARNING = "warning"
CRITICAL = "critical"
ASSESSMENT_PRIORITY = {
    OBSERVATION: 100, CANDIDATE: 200, WARNING: 300, CRITICAL: 400,
}
ASSESSMENT_SEVERITY = {
    OBSERVATION: "info", CANDIDATE: "info",
    WARNING: "warning", CRITICAL: "critical",
}
TWO_WHEELERS_ALL = ("motorcycle", "bicycle", "motor", "bike")
RIDER_CLASSES = ("rider",)


def _finite_box(box: Sequence[float], width: float, height: float):
    if box is None or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return (x1, y1, x2, y2)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    intersection = (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                    * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - intersection)
    return intersection / max(union, 1e-9)


def _inside_polygon(polygon: np.ndarray, x: float, y: float) -> bool:
    inside = False
    count = len(polygon)
    j = count - 1
    for i in range(count):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            crossing = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < crossing:
                inside = not inside
        j = i
    return inside


@dataclass(frozen=True)
class EgoCorridor:
    """Image-space driving corridor, normalized to the source frame."""

    polygon: tuple[tuple[float, float], ...] = DEFAULT_EGO_CORRIDOR
    valid: bool = True
    source: str = "STATIC_IMAGE_CORRIDOR"

    def __post_init__(self) -> None:
        points = np.asarray(self.polygon, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2:
            raise ValueError("corridor polygon must contain at least three 2D points")
        if not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
            raise ValueError("corridor points must be finite and normalized")
        centered = points - points.mean(axis=0)
        rolled = np.roll(points, -1, axis=0) - points.mean(axis=0)
        area2 = float((centered[:, 0] * rolled[:, 1]
                       - centered[:, 1] * rolled[:, 0]).sum())
        if abs(area2) < 1e-6:
            raise ValueError("corridor polygon is degenerate")

    @property
    def bottom(self) -> float:
        return float(max(point[1] for point in self.polygon))

    def contains(self, x: float, y: float) -> bool:
        if not self.valid:
            return False
        return _inside_polygon(np.asarray(self.polygon, dtype=np.float64),
                               float(x), min(float(y), self.bottom - 1e-4))

    def x_bounds(self, y: float) -> tuple[float, float] | None:
        if not self.valid:
            return None
        y = min(float(y), self.bottom - 1e-4)
        crossings = []
        polygon = self.polygon
        count = len(polygon)
        j = count - 1
        for i in range(count):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if (yi > y) != (yj > y):
                crossings.append((xj - xi) * (y - yi) / (yj - yi) + xi)
            j = i
        if len(crossings) < 2:
            return None
        return min(crossings), max(crossings)

    def contains_inner(self, x: float, y: float, fraction: float = 0.75) -> bool:
        """Require the point to sit inside the central band of the corridor."""
        if not self.valid:
            return False
        y = min(float(y), self.bottom - 1e-4)
        crossings = []
        polygon = self.polygon
        count = len(polygon)
        j = count - 1
        for i in range(count):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if (yi > y) != (yj > y):
                crossings.append((xj - xi) * (y - yi) / (yj - yi) + xi)
            j = i
        if len(crossings) < 2:
            return False
        left, right = min(crossings), max(crossings)
        half = max(0.0, (right - left) * 0.5 * min(1.0, max(0.0, fraction)))
        return abs(float(x) - (left + right) * 0.5) <= half

    def as_list(self) -> list[list[float]]:
        return [[round(float(x), 4), round(float(y), 4)] for x, y in self.polygon]


@dataclass(frozen=True)
class RoadContext:
    corridor: EgoCorridor
    lane_semantics: bool = False
    topology_status: str = "unknown"
    ego_lane_index: int | None = None
    lanes: tuple = ()
    boundaries: tuple = ()
    crosswalks: tuple = ()
    traffic_lights: tuple = ()
    traffic_state: str = "UNKNOWN"
    source_frame_ok: bool = False

    @property
    def has_observed_crosswalk(self) -> bool:
        return bool(self.crosswalks)


@dataclass(frozen=True)
class FusionConfig:
    min_vehicle_confidence: float = 0.35
    min_person_confidence: float = 0.45
    person_min_height: float = 0.12
    person_min_foot: float = 0.55
    person_min_area: float = 0.0035
    entry_lookback_s: float = 0.8
    entry_min_displacement: float = 0.012
    entry_confirmations: int = 2
    collision_confirmations: int = 2
    near_confirmations: int = 2
    near_vehicle_min_foot: float = 0.75
    near_vehicle_min_height: float = 0.22
    near_vehicle_early_foot: float = 0.77
    near_vehicle_early_height: float = 0.16
    near_vehicle_early_area: float = 0.07
    near_vehicle_min_area: float = 0.012
    near_approach_window_s: float = 0.70
    near_approach_min_samples: int = 4
    near_entry_min_growth: float = 1.04
    near_hold_min_growth: float = 1.01
    near_vehicle_exit_foot: float = 0.68
    near_vehicle_exit_height: float = 0.18
    near_vehicle_exit_area: float = 0.020
    near_vehicle_max_height: float = 0.75
    near_vehicle_max_area: float = 0.40
    near_vehicle_critical_foot: float = 0.86
    near_vehicle_critical_area: float = 0.08
    collision_window_s: float = 1.2
    collision_min_window_s: float = 0.6
    collision_min_samples: int = 6
    collision_min_growth: float = 1.20
    collision_min_log_rate: float = 0.35
    collision_min_fit: float = 0.80
    collision_min_foot: float = 0.50
    collision_min_size: float = 0.030
    collision_max_lateral_rate: float = 0.080
    collision_center_tolerance: float = 0.22
    lead_inner_corridor_fraction: float = 0.65
    hold_ttl_s: float = 1.50
    min_active_s: float = 1.00
    clear_confirmations: int = 5
    track_grace_s: float = 0.60
    duplicate_center_fraction: float = 0.06
    duplicate_foot_fraction: float = 0.08
    rearm_cooldown_s: float = 1.00
    gap_reset_s: float = 0.80
    max_history: int = 24
    occupancy_min_age_s: float = 0.20
    crossing_lateral_rate: float = 0.045
    crossing_lookahead_s: float = 0.80
    crossing_min_displacement: float = 0.025
    cut_in_min_age_s: float = 0.50
    cut_in_max_size: float = 0.20
    cut_in_lateral_rate: float = 0.040
    cut_in_lookahead_s: float = 0.80
    cut_in_confirmations: int = 3
    oncoming_window_s: float = 0.90
    oncoming_min_growth: float = 1.20
    oncoming_min_log_rate: float = 0.25
    oncoming_min_fit: float = 0.75
    oncoming_min_samples: int = 6
    oncoming_min_size: float = 0.020
    oncoming_conflict_frames: int = 3
    two_wheeler_min_size: float = 0.012
    crosswalk_min_width: float = 0.30
    crosswalk_min_overlap: float = 0.08
    brake_rise: float = 0.025
    brake_window_s: float = 1.20
    brake_min_score: float = 0.020
    oncoming_max_foot: float = 0.98


@dataclass(frozen=True)
class RuleHit:
    kind: str
    confidence: float
    evidence: dict
    assessment: str = WARNING
    priority: int = 300
    track_id: int | None = None


@dataclass(frozen=True)
class HazardEvent:
    kind: str
    severity: str
    state: str
    track_id: int | None
    source_id: str
    generation: int
    frame_id: int
    media_s: float
    first_frame_id: int
    first_media_s: float
    ttl_s: float
    valid_until_s: float
    confidence: float
    evidence: dict = field(default_factory=dict)
    bbox: tuple[float, float, float, float] | None = None
    target_class: str | None = None
    assessment: str = WARNING
    priority: int = 300
    primary: bool = True

    def as_dict(self) -> dict:
        payload = {
            "schema": SCHEMA,
            "kind": self.kind,
            "severity": self.severity,
            "state": self.state,
            "track_id": self.track_id,
            "source_id": self.source_id,
            "generation": int(self.generation),
            "frame_id": int(self.frame_id),
            "media_s": round(float(self.media_s), 4),
            "first_frame_id": int(self.first_frame_id),
            "first_media_s": round(float(self.first_media_s), 4),
            "ttl_s": round(float(self.ttl_s), 3),
            "valid_until_s": round(float(self.valid_until_s), 4),
            "confidence": round(float(self.confidence), 3),
            "assessment": self.assessment,
            "priority": int(self.priority),
            "primary": bool(self.primary),
            "evidence": _json_safe(self.evidence),
            "target_class": self.target_class,
        }
        if self.bbox is not None:
            payload["bbox"] = [round(float(value), 2) for value in self.bbox]
        return payload

    def as_hmi_detail(self, *, ttl_ms: float | None = None) -> dict:
        """Fields expected by the existing Windows Qt hazard card."""
        detail = {
            "event_type": self.kind,
            "source_id": self.source_id,
            "source_frame_id": int(self.frame_id),
            "timestamp": int(round(float(self.media_s) * 1e9)),
            "media_s": round(float(self.media_s), 4),
            "generation": int(self.generation),
            "severity": self.severity,
            "assessment": self.assessment,
            "priority": int(self.priority),
            "state": self.state,
            "valid": self.state != CLEARED,
            "ttl_ms": float(ttl_ms if ttl_ms is not None else self.ttl_s * 1000.0),
            "evidence": _evidence_label(self.evidence),
        }
        if self.track_id is not None:
            detail["track_id"] = int(self.track_id)
        if self.bbox is not None:
            detail["bbox"] = [round(float(value), 2) for value in self.bbox]
        if self.target_class:
            detail["target_class"] = self.target_class.upper()
        return detail


@dataclass(frozen=True)
class FusionObservation:
    source_id: str = ""
    generation: int = 0
    frame_id: int = -1
    media_s: float = 0.0
    events: tuple[HazardEvent, ...] = ()
    tracks: tuple[dict, ...] = ()
    reason: str = "warming_up"
    corridor: EgoCorridor = field(default_factory=EgoCorridor)

    def as_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "generation": int(self.generation),
            "frame_id": int(self.frame_id),
            "media_s": round(float(self.media_s), 4),
            "reason": self.reason,
            "events": [event.as_dict() for event in self.events],
            "tracks": _json_safe(self.tracks),
            "corridor": self.corridor.as_list(),
            "corridor_source": self.corridor.source,
        }


@dataclass
class _Sample:
    media_s: float
    cx: float
    foot: float
    width: float
    height: float

    @property
    def size(self) -> float:
        return math.sqrt(max(self.width * self.height, 1e-12))


@dataclass
class _TrackState:
    track_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    first_media_s: float
    last_media_s: float
    history: deque
    entered_at_s: float | None = None
    family: str = "vehicle"
    display_class: str = ""
    lamp_history: deque = field(default_factory=lambda: deque(maxlen=24))
    inside_since_s: float | None = None
    last_inside_s: float | None = None
    ever_lead: bool = False

    def add(self, sample: _Sample, *, maxlen: int, gap_reset_s: float) -> None:
        if self.history and sample.media_s - self.history[-1].media_s > gap_reset_s:
            self.history.clear()
        self.history.append(sample)
        while len(self.history) > maxlen:
            self.history.popleft()
        self.last_media_s = sample.media_s

    @property
    def latest(self) -> _Sample:
        return self.history[-1]

    @property
    def age_s(self) -> float:
        return self.last_media_s - self.first_media_s


@dataclass
class _EventState:
    kind: str
    track_id: int
    target_class: str
    confirmed: bool
    streak: int
    missed: int
    first_frame_id: int = -1
    first_media_s: float = 0.0
    last_hit_s: float = 0.0
    last_event: HazardEvent | None = None


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return round(float(value), 5)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def _evidence_label(evidence: dict) -> str:
    label = str(evidence.get("rule") or evidence.get("source") or "IMAGE_PERCEPTION")
    return label


def _fit_and_slope(times: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    centered = times - times.mean()
    denominator = float(centered @ centered)
    if denominator <= 1e-9:
        return 0.0, 0.0
    slope = float(centered @ (values - values.mean()) / denominator)
    residual = values - (values.mean() + slope * centered)
    variation = float(np.square(values - values.mean()).sum())
    if variation <= 1e-9:
        return 0.0, 0.0
    fit = 1.0 - float(np.square(residual).sum()) / variation
    return slope, max(0.0, min(1.0, fit))


def _aspect_ok(box: tuple[float, float, float, float]) -> bool:
    width, height = box[2] - box[0], box[3] - box[1]
    return 0.20 <= width / max(height, 1e-9) <= 3.6


class HazardFusion:
    """Stateful, source-timed hazard fusion over SceneObservation."""

    def __init__(self, corridor: EgoCorridor | None = None,
                 corridor_provider: Callable[[object], EgoCorridor] | None = None,
                 config: FusionConfig | None = None):
        self.default_corridor = corridor or EgoCorridor()
        self.corridor_provider = corridor_provider
        self.config = config or FusionConfig()
        self._fallback_tracker = ImageTracker()
        self._last_road_context = RoadContext(self.default_corridor)
        self.reset("initial")

    def reset(self, reason: str = "manual") -> None:
        self._tracks: dict[int, _TrackState] = {}
        self._states: dict[tuple[str, int], _EventState] = {}
        self._cooldowns: dict[tuple[str, int], float] = {}
        self._recent_cleared: deque = deque(maxlen=16)
        self._fallback_tracker = ImageTracker()
        self._last_key: tuple | None = None
        self._last_observation = FusionObservation(reason=reason,
                                                   corridor=self.default_corridor)
        self._last_source_id: str | None = None
        self._last_generation: int | None = None
        self._last_frame_id: int | None = None
        self._last_media_s: float | None = None

    @staticmethod
    def _ego_lane_polygon(lane) -> tuple:
        if lane is None:
            return ()
        left = [(float(x), float(y))
                for x, y in (getattr(lane, "left_points", ()) or ())
                if 0.40 <= float(y) <= 0.94]
        right = [(float(x), float(y))
                 for x, y in (getattr(lane, "right_points", ()) or ())
                 if 0.40 <= float(y) <= 0.94]
        if len(left) < 2 or len(right) < 2:
            return ()
        left.sort(key=lambda point: point[1])
        right.sort(key=lambda point: point[1], reverse=True)
        polygon = tuple(left + right)
        return polygon if len(polygon) >= 4 else ()

    def _road_context(self, scene, road_scene=None,
                      traffic_state: str = "UNKNOWN",
                      traffic_lights=()) -> RoadContext:
        if self.corridor_provider is not None:
            corridor = self.corridor_provider(scene)
            return RoadContext(
                corridor=corridor if corridor is not None
                else EgoCorridor(valid=False, source="UNAVAILABLE"),
                traffic_state=traffic_state,
                traffic_lights=tuple(traffic_lights or ()),
                source_frame_ok=True)
        if road_scene is None:
            return RoadContext(
                corridor=self.default_corridor,
                traffic_state=traffic_state,
                traffic_lights=tuple(traffic_lights or ()),
                source_frame_ok=True)
        same_frame = (
            int(getattr(road_scene, "generation", -1)) == int(scene.generation)
            and int(getattr(road_scene, "frame_id", -1)) == int(scene.frame_id)
            and abs(float(getattr(road_scene, "media_s", -1.0))
                    - float(scene.media_s)) <= 0.05)
        valid_geometry = (same_frame and bool(getattr(road_scene, "valid", False))
                          and not bool(getattr(road_scene, "stale", False)))
        road_age_s = (float(scene.media_s)
                      - float(getattr(road_scene, "media_s", -1.0)))
        tracked_geometry = (
            not same_frame
            and bool(getattr(road_scene, "valid", False))
            and 0.0 <= road_age_s <= 0.50)
        usable_geometry = valid_geometry or tracked_geometry
        topology = getattr(road_scene, "topology", None)
        lanes = tuple(getattr(topology, "lanes", ()) or ())
        ego_lane = next((lane for lane in lanes
                         if bool(getattr(lane, "is_ego", False))), None)
        lane_points = self._ego_lane_polygon(ego_lane)
        points = lane_points or tuple(getattr(road_scene, "corridor", ()) or ())
        corridor = self.default_corridor
        if usable_geometry and len(points) >= 4:
            try:
                candidate = EgoCorridor(
                    tuple((float(x), float(y)) for x, y in points),
                    valid=True,
                    source=("ROAD_SCENE" if valid_geometry
                            else "ROAD_SCENE_TRACKED"))
                near = candidate.x_bounds(0.80)
                far = candidate.x_bounds(0.60)
                if (near is not None and far is not None
                        and 0.08 <= near[1] - near[0] <= 0.35
                        and 0.05 <= far[1] - far[0] <= 0.32
                        and near[0] < near[1] and far[0] < far[1]):
                    corridor = candidate
            except ValueError:
                pass
        if corridor.source != "ROAD_SCENE":
            corridor = EgoCorridor(valid=True, source="STATIC_IMAGE_PRIOR")
        boundaries = tuple(getattr(topology, "boundaries", ()) or ())
        status = str(getattr(topology, "status", "unknown"))
        ego_index = getattr(topology, "ego_lane_index", None)
        crosswalks = tuple(getattr(road_scene, "crosswalks", ()) or ())
        observed = bool(getattr(road_scene, "observed", False)) or bool(
            getattr(corridor, "source", "") == "ROAD_SCENE")
        lane_semantics = bool(valid_geometry and observed and lanes
                              and ego_index is not None and status != "unknown")
        return RoadContext(
            corridor=corridor,
            lane_semantics=lane_semantics,
            topology_status=status,
            ego_lane_index=ego_index,
            lanes=lanes,
            boundaries=boundaries,
            crosswalks=crosswalks if valid_geometry else (),
            traffic_lights=tuple(traffic_lights or ()),
            traffic_state=traffic_state,
            source_frame_ok=same_frame)

    def _check_clock(self, scene) -> str | None:
        media_s = float(scene.media_s)
        frame_id = int(scene.frame_id)
        generation = int(scene.generation)
        source_id = str(scene.source_id)
        if self._last_generation is not None and (
                generation != self._last_generation
                or source_id != self._last_source_id):
            return "generation_changed"
        if self._last_media_s is not None:
            delta = media_s - self._last_media_s
            if delta < -0.05 or delta > self.config.gap_reset_s:
                return "seek_or_gap"
        if self._last_frame_id is not None and frame_id < self._last_frame_id:
            return "frame_regression"
        return None

    def update(self, scene, *, frame_size: tuple[int, int],
               motion=None, road_scene=None, brake_scores=None,
               traffic_state: str = "UNKNOWN",
               traffic_lights=()) -> FusionObservation:
        config = self.config
        width, height = (int(frame_size[0]), int(frame_size[1]))
        if width <= 0 or height <= 0:
            raise ValueError(f"invalid frame size: {frame_size}")
        if not math.isfinite(float(scene.media_s)):
            raise ValueError("media_s must be finite")
        key = (str(scene.source_id), int(scene.generation), int(scene.frame_id),
               round(float(scene.media_s), 6))
        if key == self._last_key:
            return self._last_observation

        reset_reason = self._check_clock(scene)
        cleared = self._drop_active_for_reset(reset_reason) if reset_reason else []
        if reset_reason:
            self._tracks.clear()
            self._states.clear()
            self._cooldowns.clear()
            self._recent_cleared.clear()
            self._fallback_tracker = ImageTracker()

        self._last_source_id = str(scene.source_id)
        self._last_generation = int(scene.generation)
        self._last_frame_id = int(scene.frame_id)
        self._last_media_s = float(scene.media_s)
        self._last_key = key

        context = self._road_context(scene, road_scene, traffic_state,
                                     traffic_lights)
        self._last_road_context = context
        corridor = context.corridor
        events: list[HazardEvent] = list(cleared)
        now_s = float(scene.media_s)
        if not scene.source_valid:
            events.extend(self._miss_all(scene, now_s,
                                        reason="source_invalid"))
            observation = self._observation(scene, corridor, events, (),
                                            "source_invalid")
            self._last_observation = observation
            return observation

        motion_ok = motion is None or bool(getattr(motion, "stable", False))
        samples = self._normalize(scene, width, height, brake_scores)
        self._update_tracks(samples, now_s)
        self._purge_tracks(now_s)

        hits: dict[tuple[str, int], RuleHit] = {}
        hits.update(self._evaluate_tracks(now_s, context, motion, motion_ok))
        hits.update(self._evaluate_road_semantics(now_s, scene, context,
                                                  motion_ok))
        active = self._apply_state_machine(scene, hits, now_s)
        active = self._arbitrate(active)
        events.extend(active)
        tracks = tuple(self._track_summary(corridor))
        reason = "tracking"
        if reset_reason:
            reason = reset_reason
        elif motion is not None and not motion_ok:
            reason = "camera_unstable"
        elif not samples:
            reason = "no_tracks"
        elif not corridor.valid:
            reason = "corridor_invalid"
        elif context.corridor.source == "STATIC_IMAGE_PRIOR":
            reason = "tracking_static_prior"
        observation = self._observation(scene, corridor, events, tracks, reason)
        self._last_observation = observation
        return observation

    def _drop_active_for_reset(self, reason: str | None) -> list[HazardEvent]:
        if not reason:
            return []
        cleared: list[HazardEvent] = []
        for state in self._states.values():
            if state.confirmed and state.last_event is not None:
                cleared.append(replace(state.last_event, state=CLEARED,
                                       valid_until_s=state.last_event.media_s,
                                       evidence={**state.last_event.evidence,
                                                 "cleared_reason": reason}))
        self._states.clear()
        return cleared

    def _normalize(self, scene, width: int, height: int, brake_scores=None):
        brake_scores = brake_scores or {}
        riders = []
        samples = []
        normalized = []
        for item in scene.detections:
            raw_class = str(item.class_name)
            class_name = CLASS_ALIASES.get(raw_class, raw_class)
            box = _finite_box(item.bbox, width, height)
            if box is None:
                continue
            confidence = float(item.confidence)
            normalized.append((class_name, confidence, box, item))
            if class_name in TWO_WHEELERS_ALL + RIDER_CLASSES and confidence >= 0.25:
                riders.append(box)
        for class_name, confidence, box, item in normalized:
            raw_class = str(item.class_name)
            display_class = raw_class
            if class_name in RIDER_CLASSES:
                # A rider box that overlaps a motor/bike is the same physical
                # target; only one canonical two-wheeler track may report.
                if any(_iou(box, rider) > 0.10 for rider in riders
                       if rider is not box):
                    continue
                family = "two_wheeler"
                class_name = "two_wheeler"
                minimum = 0.30
            elif class_name in TWO_WHEELERS_ALL:
                family = "two_wheeler"
                class_name = "two_wheeler"
                minimum = 0.30
            elif class_name in PERSONS:
                family = "person"
                minimum = self.config.min_person_confidence
            elif class_name in VEHICLES:
                family = "vehicle"
                minimum = self.config.min_vehicle_confidence
            else:
                continue
            if confidence < minimum:
                continue
            track_id = getattr(item, "track_id", None)
            if (family == "vehicle" and box[3] / height > 0.84
                    and (box[2] - box[0]) / max(box[3] - box[1], 1e-9) > 2.7):
                continue
            if family == "person" and any(_iou(box, rider) > 0.18
                                          for rider in riders):
                continue
            x1, y1, x2, y2 = box
            samples.append({
                "class_name": class_name,
                "display_class": display_class,
                "family": family,
                "confidence": confidence,
                "bbox": box,
                "track_id": track_id,
                "brake_score": (float(brake_scores.get(track_id, 0.0))
                                if isinstance(track_id, int) else 0.0),
                "sample": _Sample(
                    media_s=float(scene.media_s),
                    cx=((x1 + x2) / 2.0) / width,
                    foot=y2 / height,
                    width=(x2 - x1) / width,
                    height=(y2 - y1) / height),
            })
        with_ids = [item for item in samples                 if isinstance(item["track_id"], int)
                    and not isinstance(item["track_id"], bool)]
        if len(with_ids) != len(samples):
            self._assign_fallback_ids(samples, width, height)
        return samples

    def _assign_fallback_ids(self, samples, width, height) -> None:
        items = [{"class_name": item["class_name"], "bbox": list(item["bbox"])}
                 for item in samples]
        timestamp_ns = int(round(samples[0]["sample"].media_s * 1e9)) if samples else 0
        tracks = self._fallback_tracker.update(
            items, timestamp_ns, (height, width), _StaticMotion())
        for item in samples:
            if isinstance(item["track_id"], int) and not isinstance(
                    item["track_id"], bool):
                continue
            best, best_iou = None, 0.0
            for track in tracks:
                score = _iou(item["bbox"], track.box)
                if score > best_iou:
                    best, best_iou = track, score
            item["track_id"] = (
                1_000_000 + best.track_id
                if best is not None and best_iou > 0.1 else None)

    def _update_tracks(self, samples: Iterable[dict], now_s: float) -> None:
        config = self.config
        for item in samples:
            track_id = item["track_id"]
            if not isinstance(track_id, int) or isinstance(track_id, bool):
                continue
            state = self._tracks.get(track_id)
            if state is None or state.class_name != item["class_name"]:
                state = _TrackState(
                    track_id=track_id, class_name=item["class_name"],
                    family=item.get("family", "vehicle"),
                    display_class=item.get("display_class", item["class_name"]),
                    bbox=item["bbox"], first_media_s=now_s,
                    last_media_s=now_s, history=deque(maxlen=config.max_history))
                self._tracks[track_id] = state
            state.bbox = item["bbox"]
            state.family = item.get("family", state.family)
            state.display_class = item.get("display_class",
                                           state.display_class)
            state.add(item["sample"], maxlen=config.max_history,
                      gap_reset_s=config.gap_reset_s)
            state.last_media_s = now_s
            state.lamp_history.append((now_s, float(item.get("brake_score", 0.0))))

    def _purge_tracks(self, now_s: float) -> None:
        stale = [track_id for track_id, state in self._tracks.items()
                 if now_s - state.last_media_s > self.config.track_grace_s]
        for track_id in stale:
            self._tracks.pop(track_id, None)

    def _evaluate_tracks(self, now_s: float, context: RoadContext,
                         motion=None,
                         motion_ok: bool = True) -> dict[tuple[str, int], RuleHit]:
        corridor = context.corridor
        if not corridor.valid:
            return {}
        lead_id = self._lead_vehicle_id(corridor)
        background_rate = 0.0
        if motion is not None and bool(getattr(motion, "valid", False)):
            background_rate = max(0.0, float(getattr(motion, "scale_rate", 0.0)))
        lateral_ok = motion is None or (
            bool(getattr(motion, "valid", False))
            and bool(getattr(motion, "stable", False))
            and abs(float(getattr(motion, "rotation_dps", 0.0))) <= 2.0
            and abs(float(getattr(motion, "dx", 0.0))) <= 0.08)
        hits: dict[tuple[str, int], RuleHit] = {}
        for track_id, state in self._tracks.items():
            latest = state.latest
            inside = corridor.contains(latest.cx, latest.foot)
            if inside:
                if state.inside_since_s is None:
                    state.inside_since_s = now_s
                state.last_inside_s = now_s
            else:
                state.inside_since_s = None

            if state.family == "person":
                if not motion_ok:
                    continue
                occupancy = self._pedestrian_in_path(state, corridor, now_s)
                if occupancy is not None:
                    hits[(occupancy.kind, track_id)] = occupancy
                crossing = self._pedestrian_crossing(state, corridor, now_s)
                if crossing is not None:
                    hits[(crossing.kind, track_id)] = crossing
                if context.has_observed_crosswalk:
                    crosswalk = self._crosswalk_pedestrian_risk(
                        state, context, now_s)
                    if crosswalk is not None:
                        hits[(crosswalk.kind, track_id)] = crosswalk
                continue

            if state.family not in ("vehicle", "two_wheeler"):
                continue
            if track_id == lead_id:
                state.ever_lead = True
                collision = None
                if motion_ok:
                    collision = self._front_collision(
                        state, corridor, now_s, background_rate)
                if collision is not None:
                    hits[(collision.kind, track_id)] = collision
                elif not self._collision_context_active(track_id, now_s):
                    near_state = self._states.get((NEAR_LEAD, track_id))
                    near = self._near_lead_vehicle(
                        state, corridor, now_s,
                        confirmed=bool(near_state and near_state.confirmed))
                    if near is not None:
                        hits[(near.kind, track_id)] = near
                braking = self._lead_braking(state, now_s,
                                             collision is not None)
                if braking is not None:
                    hits[(braking.kind, track_id)] = braking

            if not inside and lateral_ok:
                cut_in = self._cut_in_candidate(state, corridor, now_s,
                                                motion, context)
                if cut_in is not None:
                    hits[(cut_in.kind, track_id)] = cut_in
                oncoming = self._oncoming_approach(state, corridor, now_s,
                                                   background_rate)
                if oncoming is not None:
                    hits[(oncoming.kind, track_id)] = oncoming
                conflict = self._oncoming_conflict(state, corridor, now_s,
                                                   background_rate)
                if conflict is not None:
                    hits[(conflict.kind, track_id)] = conflict
            if state.family == "two_wheeler" and lateral_ok:
                conflict = self._two_wheeler_conflict(
                    state, corridor, now_s, motion)
                if conflict is not None:
                    hits[(conflict.kind, track_id)] = conflict
        return hits

    def _collision_context_active(self, track_id: int, now_s: float) -> bool:
        state = self._states.get((COLLISION, track_id))
        if state is not None and state.confirmed:
            return True
        track = self._tracks.get(track_id)
        if track is None:
            return False
        for entry in self._recent_cleared:
            if entry["kind"] != COLLISION:
                continue
            if now_s - entry["media_s"] > self.config.rearm_cooldown_s:
                continue
            if (abs(entry["cx"] - track.latest.cx)
                    < self.config.duplicate_center_fraction
                    and abs(entry["foot"] - track.latest.foot)
                    < self.config.duplicate_foot_fraction):
                return True
        return False

    def _near_lead_vehicle(self, state: _TrackState, corridor: EgoCorridor,
                           now_s: float, confirmed: bool = False
                           ) -> RuleHit | None:
        config = self.config
        latest = state.latest
        area = latest.width * latest.height
        if not confirmed and (latest.height > config.near_vehicle_max_height
                              or area > config.near_vehicle_max_area):
            return None
        if confirmed:
            close = (latest.foot >= config.near_vehicle_exit_foot
                     and latest.height >= config.near_vehicle_exit_height
                     and area >= config.near_vehicle_exit_area)
        else:
            close = ((latest.foot >= config.near_vehicle_min_foot
                      and latest.height >= config.near_vehicle_min_height)
                     or (latest.foot >= config.near_vehicle_early_foot
                         and latest.height >= config.near_vehicle_early_height
                         and area >= config.near_vehicle_early_area))
        if not close or area < config.near_vehicle_min_area:
            return None
        if not corridor.contains_inner(latest.cx, latest.foot,
                                       config.lead_inner_corridor_fraction):
            return None
        if not _aspect_ok(state.bbox):
            return None
        if state.age_s < 0.20 or len(state.history) < 3:
            return None
        severity = ("critical"
                    if latest.foot >= config.near_vehicle_critical_foot
                    and area >= config.near_vehicle_critical_area
                    else "warning")
        evidence = {
            "rule": "IMAGE_PROXIMITY_LEAD_VEHICLE",
            "foot_fraction": latest.foot,
            "height_fraction": latest.height,
            "area_fraction": area,
            "track_age_s": state.age_s,
            "severity": severity,
            "distance_note": "IMAGE_PROXIMITY_CUE_NOT_METRIC_RANGE",
            "corridor_source": corridor.source,
        }
        confidence = min(0.92, 0.50 + 2.0 * min(area, 0.20))
        return RuleHit(NEAR_LEAD, confidence, evidence,
                       assessment=(CRITICAL if severity == "critical"
                                   else WARNING),
                       priority=310)

    def _lead_vehicle_id(self, corridor: EgoCorridor) -> int | None:
        lead, lead_foot = None, -1.0
        for track_id, state in self._tracks.items():
            if state.class_name not in VEHICLES:
                continue
            sample = state.latest
            if sample.foot < 0.40 or not corridor.contains_inner(
                    sample.cx, sample.foot,
                    self.config.lead_inner_corridor_fraction):
                continue
            if not _aspect_ok(state.bbox):
                continue
            if sample.foot > lead_foot:
                lead, lead_foot = track_id, sample.foot
        return lead

    def _pedestrian_in_path(self, state: _TrackState,
                            corridor: EgoCorridor,
                            now_s: float | None = None) -> RuleHit | None:
        config = self.config
        if len(state.history) < 3:
            return None
        latest = state.latest
        if latest.foot < config.person_min_foot:
            return None
        if latest.height < config.person_min_height:
            return None
        if latest.width * latest.height < config.person_min_area:
            return None
        if not corridor.contains(latest.cx, latest.foot):
            state.entered_at_s = None
            return None
        inside_age = (latest.media_s - state.inside_since_s
                      if state.inside_since_s is not None else state.age_s)
        previous_outside = None
        history = list(state.history)
        for sample in reversed(history[:-1]):
            if latest.media_s - sample.media_s > config.entry_lookback_s:
                break
            if not corridor.contains(sample.cx, sample.foot):
                previous_outside = sample
                break
        if state.entered_at_s is not None or (
                previous_outside is None
                and inside_age >= config.occupancy_min_age_s
                and len(state.history) >= 2 and latest.foot >= 0.60):
            basis = ("PATH_ENTRY" if state.entered_at_s is not None
                     else "PATH_OCCUPANCY")
            evidence = {
                "rule": "PEDESTRIAN_IN_PATH",
                "basis": basis,
                "sustained": True,
                "entry_media_s": state.entered_at_s,
                "time_in_corridor_s": inside_age,
                "to_cx": latest.cx,
                "to_foot": latest.foot,
                "track_age_s": state.age_s,
                "corridor_source": corridor.source,
            }
            confidence = min(0.90, 0.55 + 0.05 * min(len(state.history), 6))
            return RuleHit(PEDESTRIAN_ENTRY, confidence, evidence,
                           assessment=WARNING, priority=340)
        if previous_outside is None:
            return None
        displacement = abs(latest.cx - previous_outside.cx)
        if displacement < config.entry_min_displacement:
            return None
        if latest.foot < previous_outside.foot - 0.10:
            return None
        toward_center = abs(latest.cx - 0.5) <= abs(previous_outside.cx - 0.5)
        evidence = {
            "rule": "PEDESTRIAN_IN_PATH",
            "basis": "PATH_ENTRY",
            "from_cx": previous_outside.cx,
            "from_foot": previous_outside.foot,
            "to_cx": latest.cx,
            "to_foot": latest.foot,
            "lateral_entry_fraction": displacement,
            "toward_corridor_center": toward_center,
            "entry_latency_s": latest.media_s - previous_outside.media_s,
            "track_age_s": state.age_s,
            "corridor_source": corridor.source,
        }
        confidence = min(0.95, 0.60 + 0.10 * min(len(state.history), 4)
                         + 0.15 * min(1.0, displacement / 0.05))
        return RuleHit(PEDESTRIAN_ENTRY, confidence, evidence,
                       assessment=WARNING, priority=340)

    def _pedestrian_crossing(self, state: _TrackState, corridor: EgoCorridor,
                             now_s: float) -> RuleHit | None:
        config = self.config
        if len(state.history) < 4:
            return None
        latest = state.latest
        if (latest.foot < 0.50 or latest.height < 0.08
                or latest.width * latest.height < config.person_min_area):
            return None
        window = [sample for sample in state.history
                  if now_s - sample.media_s <= 0.80]
        if len(window) < 3 or window[-1].media_s - window[0].media_s < 0.30:
            return None
        times = np.asarray([sample.media_s - window[0].media_s
                            for sample in window])
        lateral = np.asarray([sample.cx for sample in window])
        rate, fit = _fit_and_slope(times, lateral)
        displacement = abs(latest.cx - window[0].cx)
        if abs(rate) < config.crossing_lateral_rate:
            return None
        if displacement < config.crossing_min_displacement:
            return None
        bounds = corridor.x_bounds(latest.foot)
        predicted = latest.cx + rate * config.crossing_lookahead_s
        intersects = corridor.contains(latest.cx, latest.foot)
        if bounds is not None:
            intersects = intersects or (bounds[0] - 0.02 <= predicted
                                        <= bounds[1] + 0.02)
        toward = abs(predicted - 0.5) < abs(latest.cx - 0.5)
        if not intersects or not toward:
            return None
        evidence = {
            "rule": "PEDESTRIAN_LATERAL_TRAJECTORY",
            "lateral_rate_fraction_per_s": rate,
            "lateral_fit_r2": fit,
            "displacement_fraction": displacement,
            "predicted_cx": predicted,
            "currently_in_path": corridor.contains(latest.cx, latest.foot),
            "to_foot": latest.foot,
            "track_age_s": state.age_s,
            "corridor_source": corridor.source,
        }
        confidence = min(0.92, 0.50 + 0.30 * fit
                         + 0.15 * min(1.0, abs(rate) / 0.15))
        return RuleHit(PEDESTRIAN_CROSSING, confidence, evidence,
                       assessment=WARNING, priority=440)

    def _predicted_cx(self, state: _TrackState, horizon_s: float) -> float:
        window = [sample for sample in state.history
                  if state.latest.media_s - sample.media_s <= 0.70]
        if len(window) < 3:
            return state.latest.cx
        times = np.asarray([sample.media_s - window[0].media_s
                            for sample in window])
        values = np.asarray([sample.cx for sample in window])
        rate, _ = _fit_and_slope(times, values)
        return state.latest.cx + rate * horizon_s

    def _lane_crossing_evidence(self, state: _TrackState,
                                context: RoadContext) -> str:
        if not context.lane_semantics or not context.boundaries:
            return "UNVERIFIED"
        latest = state.latest
        previous = None
        for sample in reversed(list(state.history)[:-1]):
            if latest.media_s - sample.media_s <= 0.45:
                previous = sample
                break
        if previous is None:
            return "UNVERIFIED"
        for boundary in context.boundaries:
            if str(getattr(boundary, "kind", "")) != "lane":
                continue
            points = getattr(boundary, "points", ()) or ()
            if len(points) < 2:
                continue
            xs = []
            for (x0, y0), (x1, y1) in zip(points, points[1:]):
                lo, hi = sorted((y0, y1))
                if lo <= latest.foot <= hi and abs(y1 - y0) > 1e-6:
                    xs.append(x0 + (x1 - x0) * (latest.foot - y0) / (y1 - y0))
            if not xs:
                continue
            bx = min(xs, key=lambda value: abs(value - latest.cx))
            if ((previous.cx < bx < latest.cx)
                    or (latest.cx < bx < previous.cx)):
                return "OBSERVED"
        return "UNVERIFIED"

    def _cut_in_candidate(self, state: _TrackState, corridor: EgoCorridor,
                          now_s: float, motion=None,
                          context: RoadContext | None = None
                          ) -> RuleHit | None:
        config = self.config
        if state.age_s < config.cut_in_min_age_s or len(state.history) < 5:
            return None
        latest = state.latest
        if latest.foot < 0.45 or latest.size < 0.020:
            return None
        if latest.size > config.cut_in_max_size:
            return None
        if state.ever_lead:
            return None
        if state.last_inside_s is not None and now_s - state.last_inside_s < 1.5:
            return None
        if not _aspect_ok(state.bbox):
            return None
        window = [sample for sample in state.history
                  if now_s - sample.media_s <= 0.60]
        if len(window) < 4:
            return None
        times = np.asarray([sample.media_s - window[0].media_s
                            for sample in window])
        lateral = np.asarray([sample.cx for sample in window])
        rate, fit = _fit_and_slope(times, lateral)
        background = (float(getattr(motion, "dx", 0.0))
                      if motion is not None
                      and bool(getattr(motion, "valid", False)) else 0.0)
        rate -= background
        if abs(rate) < config.cut_in_lateral_rate or fit < 0.45:
            return None
        predicted = latest.cx + rate * config.cut_in_lookahead_s
        bounds = corridor.x_bounds(latest.foot)
        if bounds is None:
            return None
        center = (bounds[0] + bounds[1]) * 0.5
        half = (bounds[1] - bounds[0]) * 0.5
        inner_lo, inner_hi = center - half * 0.65, center + half * 0.65
        if inner_lo <= latest.cx <= inner_hi:
            return None
        toward = abs(predicted - center) < abs(latest.cx - center)
        predicted_in = inner_lo - 0.02 <= predicted <= inner_hi + 0.02
        current_in = bounds[0] <= latest.cx <= bounds[1]
        if not toward or not (predicted_in or current_in):
            return None
        lane_crossing = (self._lane_crossing_evidence(state, context)
                         if context is not None else "UNVERIFIED")
        evidence = {
            "rule": "LATERAL_PATH_APPROACH",
            "lateral_rate_fraction_per_s": rate,
            "lateral_fit_r2": fit,
            "predicted_cx": predicted,
            "predicted_inner_corridor": predicted_in,
            "lane_crossing": lane_crossing,
            "lane_semantics": bool(context and context.lane_semantics),
            "lane_status": (context.topology_status if context else "unknown"),
            "target_family": state.family,
            "target_size": latest.size,
            "corridor_source": corridor.source,
        }
        confidence = min(0.90, 0.45 + 0.30 * fit
                         + 0.20 * min(1.0, abs(rate) / 0.15))
        return RuleHit(CUT_IN, confidence, evidence,
                       assessment=CANDIDATE, priority=330)

    def _oncoming_approach(self, state: _TrackState, corridor: EgoCorridor,
                           now_s: float, background_rate: float = 0.0
                           ) -> RuleHit | None:
        config = self.config
        if state.age_s < 0.80:
            return None
        latest = state.latest
        if latest.foot < 0.35 or latest.size < config.oncoming_min_size:
            return None
        if latest.foot > config.oncoming_max_foot:
            return None
        if corridor.contains_inner(latest.cx, latest.foot,
                                   config.lead_inner_corridor_fraction):
            return None
        bounds = corridor.x_bounds(latest.foot)
        if bounds is None:
            return None
        predicted = self._predicted_cx(state, 0.60)
        near_path = (bounds[0] - 0.10 <= latest.cx <= bounds[1] + 0.10
                     or bounds[0] - 0.08 <= predicted <= bounds[1] + 0.08)
        if not near_path:
            return None
        samples = [sample for sample in state.history
                   if now_s - sample.media_s <= config.oncoming_window_s]
        if len(samples) < config.oncoming_min_samples:
            return None
        times = np.asarray([sample.media_s - samples[0].media_s
                            for sample in samples])
        sizes = np.asarray([max(sample.size, 1e-9) for sample in samples])
        if times[-1] < 0.50:
            return None
        log_rate, fit = _fit_and_slope(times, np.log(sizes))
        relative_rate = log_rate - background_rate
        growth = float(sizes[-1] / sizes[0])
        relative_growth = math.exp(relative_rate * float(times[-1]))
        if (relative_growth < config.oncoming_min_growth
                or relative_rate < config.oncoming_min_log_rate + 0.10
                or fit < config.oncoming_min_fit):
            return None
        evidence = {
            "rule": "SIDE_FRONT_SCALE_APPROACH",
            "bbox_scale_growth": growth,
            "relative_log_expansion_per_s": relative_rate,
            "background_log_expansion_per_s": background_rate,
            "fit_r2": fit,
            "foot_fraction": latest.foot,
            "to_cx": latest.cx,
            "target_family": state.family,
            "direction_status": "UNCONFIRMED",
            "direction_note": "LANE_PAINT_DOES_NOT_ENCODE_TRAVEL_DIRECTION",
            "corridor_source": corridor.source,
        }
        confidence = min(0.90, 0.45 + 0.30 * fit
                         + 0.20 * min(1.0, log_rate / 1.0))
        return RuleHit(ONCOMING_APPROACH, confidence, evidence,
                       assessment=CANDIDATE, priority=320)

    def _oncoming_conflict(self, state: _TrackState, corridor: EgoCorridor,
                           now_s: float, background_rate: float = 0.0
                           ) -> RuleHit | None:
        approach = self._oncoming_approach(state, corridor, now_s,
                                           background_rate)
        if approach is None:
            return None
        latest = state.latest
        if latest.foot < 0.45:
            return None
        bounds = corridor.x_bounds(latest.foot)
        if bounds is None:
            return None
        center = (bounds[0] + bounds[1]) * 0.5
        half = (bounds[1] - bounds[0]) * 0.5
        predicted = self._predicted_cx(state, 0.80)
        current_in = bounds[0] <= latest.cx <= bounds[1]
        predicted_in = (center - half * 0.90
                        <= predicted <= center + half * 0.90)
        toward = abs(predicted - center) < abs(latest.cx - center)
        if not (current_in or (predicted_in and toward)):
            return None
        evidence = dict(approach.evidence)
        evidence.update({
            "rule": "ONCOMING_PATH_CONFLICT",
            "predicted_cx": predicted,
            "currently_in_path": current_in,
            "predicted_in_path": predicted_in,
            "assessment": WARNING,
        })
        return RuleHit(ONCOMING_CONFLICT, min(0.92, approach.confidence + 0.10),
                       evidence, assessment=WARNING, priority=420)

    def _two_wheeler_conflict(self, state: _TrackState,
                              corridor: EgoCorridor, now_s: float,
                              motion=None) -> RuleHit | None:
        config = self.config
        latest = state.latest
        if state.age_s < 0.35 or latest.foot < 0.45:
            return None
        if latest.size < config.two_wheeler_min_size:
            return None
        inside = corridor.contains(latest.cx, latest.foot)
        window = [sample for sample in state.history
                  if now_s - sample.media_s <= 0.70]
        if len(window) < 3:
            return None
        times = np.asarray([sample.media_s - window[0].media_s
                            for sample in window])
        lateral = np.asarray([sample.cx for sample in window])
        rate, fit = _fit_and_slope(times, lateral)
        background = (float(getattr(motion, "dx", 0.0))
                      if motion is not None
                      and bool(getattr(motion, "valid", False)) else 0.0)
        rate -= background
        bounds = corridor.x_bounds(latest.foot)
        predicted = latest.cx + rate * 0.80
        predicted_in = (bounds is not None
                        and bounds[0] - 0.02 <= predicted <= bounds[1] + 0.02)
        approaching = abs(rate) >= 0.030 and predicted_in
        if not (inside or approaching):
            return None
        evidence = {
            "rule": "TWO_WHEELER_PATH_CONFLICT",
            "target_family": "two_wheeler",
            "currently_in_path": inside,
            "predicted_cx": predicted,
            "lateral_rate_fraction_per_s": rate,
            "lateral_fit_r2": fit,
            "track_age_s": state.age_s,
            "corridor_source": corridor.source,
        }
        assessment = WARNING if inside else CANDIDATE
        return RuleHit(TWO_WHEELER_CONFLICT, 0.70 if inside else 0.55,
                       evidence, assessment=assessment, priority=430)

    def _crosswalk_pedestrian_risk(self, state: _TrackState,
                                   context: RoadContext,
                                   now_s: float) -> RuleHit | None:
        config = self.config
        latest = state.latest
        if latest.foot < 0.45 or latest.height < 0.08:
            return None
        for crosswalk in context.crosswalks:
            x0, y0, x1, y1 = (float(value) for value in crosswalk.bbox)
            if (x1 - x0) < config.crosswalk_min_width or crosswalk.center_y < 0.45:
                continue
            bounds = context.corridor.x_bounds(crosswalk.center_y)
            if bounds is None:
                continue
            overlap_left = max(x0, bounds[0])
            overlap_right = min(x1, bounds[1])
            overlap = overlap_right - overlap_left
            if overlap < config.crosswalk_min_overlap:
                continue
            near = (overlap_left - 0.04 <= latest.cx <= overlap_right + 0.04
                    and y0 - 0.06 <= latest.foot <= y1 + 0.06)
            if not near:
                continue
            evidence = {
                "rule": "CROSSWALK_PERSON_ASSOCIATION",
                "crosswalk_bbox": [round(value, 4) for value in crosswalk.bbox],
                "crosswalk_confidence": float(crosswalk.confidence),
                "crosswalk_center_y": float(crosswalk.center_y),
                "to_cx": latest.cx,
                "to_foot": latest.foot,
                "corridor_overlap": overlap,
                "road_status": context.topology_status,
                "corridor_source": context.corridor.source,
            }
            return RuleHit(CROSSWALK_RISK, min(0.90, 0.50
                                                + 0.30 * crosswalk.confidence),
                           evidence, assessment=WARNING, priority=350)
        return None

    def _lead_braking(self, state: _TrackState, now_s: float,
                      closing: bool) -> RuleHit | None:
        config = self.config
        history = [(ts, score) for ts, score in state.lamp_history
                   if now_s - ts <= config.brake_window_s]
        if len(history) < 3:
            return None
        latest_score = history[-1][1]
        if latest_score < config.brake_min_score:
            return None
        before = [score for ts, score in history
                  if 0.35 <= now_s - ts <= config.brake_window_s]
        if not before:
            return None
        rise = latest_score - min(before)
        if rise < config.brake_rise:
            return None
        evidence = {
            "rule": "PAIRED_RED_LAMP_RISE",
            "paired_red_lamp_fraction": latest_score,
            "lamp_rise": rise,
            "closing_evidence": bool(closing),
            "direction_status": "UNCONFIRMED",
        }
        assessment = WARNING if closing else CANDIDATE
        return RuleHit(LEAD_BRAKING, min(0.85, 0.45 + rise * 4.0),
                       evidence, assessment=assessment,
                       priority=360 if closing else 260)

    def _ego_lane_center(self, context: RoadContext) -> float | None:
        for lane in context.lanes:
            if bool(getattr(lane, "is_ego", False)):
                value = getattr(lane, "center_ref", None)
                if value is not None:
                    return float(value)
        for lane in context.lanes:
            if getattr(lane, "index", None) == context.ego_lane_index:
                value = getattr(lane, "center_ref", None)
                if value is not None:
                    return float(value)
        return None

    def _evaluate_road_semantics(self, now_s: float, scene,
                                 context: RoadContext,
                                 motion_ok: bool) -> dict[tuple[str, int], RuleHit]:
        hits: dict[tuple[str, int], RuleHit] = {}
        if context.traffic_state != "RED" or not context.traffic_lights:
            return hits
        hits[(RED_OBSERVED, -1)] = RuleHit(
            RED_OBSERVED, 0.60,
            {"rule": "RED_LAMP_OBSERVED",
             "lights": len(context.traffic_lights),
             "traffic_state": context.traffic_state,
             "direction_status": "UNCONFIRMED"},
            assessment=OBSERVATION, priority=120)
        if not context.lane_semantics:
            return hits
        ego_center = self._ego_lane_center(context)
        if ego_center is None:
            return hits
        for light in context.traffic_lights:
            box = (light.get("bbox") if isinstance(light, dict)
                   else getattr(light, "bbox", None))
            if not box:
                continue
            source_w = max(1.0, float(getattr(scene, "source_width", 1280)))
            source_h = max(1.0, float(getattr(scene, "source_height", 720)))
            x1, y1, x2, y2 = (float(value) for value in box)
            cx = (x1 + x2) * 0.5 / source_w
            y1 /= source_h
            y2 /= source_h
            height = y2 - y1
            if y1 < 0.12 or height < 0.008:
                continue
            if abs(cx - ego_center) > 0.10:
                continue
            hits[(RED_CONFLICT, -1)] = RuleHit(
                RED_CONFLICT, 0.45,
                {"rule": "RED_LIGHT_ASSOCIATED_TO_EGO_LANE",
                 "light_cx": cx,
                 "ego_lane_center": ego_center,
                 "light_bbox": [x1, y1, x2, y2],
                 "direction_status": "UNCONFIRMED",
                 "warning": "NO_MAP_OR_SIGNAL_GROUP_ASSOCIATION"},
                assessment=CANDIDATE, priority=250)
            break
        return hits

    def _arbitrate(self, events: list[HazardEvent]) -> list[HazardEvent]:
        if not events:
            return events
        groups: dict[int, list[HazardEvent]] = {}
        for event in events:
            key = -1 if event.track_id is None else int(event.track_id)
            groups.setdefault(key, []).append(event)
        resolved: list[HazardEvent] = []
        for key, group in groups.items():
            best = max(group, key=lambda item: (item.priority,
                                                ASSESSMENT_PRIORITY.get(
                                                    item.assessment, 0)))
            for event in group:
                resolved.append(replace(
                    event, primary=(event is best)))
        resolved = [
            replace(event, primary=False)
            if (event.kind in (ONCOMING_APPROACH, RED_OBSERVED)
                or (event.kind == CUT_IN
                    and float(event.evidence.get("target_size", 0.0)) > 0.20))
            else event
            for event in resolved]
        path_conflict = any(
            event.primary and event.track_id is not None
            and event.assessment in (WARNING, CRITICAL)
            for event in resolved)
        if path_conflict:
            resolved = [
                replace(event, primary=False)
                if event.kind in (RED_OBSERVED, RED_CONFLICT) else event
                for event in resolved]
        return resolved

    def _front_collision(self, state: _TrackState, corridor: EgoCorridor,
                         now_s: float, background_rate: float = 0.0) -> RuleHit | None:
        config = self.config
        if len(state.history) < config.collision_min_samples:
            return None
        samples = [sample for sample in state.history
                   if now_s - sample.media_s <= config.collision_window_s]
        if len(samples) < config.collision_min_samples:
            return None
        if samples[-1].media_s - samples[0].media_s < config.collision_min_window_s:
            return None
        times = np.asarray([sample.media_s - samples[0].media_s for sample in samples])
        sizes = np.asarray([max(sample.size, 1e-9) for sample in samples])
        lateral = np.asarray([sample.cx for sample in samples])
        feet = np.asarray([sample.foot for sample in samples])
        log_rate, fit = _fit_and_slope(times, np.log(sizes))
        relative_rate = log_rate - background_rate
        lateral_rate, _ = _fit_and_slope(times, lateral)
        foot_rate, _ = _fit_and_slope(times, feet)
        growth = float(sizes[-1] / sizes[0])
        latest = samples[-1]
        if latest.foot < config.collision_min_foot:
            return None
        if latest.size < config.collision_min_size:
            return None
        if not corridor.contains_inner(latest.cx, latest.foot,
                                       config.lead_inner_corridor_fraction):
            return None
        if abs(latest.cx - 0.5) > config.collision_center_tolerance:
            return None
        if abs(lateral_rate) > config.collision_max_lateral_rate:
            return None
        relative_growth = math.exp(relative_rate * float(times[-1]))
        if growth < config.collision_min_growth or relative_growth < config.collision_min_growth:
            return None
        if relative_rate < config.collision_min_log_rate or fit < config.collision_min_fit:
            return None
        if foot_rate < 0.0:
            return None
        tau = 1.0 / relative_rate if relative_rate > 1e-6 else None
        severity = "critical" if (tau is not None and tau <= 1.5) or latest.size >= 0.22 else "warning"
        evidence = {
            "rule": "IMAGE_SCALE_EXPANSION_CONFIRMED",
            "severity": severity,
            "bbox_scale_growth": growth,
            "relative_log_expansion_per_s": relative_rate,
            "background_log_expansion_per_s": background_rate,
            "visual_time_to_contact_s": tau,
            "fit_r2": fit,
            "lateral_rate_fraction_per_s": lateral_rate,
            "foot_rate_per_s": foot_rate,
            "window_s": float(times[-1]),
            "samples": len(samples),
            "foot_fraction": latest.foot,
            "distance_note": "IMAGE_SPACE_CUE_NOT_METRIC_RANGE_OR_SPEED",
            "corridor_source": corridor.source,
        }
        confidence = min(0.95, 0.45 + 0.30 * fit
                         + 0.20 * min(1.0, relative_rate / 1.0))
        return RuleHit(COLLISION, confidence, evidence,
                       assessment=(CRITICAL if severity == "critical"
                                   else WARNING),
                       priority=410)

    def _required_confirmations(self, kind: str) -> int:
        config = self.config
        if kind == CUT_IN:
            return config.cut_in_confirmations
        if kind in (PEDESTRIAN_ENTRY, PEDESTRIAN_CROSSING,
                    TWO_WHEELER_CONFLICT, CROSSWALK_RISK):
            return config.entry_confirmations
        if kind in (NEAR_LEAD,):
            return config.near_confirmations
        if kind in (ONCOMING_CONFLICT,):
            return config.oncoming_conflict_frames
        if kind in (RED_OBSERVED,):
            return 1
        return config.collision_confirmations

    def _apply_state_machine(self, scene, hits: dict[tuple[str, int], RuleHit],
                             now_s: float) -> list[HazardEvent]:
        config = self.config
        emitted: list[HazardEvent] = []
        for key, state in list(self._states.items()):
            if key in hits:
                continue
            if key[0] == NEAR_LEAD and (COLLISION, key[1]) in hits:
                continue
            self._miss_state(scene, state, now_s, emitted)
        self._cooldowns = {key: until for key, until in self._cooldowns.items()
                           if now_s < until}
        for key, hit in hits.items():
            state = self._states.get(key)
            if state is None:
                if key in self._cooldowns:
                    continue
                state = _EventState(kind=hit.kind, track_id=key[1],
                                    target_class=self._track_class(key[1]),
                                    confirmed=False, streak=0, missed=0)
                self._states[key] = state
            if not state.confirmed:
                state.streak += 1
                required = self._required_confirmations(hit.kind)
                if state.streak < required:
                    continue
                if (self._duplicate_of_active(hit.kind, key[1])
                        or self._duplicate_of_recent(hit.kind, key[1], now_s)):
                    self._states.pop(key, None)
                    self._cooldowns[key] = (
                        now_s + self.config.rearm_cooldown_s)
                    continue
                state.confirmed = True
                state.first_frame_id = int(scene.frame_id)
                state.first_media_s = now_s
                state.last_hit_s = now_s
                track = self._tracks.get(state.track_id)
                if track is not None and "from_cx" in hit.evidence:
                    track.entered_at_s = now_s
                event = self._make_event(scene, state, hit, TRIGGERED, now_s)
                state.last_event = event
                emitted.append(event)
                if hit.kind == COLLISION:
                    self._supersede_near(scene, state.track_id, now_s, emitted)
                continue
            state.last_hit_s = now_s
            state.missed = 0
            event = self._make_event(scene, state, hit, ACTIVE, now_s)
            state.last_event = event
            emitted.append(event)
            if hit.kind == COLLISION:
                self._supersede_near(scene, state.track_id, now_s, emitted)
        return emitted

    def _supersede_near(self, scene, track_id: int, now_s: float,
                        emitted: list[HazardEvent]) -> None:
        key = (NEAR_LEAD, track_id)
        near = self._states.get(key)
        if near is None or not near.confirmed:
            return
        emitted.append(self._cleared_event(scene, near, now_s,
                                           "superseded_by_collision_risk"))
        self._states.pop(key, None)
        self._cooldowns[key] = now_s + self.config.rearm_cooldown_s
        self._remember_cleared(near, now_s)

    def _duplicate_of_recent(self, kind: str, track_id: int,
                             now_s: float) -> bool:
        track = self._tracks.get(track_id)
        if track is None:
            return False
        for entry in self._recent_cleared:
            if entry["kind"] != kind:
                continue
            if now_s - entry["media_s"] > self.config.rearm_cooldown_s:
                continue
            if _iou(track.bbox, entry["bbox"]) > 0.25:
                return True
            if (abs(track.latest.cx - entry["cx"])
                    < self.config.duplicate_center_fraction
                    and abs(track.latest.foot - entry["foot"])
                    < self.config.duplicate_foot_fraction):
                return True
        return False

    def _remember_cleared(self, state: _EventState, now_s: float) -> None:
        track = self._tracks.get(state.track_id)
        if track is None:
            return
        self._recent_cleared.append({
            "kind": state.kind,
            "media_s": now_s,
            "bbox": track.bbox,
            "cx": track.latest.cx,
            "foot": track.latest.foot,
        })

    def _duplicate_of_active(self, kind: str, track_id: int) -> bool:
        mine = self._tracks.get(track_id)
        if mine is None:
            return False
        for (other_kind, other_id), other_state in self._states.items():
            if other_kind != kind or other_id == track_id:
                continue
            if not other_state.confirmed:
                continue
            other = self._tracks.get(other_id)
            if other is None:
                continue
            if _iou(mine.bbox, other.bbox) > 0.30:
                return True
            if (abs(mine.latest.cx - other.latest.cx)
                    < self.config.duplicate_center_fraction
                    and abs(mine.latest.foot - other.latest.foot)
                    < self.config.duplicate_foot_fraction):
                return True
        return False

    def _miss_state(self, scene, state: _EventState, now_s: float,
                    emitted: list[HazardEvent]) -> None:
        config = self.config
        key = (state.kind, state.track_id)
        if not state.confirmed:
            state.streak = max(0, state.streak - 1)
            if state.streak == 0:
                self._states.pop(key, None)
            return
        if now_s - state.last_hit_s >= config.hold_ttl_s:
            emitted.append(self._cleared_event(scene, state, now_s,
                                               "validity_expired"))
            self._states.pop(key, None)
            self._rearm(state, now_s)
            return
        state.missed += 1
        elapsed = now_s - state.first_media_s
        if (state.missed >= config.clear_confirmations
                and elapsed >= config.min_active_s):
            emitted.append(self._cleared_event(scene, state, now_s,
                                               "condition_cleared"))
            self._states.pop(key, None)
            self._rearm(state, now_s)
            return
        if state.last_event is not None:
            held = replace(state.last_event, state=ACTIVE,
                           evidence={**state.last_event.evidence,
                                     "held": True,
                                     "missed_observations": state.missed})
            emitted.append(held)

    def _rearm(self, state: _EventState, now_s: float) -> None:
        self._cooldowns[(state.kind, state.track_id)] = (
            now_s + self.config.rearm_cooldown_s)
        self._remember_cleared(state, now_s)
        track = self._tracks.get(state.track_id)
        if track is not None:
            track.entered_at_s = None

    def _cleared_event(self, scene, state: _EventState, now_s: float,
                       reason: str) -> HazardEvent:
        if state.last_event is None:
            hit = RuleHit(state.kind, 0.0, {"rule": reason})
            base = self._make_event(scene, state, hit, CLEARED, now_s)
        else:
            base = replace(state.last_event, state=CLEARED, frame_id=int(scene.frame_id),
                           media_s=now_s, valid_until_s=now_s,
                           evidence={**state.last_event.evidence,
                                     "cleared_reason": reason})
        return base

    def _make_event(self, scene, state: _EventState, hit: RuleHit,
                    event_state: str, now_s: float) -> HazardEvent:
        track = self._tracks.get(state.track_id)
        bbox = track.bbox if track is not None else None
        severity = hit.evidence.get(
            "severity", ASSESSMENT_SEVERITY.get(hit.assessment, "warning"))
        track = self._tracks.get(state.track_id)
        target_class = ((track.display_class if track is not None else "")
                        or state.target_class)
        if hit.kind in (RED_OBSERVED, RED_CONFLICT):
            target_class = "traffic light"
        return HazardEvent(
            kind=hit.kind,
            severity=severity,
            state=event_state,
            track_id=state.track_id,
            source_id=str(scene.source_id),
            generation=int(scene.generation),
            frame_id=int(scene.frame_id),
            media_s=now_s,
            first_frame_id=state.first_frame_id,
            first_media_s=state.first_media_s,
            ttl_s=self.config.hold_ttl_s,
            valid_until_s=now_s + self.config.hold_ttl_s,
            confidence=hit.confidence,
            evidence=dict(hit.evidence),
            bbox=bbox,
            target_class=target_class,
            assessment=hit.assessment,
            priority=hit.priority,
        )

    def _track_class(self, track_id: int) -> str:
        track = self._tracks.get(track_id)
        return track.class_name if track is not None else ""

    def _miss_all(self, scene, now_s: float, *, reason: str) -> list[HazardEvent]:
        emitted: list[HazardEvent] = []
        for state in list(self._states.values()):
            self._miss_state(scene, state, now_s, emitted)
        return emitted

    def _track_summary(self, corridor: EgoCorridor) -> list[dict]:
        result = []
        for track_id, state in sorted(self._tracks.items()):
            sample = state.latest
            result.append({
                "track_id": int(track_id),
                "class_name": state.class_name,
                "display_class": state.display_class,
                "family": state.family,
                "bbox": [round(float(value), 2) for value in state.bbox],
                "cx": sample.cx,
                "foot": sample.foot,
                "samples": len(state.history),
                "age_s": state.age_s,
                "in_corridor": corridor.contains(sample.cx, sample.foot),
            })
        return result

    def _observation(self, scene, corridor, events, tracks,
                     reason: str) -> FusionObservation:
        return FusionObservation(
            source_id=str(scene.source_id),
            generation=int(scene.generation),
            frame_id=int(scene.frame_id),
            media_s=float(scene.media_s),
            events=tuple(events),
            tracks=tuple(tracks),
            reason=reason,
            corridor=corridor)

    def current(self, now_media_s: float) -> FusionObservation:
        observed = self._last_observation
        if abs(float(now_media_s) - observed.media_s) > self.config.hold_ttl_s:
            return FusionObservation(
                source_id=observed.source_id,
                generation=observed.generation,
                frame_id=observed.frame_id,
                media_s=observed.media_s,
                reason="observation_expired",
                corridor=observed.corridor)
        return observed


class _StaticMotion:
    valid = False
    dx = 0.0
    dy = 0.0
    rotation_dps = 0.0
    scale_rate = 0.0

    @property
    def stable(self) -> bool:
        return True
