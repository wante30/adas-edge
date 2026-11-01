"""Platform-neutral RoadScene extraction from MTL-FAST segmentation masks.

This module is the shared algorithm boundary for Windows ONNX and RK3566
RKNN C API. It never imports onnxruntime, rknn, Qt, FFmpeg or FPGA code.

RoadScene coordinates are normalized to the source image:
(0,0) top-left, (1,1) bottom-right.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from competition_ai.road_geometry import (
    boundary_candidates,
    crosswalk_polygons,
    drivable_surface,
    lane_markings,
    model_to_source,
    scanline_bounds,
    smooth_bounds,
)
from competition_ai.road_topology import (
    RESET_GAP_S,
    LaneTopology,
    LaneTopologyTracker,
    single_frame_topology,
)
from competition_ai.scene_observation import SceneObservation

SCAN_LEVELS = tuple(float(value) for value in np.linspace(0.52, 0.86, 12))
CROSSWALK_HOLD_S = 0.35
LANE_HOLD_S = 0.25
BOUNDARY_HOLD_S = 0.30
CROSSWALK_BLOCK_WIDTH = 0.55
CROSSWALK_BLOCK_FRAMES = 2


@dataclass
class LaneMarking:
    category: int
    category_name: str
    style: str
    double: bool
    confidence: float
    points: list[tuple[float, float]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": int(self.category),
            "name": self.category_name,
            "style": self.style,
            "double": bool(self.double),
            "confidence": round(float(self.confidence), 3),
            "points": [[round(float(x), 4), round(float(y), 4)]
                       for x, y in self.points],
        }


@dataclass
class Crosswalk:
    polygon: list[tuple[float, float]]
    bbox: tuple[float, float, float, float]
    center_y: float
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "polygon": [[round(float(x), 4), round(float(y), 4)]
                        for x, y in self.polygon],
            "bbox": [round(float(value), 4) for value in self.bbox],
            "center_y": round(float(self.center_y), 4),
            "confidence": round(float(self.confidence), 3),
        }


@dataclass
class RoadScene:
    source_id: str
    generation: int
    frame_id: int
    media_s: float
    valid: bool
    source_width: int = 1280
    source_height: int = 720
    left: list[tuple[float, float]] = field(default_factory=list)
    right: list[tuple[float, float]] = field(default_factory=list)
    corridor: list[tuple[float, float]] = field(default_factory=list)
    lane_markings: list[LaneMarking] = field(default_factory=list)
    crosswalks: list[Crosswalk] = field(default_factory=list)
    topology: LaneTopology = field(default_factory=LaneTopology)
    # Raw per-frame lane_cat candidates (diagnostics / tracker input).
    boundary_candidates: list[dict] = field(default_factory=list)
    segment_kind: str = "direct"
    observed: bool = False
    stale: bool = False
    age_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "generation": int(self.generation),
            "frame_id": int(self.frame_id),
            "media_s": round(float(self.media_s), 3),
            "valid": bool(self.valid),
            "fallback": not bool(self.valid),
            "observed": bool(self.observed),
            "stale": bool(self.stale),
            "age_ms": round(float(self.age_ms), 1),
            "source_width": int(self.source_width),
            "source_height": int(self.source_height),
            "left": [[round(float(x), 4), round(float(y), 4)]
                     for x, y in self.left],
            "right": [[round(float(x), 4), round(float(y), 4)]
                      for x, y in self.right],
            "corridor": [[round(float(x), 4), round(float(y), 4)]
                         for x, y in self.corridor],
            "lanes": [item.as_dict() for item in self.lane_markings],
            "crosswalks": [item.as_dict() for item in self.crosswalks],
            "topology": self.topology.as_dict(),
            "segment_kind": self.segment_kind,
        }


def _empty(observation: SceneObservation) -> RoadScene:
    return RoadScene(
        source_id=observation.source_id,
        generation=observation.generation,
        frame_id=observation.frame_id,
        media_s=observation.media_s,
        valid=False,
        source_width=observation.source_width,
        source_height=observation.source_height)


def extract_road_scene(observation: SceneObservation) -> RoadScene:
    """Extract an unfiltered RoadScene from one observation."""
    if (observation.lane_cat is None or observation.lane_style is None
            or observation.drivable is None or not observation.source_valid):
        return _empty(observation)

    mask_shape = observation.lane_cat.shape
    letterbox = observation.letterbox
    source_w = observation.source_width
    source_h = observation.source_height

    boundary, ego, used_direct = drivable_surface(observation.drivable)
    if boundary is None:
        scene = _empty(observation)
        scene.boundary_candidates = boundary_candidates(
            observation.lane_cat, observation.lane_style, mask_shape,
            letterbox, source_w, source_h)
        scene.topology = single_frame_topology(scene.boundary_candidates)
        scene.lane_markings = [
            LaneMarking(**item) for item in lane_markings(
                observation.lane_cat, observation.lane_style, mask_shape,
                letterbox, source_w, source_h)]
        scene.crosswalks = [
            Crosswalk(**item) for item in crosswalk_polygons(
                observation.lane_cat, mask_shape, letterbox, source_w, source_h)]
        return scene

    height, width = boundary.shape
    ego_u = width * 0.5
    samples = _samples_to_source(
        smooth_bounds(scanline_bounds(
            boundary, SCAN_LEVELS, ego_u, bridged_gap=0.22)),
        mask_shape, letterbox, source_w, source_h)
    ego_samples = []
    if ego is not None and ego is not boundary:
        ego_samples = _samples_to_source(
            smooth_bounds(scanline_bounds(
                ego, SCAN_LEVELS, ego_u, bridged_gap=0.05)),
            mask_shape, letterbox, source_w, source_h)

    valid = False
    if len(samples) >= 4:
        nearest = samples[:min(5, len(samples))]
        widths = sorted(item["width"] for item in nearest)
        median_width = widths[len(widths) // 2]
        centers = sorted(item["center"] for item in nearest)
        center_near = centers[len(centers) // 2]
        spans_center = any(item["left"] < 0.5 < item["right"]
                           for item in nearest)
        valid = (median_width >= 0.06
                 and (spans_center or 0.30 <= center_near <= 0.70))

    left = [(item["left"], item["y"]) for item in samples]
    right = [(item["right"], item["y"]) for item in samples]
    corridor = _corridor_from_samples(samples, ego_samples)
    # Lane boundaries come from lane_cat alone and are independent of the
    # drivable validity: a vehicle hiding the drivable surface must not
    # remove painted lines that are still visible.
    candidates = boundary_candidates(
        observation.lane_cat, observation.lane_style, mask_shape, letterbox,
        source_w, source_h)
    topology = single_frame_topology(candidates)

    lanes = [
        LaneMarking(
            category=int(item["category"]),
            category_name=str(item["category_name"]),
            style=str(item["style"]),
            double=bool(item["double"]),
            confidence=float(item["confidence"]),
            points=[(float(x), float(y)) for x, y in item["points"]],
        )
        for item in lane_markings(
            observation.lane_cat, observation.lane_style, mask_shape,
            letterbox, source_w, source_h)
    ]
    crosswalks = [
        Crosswalk(
            polygon=[(float(x), float(y)) for x, y in item["polygon"]],
            bbox=tuple(float(value) for value in item["bbox"]),
            center_y=float(item["center_y"]),
            confidence=float(item["confidence"]),
        )
        for item in crosswalk_polygons(
            observation.lane_cat, mask_shape, letterbox, source_w, source_h)
    ]
    return RoadScene(
        source_id=observation.source_id,
        generation=observation.generation,
        frame_id=observation.frame_id,
        media_s=observation.media_s,
        valid=bool(valid),
        source_width=source_w,
        source_height=source_h,
        left=[(float(x), float(y)) for x, y in left],
        right=[(float(x), float(y)) for x, y in right],
        corridor=[(float(x), float(y)) for x, y in corridor],
        lane_markings=lanes,
        crosswalks=crosswalks,
        topology=topology,
        boundary_candidates=candidates,
        segment_kind="direct" if used_direct else "alternative",
        observed=bool(valid),
        stale=False,
        age_ms=0.0,
    )


class RoadSceneStabilizer:
    """Small temporal filter for display geometry.

    Raw masks are 256x144 and can flicker. This keeps boundaries, lanes and
    crosswalks stable for a few frames, but resets completely on generation,
    seek/replay or a media-time gap. It never carries geometry across video
    sessions.
    """

    def __init__(self) -> None:
        self.generation: int | None = None
        self.last_media_s = -1.0
        self.last_scene: RoadScene | None = None
        self.last_valid_media_s = -1.0
        self.last_crosswalks: list[Crosswalk] = []
        self.last_crosswalk_media_s = -1.0
        self.last_lanes: list[LaneMarking] = []
        self.last_lane_media_s = -1.0
        self.topology_tracker = LaneTopologyTracker()
        self.crosswalk_block_hits = 0

    def reset(self) -> None:
        self.generation = None
        self.last_media_s = -1.0
        self.last_scene = None
        self.last_valid_media_s = -1.0
        self.last_crosswalks = []
        self.last_crosswalk_media_s = -1.0
        self.last_lanes = []
        self.last_lane_media_s = -1.0
        self.topology_tracker.reset()
        self.crosswalk_block_hits = 0

    def _check_generation(self, observation: SceneObservation) -> None:
        if self.generation != observation.generation:
            self.reset()
            self.generation = observation.generation
        if (self.last_media_s >= 0.0
                and (observation.media_s < self.last_media_s - 0.05
                     or observation.media_s - self.last_media_s > RESET_GAP_S)):
            self.reset()
            self.generation = observation.generation
        self.last_media_s = observation.media_s

    def update(self, observation: SceneObservation) -> RoadScene:
        self._check_generation(observation)
        fresh = extract_road_scene(observation)
        # Crosswalk stripes run along the road and converge to the vanishing
        # point like lane lines; inside a crosswalk lane_cat pieces are not
        # lane boundaries. They are withheld from the tracker and, when the
        # crosswalk covers the evaluation band in front of ego, topology is
        # reported unknown instead of counting stripes as lanes.
        candidates = _outside_crosswalks(fresh.boundary_candidates, fresh.crosswalks)
        # Without masks there is no evidence at all; candidates=None keeps
        # the tracker ageing its boundaries rather than inventing any.
        fresh.topology = self.topology_tracker.update(
            candidates if observation.masks_available else None,
            observation.media_s, observation.generation,
            observation.frame_id)
        # A one-frame crosswalk blob (arrow, stop line) must not blank the
        # topology; a real crossing persists over consecutive AI frames.
        if _crosswalk_blocks_reference(fresh.crosswalks, fresh.topology):
            self.crosswalk_block_hits += 1
        else:
            self.crosswalk_block_hits = 0
        if self.crosswalk_block_hits >= CROSSWALK_BLOCK_FRAMES:
            fresh.topology = LaneTopology(
                boundaries=fresh.topology.boundaries,
                reason="crosswalk / intersection at reference height")

        # Lanes and crosswalks are independent of road-boundary validity. A
        # taxi occluding the drivable mask must not erase a real crosswalk.
        if fresh.crosswalks:
            self.last_crosswalks = fresh.crosswalks
            self.last_crosswalk_media_s = observation.media_s
        elif (self.last_crosswalks
              and observation.media_s - self.last_crosswalk_media_s <= CROSSWALK_HOLD_S):
            fresh.crosswalks = list(self.last_crosswalks)
        else:
            self.last_crosswalks = []
            fresh.crosswalks = []

        if fresh.lane_markings:
            self.last_lanes = fresh.lane_markings
            self.last_lane_media_s = observation.media_s
        elif (self.last_lanes
              and observation.media_s - self.last_lane_media_s <= LANE_HOLD_S):
            fresh.lane_markings = list(self.last_lanes)
        else:
            self.last_lanes = []
            fresh.lane_markings = []

        if fresh.valid:
            if (self.last_scene is not None and self.last_scene.valid
                    and observation.media_s - self.last_valid_media_s < BOUNDARY_HOLD_S):
                previous = {round(y, 3): (x, y) for x, y in self.last_scene.left}
                smoothed_left = []
                for x, y in fresh.left:
                    old = previous.get(round(y, 3))
                    smoothed_left.append(
                        (0.65 * x + 0.35 * old[0], y) if old else (x, y))
                previous_r = {round(y, 3): (x, y) for x, y in self.last_scene.right}
                smoothed_right = []
                for x, y in fresh.right:
                    old = previous_r.get(round(y, 3))
                    smoothed_right.append(
                        (0.65 * x + 0.35 * old[0], y) if old else (x, y))
                fresh.left = smoothed_left
                fresh.right = smoothed_right
                fresh.corridor = _corridor_from_bounds(fresh.left, fresh.right)
            fresh.observed = True
            fresh.stale = False
            fresh.age_ms = 0.0
            self.last_valid_media_s = observation.media_s
            # Candidates are per-frame debug data and never reused when the
            # scene is held; do not pay for copying them.
            candidates_now = fresh.boundary_candidates
            fresh.boundary_candidates = []
            self.last_scene = copy.deepcopy(fresh)
            fresh.boundary_candidates = candidates_now
            return fresh

        # Current geometry is unrecoverable. Hold the last trusted curve for a
        # very short display window; mark it stale so hazard logic never treats
        # it as a current observation.
        age_s = max(0.0, observation.media_s - self.last_valid_media_s)
        if (self.last_scene is not None and self.last_scene.valid
                and age_s <= BOUNDARY_HOLD_S):
            held = copy.deepcopy(self.last_scene)
            held.frame_id = observation.frame_id
            held.media_s = observation.media_s
            held.generation = observation.generation
            held.valid = True
            held.observed = False
            held.stale = True
            held.age_ms = round(age_s * 1000.0, 1)
            held.crosswalks = fresh.crosswalks
            held.lane_markings = fresh.lane_markings
            held.topology = fresh.topology
            held.boundary_candidates = fresh.boundary_candidates
            return held

        fresh.valid = False
        fresh.observed = False
        fresh.stale = False
        fresh.age_ms = 0.0
        return fresh


def _samples_to_source(samples: list[dict], mask_shape, letterbox,
                       source_width: int, source_height: int):
    if not samples:
        return samples
    mask_h, mask_w = mask_shape
    result = []
    for item in samples:
        left_x, left_y = model_to_source(
            item["left"] * mask_w, item["y"] * (mask_h - 1),
            mask_shape, letterbox, source_width, source_height)
        right_x, right_y = model_to_source(
            item["right"] * mask_w, item["y"] * (mask_h - 1),
            mask_shape, letterbox, source_width, source_height)
        result.append({
            "y": float((left_y + right_y) * 0.5),
            "left": float(left_x),
            "right": float(right_x),
            "center": float((left_x + right_x) * 0.5),
            "width": float(max(0.0, right_x - left_x)),
        })
    result.sort(key=lambda item: item["y"], reverse=True)
    return result


def _corridor_from_samples(boundary_samples: list[dict],
                           ego_samples: list[dict]):
    source = None
    if len(ego_samples) >= 3:
        centers = sorted(item["center"] for item in ego_samples)
        widths = sorted(item["width"] for item in ego_samples)
        center = centers[len(centers) // 2]
        width = widths[len(widths) // 2]
        if 0.30 <= center <= 0.70 and width >= 0.05:
            source = ego_samples
    points = []
    if source is not None:
        for item in source:
            points.append((item["left"] + 0.02, item["y"]))
        for item in reversed(source):
            points.append((item["right"] - 0.02, item["y"]))
        return points
    for item in boundary_samples:
        half = min(0.14, max(0.035, item["width"] * 0.22))
        points.append((item["center"] - half, item["y"]))
    for item in reversed(boundary_samples):
        half = min(0.14, max(0.035, item["width"] * 0.22))
        points.append((item["center"] + half, item["y"]))
    return points


def _corridor_from_bounds(left: list[tuple[float, float]],
                          right: list[tuple[float, float]]):
    if len(left) != len(right) or len(left) < 2:
        return []
    points = []
    for (lx, y), (rx, _) in zip(left, right):
        half = min(0.14, max(0.035, (rx - lx) * 0.22))
        center = (lx + rx) * 0.5
        points.append((center - half, y))
    for (lx, y), (rx, _) in reversed(list(zip(left, right))):
        half = min(0.14, max(0.035, (rx - lx) * 0.22))
        center = (lx + rx) * 0.5
        points.append((center + half, y))
    return points


def _outside_crosswalks(candidates: list[dict], crosswalks) -> list[dict]:
    # Narrow "crosswalk" blobs are mostly arrows / stop lines on a normal
    # road: they must not hide lane-line pieces that pass through them.
    crosswalks = [cw for cw in crosswalks
                  if cw.bbox[2] - cw.bbox[0] >= CROSSWALK_BLOCK_WIDTH]
    if not crosswalks:
        return list(candidates)
    result = []
    for cand in candidates:
        points = cand.get("points") or []
        inside = 0
        for x, y in points:
            for cw in crosswalks:
                x0, y0, x1, y1 = cw.bbox
                if x0 <= x <= x1 and y0 - 0.02 <= y <= y1 + 0.02:
                    inside += 1
                    break
        if points and inside >= 0.5 * len(points):
            continue
        result.append(cand)
    return result


def _crosswalk_blocks_reference(crosswalks, topology: LaneTopology) -> bool:
    y = topology.reference_y
    x = topology.ego_reference_norm
    # lane_cat "crosswalk" also fires on arrows and stop lines; only a wide
    # crossing that spans ego and the reference row counts as a junction.
    for cw in crosswalks:
        x0, y0, x1, y1 = cw.bbox
        if (x1 - x0 >= CROSSWALK_BLOCK_WIDTH and x0 <= x <= x1
                and y0 <= y <= y1 + 0.02):
            return True
    return False
