"""Lane topology from MTL-FAST lane_cat boundary candidates.

Shared by Windows and RK3566: NumPy only, no Qt / ONNX / RKNN / FPGA types.

Observation model
-----------------
* Longitudinal boundaries come only from lane_cat via
  road_geometry.boundary_candidates(). drivable is never used to clip or to
  invent a boundary: it is "where ego may drive", not the carriageway.
* LaneBoundaryTracker keeps each boundary as a curve sampled on a fixed
  image-height grid, matches new candidates against the predicted curve
  (mean lateral error over the shared heights, not a single x), tolerates a
  few missed AI frames and never survives a generation change / seek.
* build_topology() orders the boundaries at a reference height and turns
  adjacent pairs into lane intervals. At a fixed image height all lanes of a
  flat road have the same image width, so the interval width is checked
  against the ego/typical lane width:
    - much narrower  -> double line / edge-line+curb gap, not a lane;
    - much wider     -> a boundary is missing: interval is "unknown", it is
                        neither counted nor split;
    - otherwise      -> lane.
  An outermost dashed line is evidence of a lane beyond it, reported as a
  "partial" (open) lane, never as an invented boundary.
* lane counts are explicit: closed (both boundaries seen), visible (closed +
  partial) and total (only when both road edges are evidenced, else None).
* the ego reference point is a fixed image point (camera axis approximation,
  x=0.5 at the reference height); ego offset and lane index are geometry
  relative to that point, the lane is never re-centred on the ego.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CONFIRM_FRAMES = 3          # hits before a new boundary track is used
TOPOLOGY_CONFIRM = 3        # frames before a new lane signature replaces the old
MAX_MISSED_S = 0.6          # boundary kept as "tracked" at most this long ...
MAX_MISSED_FRAMES = 3       # ... but always for at least this many AI frames
RESET_GAP_S = 3.0           # forward jump treated as a seek (AI may run at
                            # 1-3 fps under GPU load; real seeks also bump
                            # the generation, which always resets)
Y_GRID = np.round(np.linspace(0.56, 0.96, 21), 3)
REFERENCE_Y = 0.80          # topology / ego evaluation height (normalized)
EGO_REFERENCE_X = 0.50      # camera-axis approximation, no calibration yet
EXTRAPOLATE_DOWN = 0.14     # candidate may be extended toward the camera
MATCH_COST = 0.032          # mean |dx| (normalized) for track association
DUPLICATE_COST = 0.014      # two tracks this close are the same line
SPAWN_QUALITY = 0.35
SPAWN_MIN_Y_BOTTOM = 0.78   # new tracks must be observed near REFERENCE_Y;
                             # far-only pieces may only extend existing tracks
NARROW_RATIO = 0.62         # flat road: all lanes share one image width per row
WIDE_RATIO = 1.60
STRADDLE_RATIO = 0.08       # ego this close to a line -> lane index unknown
ABS_LANE_WIDTH = (0.09, 0.45)   # at REFERENCE_Y when no typical width exists
DIVIDER_CATEGORIES = (1, 2, 3)  # double lines: treated as carriageway edge


def _r(value: float) -> float:
    return round(float(value), 4)


@dataclass
class Boundary:
    boundary_id: int
    kind: str                       # lane | curb
    category: int
    style: str                      # solid | dashed | unknown
    points: list[tuple[float, float]] = field(default_factory=list)
    confidence: float = 0.0         # geometric quality 0..1
    inferred: bool = False          # kept for consumers; never True here
    state: str = "observed"         # observed | tracked
    hits: int = 1
    missed: int = 0
    first_seen_s: float = 0.0
    last_seen_s: float = 0.0
    x_ref: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": int(self.boundary_id),
            "kind": self.kind,
            "category": int(self.category),
            "style": self.style,
            "confidence": round(float(self.confidence), 3),
            "inferred": bool(self.inferred),
            "state": self.state,
            "hits": int(self.hits),
            "missed": int(self.missed),
            "first_seen_s": round(float(self.first_seen_s), 3),
            "last_seen_s": round(float(self.last_seen_s), 3),
            "x_ref": None if self.x_ref is None else _r(self.x_ref),
            "points": [[_r(x), _r(y)] for x, y in self.points],
        }


@dataclass
class Lane:
    index: int
    left: str                       # boundary id, or "open"
    right: str
    points: list[tuple[float, float]] = field(default_factory=list)
    left_points: list[tuple[float, float]] = field(default_factory=list)
    right_points: list[tuple[float, float]] = field(default_factory=list)
    status: str = "observed"        # observed | tracked | partial
    width_ref: float = 0.0
    center_ref: float = 0.0
    is_ego: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "left": self.left,
            "right": self.right,
            "status": self.status,
            "width_ref": _r(self.width_ref),
            "center_ref": _r(self.center_ref),
            "is_ego": bool(self.is_ego),
            "points": [[_r(x), _r(y)] for x, y in self.points],
            "left_points": [[_r(x), _r(y)] for x, y in self.left_points],
            "right_points": [[_r(x), _r(y)] for x, y in self.right_points],
        }


@dataclass
class LaneTopology:
    valid: bool = False
    observed: bool = False
    stale: bool = False
    age_ms: float = 0.0
    status: str = "unknown"         # observed | tracked | pending | unknown
    reason: str = ""
    lane_count: int | None = None   # == visible_lane_count (compatibility)
    closed_lane_count: int | None = None
    visible_lane_count: int | None = None
    total_lane_count: int | None = None
    left_edge_known: bool = False
    right_edge_known: bool = False
    ego_lane_index: int | None = None
    ego_reference_norm: float = EGO_REFERENCE_X
    reference_y: float = REFERENCE_Y
    ego_center_norm: float | None = None
    ego_offset_norm: float | None = None     # ego_x - lane_center (+ = right)
    ego_offset_ratio: float | None = None    # offset / lane width (-0.5..0.5)
    ego_near_boundary: bool = False
    lanes_left_of_ego: int | None = None    # visible lanes left of ego lane
    lanes_right_of_ego: int | None = None
    boundaries: list[Boundary] = field(default_factory=list)
    lanes: list[Lane] = field(default_factory=list)
    unknown_gaps: list[tuple[float, float]] = field(default_factory=list)
    confidence: float = 0.0

    def signature(self) -> tuple:
        return (self.valid, self.visible_lane_count, self.closed_lane_count,
                self.ego_lane_index)

    def as_dict(self) -> dict[str, Any]:
        def opt(value):
            return None if value is None else _r(value)
        return {
            "valid": bool(self.valid),
            "observed": bool(self.observed),
            "stale": bool(self.stale),
            "age_ms": round(float(self.age_ms), 1),
            "status": self.status,
            "reason": self.reason,
            "lane_count": self.lane_count,
            "closed_lane_count": self.closed_lane_count,
            "visible_lane_count": self.visible_lane_count,
            "total_lane_count": self.total_lane_count,
            "left_edge_known": bool(self.left_edge_known),
            "right_edge_known": bool(self.right_edge_known),
            "ego_lane_index": self.ego_lane_index,
            "ego_reference_norm": _r(self.ego_reference_norm),
            "reference_y": _r(self.reference_y),
            "ego_center_norm": opt(self.ego_center_norm),
            "ego_offset_norm": opt(self.ego_offset_norm),
            "ego_offset_ratio": opt(self.ego_offset_ratio),
            "ego_near_boundary": bool(self.ego_near_boundary),
            "lanes_left_of_ego": self.lanes_left_of_ego,
            "lanes_right_of_ego": self.lanes_right_of_ego,
            "confidence": round(float(self.confidence), 3),
            "unknown_gaps": [[_r(a), _r(b)] for a, b in self.unknown_gaps],
            "boundaries": [item.as_dict() for item in self.boundaries],
            "lanes": [item.as_dict() for item in self.lanes],
        }


# ---------------------------------------------------------------------------
# Curve sampling
# ---------------------------------------------------------------------------
def sample_candidate(candidate: dict) -> np.ndarray:
    """Candidate x on Y_GRID; NaN outside its support (+ short extension)."""
    coef = np.asarray(candidate["coef"], dtype=np.float64)
    y_top = float(candidate["y_top"])
    y_bot = float(candidate["y_bottom"])
    xs = np.polyval(coef, Y_GRID)
    # Below the observed end use the tangent line: a quadratic must not bend
    # off-screen when extrapolated toward the camera.
    slope = float(np.polyval(np.polyder(coef), y_bot)) if coef.size > 1 else 0.0
    x_bot = float(np.polyval(coef, y_bot))
    below = Y_GRID > y_bot
    xs[below] = x_bot + slope * (Y_GRID[below] - y_bot)
    # A short piece predicts its own continuation poorly: extend at most
    # about its own length toward the camera.
    extend = min(EXTRAPOLATE_DOWN, max(0.03, 1.2 * (y_bot - y_top)))
    support = (Y_GRID >= y_top - 0.02) & (Y_GRID <= y_bot + extend)
    xs[~support] = np.nan
    xs[(xs < -0.3) | (xs > 1.3)] = np.nan
    return xs


def _curve_x(xs: np.ndarray, y: float) -> float | None:
    valid = ~np.isnan(xs)
    if valid.sum() < 2:
        return None
    ys = Y_GRID[valid]
    if y < ys.min() - 1e-6 or y > ys.max() + 1e-6:
        return None
    return float(np.interp(y, ys, xs[valid]))


def _curve_cost(a: np.ndarray, b: np.ndarray, min_common: int = 3):
    common = ~np.isnan(a) & ~np.isnan(b)
    if int(common.sum()) < min_common:
        return None
    return float(np.mean(np.abs(a[common] - b[common])))


def _points(xs: np.ndarray) -> list[tuple[float, float]]:
    valid = ~np.isnan(xs)
    pts = [(float(x), float(y)) for x, y in zip(xs[valid], Y_GRID[valid])]
    pts.sort(key=lambda item: item[1], reverse=True)   # near -> far
    return pts


# ---------------------------------------------------------------------------
# Topology from a set of boundaries
# ---------------------------------------------------------------------------
@dataclass
class _Line:
    boundary: Boundary
    xs: np.ndarray
    x: float


def _lane_polygon(left: np.ndarray | None, right: np.ndarray | None):
    if left is None and right is None:
        return [], []
    lx = left if left is not None else np.full_like(right, 0.0)
    rx = right if right is not None else np.full_like(left, 1.0)
    if left is None:
        lx = np.where(np.isnan(rx), np.nan, 0.0)
    if right is None:
        rx = np.where(np.isnan(lx), np.nan, 1.0)
    valid = ~np.isnan(lx) & ~np.isnan(rx) & (rx - lx > 0.01)
    lp = [(float(x), float(y)) for x, y in zip(lx[valid], Y_GRID[valid])]
    rp = [(float(x), float(y)) for x, y in zip(rx[valid], Y_GRID[valid])]
    lp.sort(key=lambda item: item[1], reverse=True)
    rp.sort(key=lambda item: item[1], reverse=True)
    return lp, rp


def _is_edge(line: _Line, neighbours: list[_Line], typical: float) -> bool:
    b = line.boundary
    if b.kind == "curb" or b.category in DIVIDER_CATEGORIES:
        return True
    if b.style == "solid":
        # Solid line with a curb right next to it = carriageway edge line.
        return any(n.boundary.kind == "curb"
                   and abs(n.x - line.x) < NARROW_RATIO * typical
                   for n in neighbours)
    return False


def build_topology(boundaries: list[tuple[Boundary, np.ndarray]],
                   reference_y: float = REFERENCE_Y,
                   ego_x: float = EGO_REFERENCE_X) -> LaneTopology:
    lines: list[_Line] = []
    for boundary, xs in boundaries:
        x = _curve_x(xs, reference_y)
        if x is None or not -0.15 <= x <= 1.15:
            continue
        boundary.x_ref = x
        lines.append(_Line(boundary, xs, x))
    lines.sort(key=lambda item: item.x)
    all_boundaries = [item.boundary for item in lines]
    paint = [item for item in lines if item.boundary.kind == "lane"]
    if len(paint) < 1:
        return LaneTopology(boundaries=all_boundaries,
                            reason="no lane marking at reference height")

    # Curbs only bound the road; lanes are built from paint, plus a curb when
    # no paint edge line sits next to it.
    usable: list[_Line] = []
    for item in lines:
        if item.boundary.kind == "curb":
            near_paint = any(abs(p.x - item.x) < 0.06 for p in paint)
            if near_paint:
                continue
        usable.append(item)

    # Typical lane width: interval around the ego point when plausible,
    # otherwise the median of absolutely plausible intervals.
    gaps = [(a, b) for a, b in zip(usable, usable[1:])]
    widths = [b.x - a.x for a, b in gaps]
    typical = None
    for (a, b), w in zip(gaps, widths):
        if a.x <= ego_x <= b.x and ABS_LANE_WIDTH[0] <= w <= ABS_LANE_WIDTH[1]:
            typical = w
    if typical is None:
        plausible = [w for w in widths
                     if ABS_LANE_WIDTH[0] <= w <= ABS_LANE_WIDTH[1]]
        typical = float(np.median(plausible)) if plausible else None
    if typical is None:
        return LaneTopology(boundaries=all_boundaries,
                            reason="no plausible lane interval")

    # Collapse narrow intervals (double lines, edge line + curb): keep the
    # inner line of the pair as the boundary, remember the outer for edges.
    merged: list[_Line] = []
    for item in usable:
        if merged and item.x - merged[-1].x < NARROW_RATIO * typical:
            previous = merged[-1]

            def rank(line: _Line):
                b = line.boundary
                return (b.kind == "lane", b.state == "observed",
                        b.confidence, -abs(line.x - ego_x))
            # Two lines closer than a lane: double line, edge line + curb,
            # or a far-field piece of a neighbour. Keep the best supported
            # one; never let a curb replace paint.
            keep = item if rank(item) > rank(previous) else previous
            if (previous.boundary.kind == "lane" and item.boundary.kind == "lane"
                    and previous.boundary.category != item.boundary.category):
                keep.boundary.category = max(previous.boundary.category,
                                             item.boundary.category)
            merged[-1] = keep
            continue
        merged.append(item)

    lanes: list[Lane] = []
    unknown_gaps: list[tuple[float, float]] = []
    intervals = []
    for a, b in zip(merged, merged[1:]):
        w = b.x - a.x
        if w > WIDE_RATIO * typical:
            unknown_gaps.append((a.x, b.x))
            intervals.append(None)
            continue
        intervals.append((a, b))

    # Partial lanes beyond an outermost dashed line.
    left_edge = _is_edge(merged[0], lines, typical)
    right_edge = _is_edge(merged[-1], lines, typical)
    entries = []
    if not left_edge and merged[0].boundary.style == "dashed" \
            and merged[0].boundary.kind == "lane":
        entries.append(("partial_left", None, merged[0]))
    for item in intervals:
        entries.append(("gap", None, None) if item is None
                       else ("closed", item[0], item[1]))
    if not right_edge and merged[-1].boundary.style == "dashed" \
            and merged[-1].boundary.kind == "lane":
        entries.append(("partial_right", merged[-1], None))

    index = 0
    ego_lane = None
    for kind, a, b in entries:
        if kind == "gap":
            continue
        if kind == "closed":
            lp, rp = _lane_polygon(a.xs, b.xs)
            status = ("observed" if a.boundary.state == "observed"
                      and b.boundary.state == "observed" else "tracked")
            left_id, right_id = str(a.boundary.boundary_id), str(b.boundary.boundary_id)
            lx, rx = a.x, b.x
        elif kind == "partial_left":
            lp, rp = _lane_polygon(None, b.xs if b else merged[0].xs)
            status, left_id = "partial", "open"
            right_id = str(merged[0].boundary.boundary_id)
            lx, rx = max(0.0, merged[0].x - typical), merged[0].x
        else:
            lp, rp = _lane_polygon(a.xs, None)
            status, right_id = "partial", "open"
            left_id = str(a.boundary.boundary_id)
            lx, rx = a.x, min(1.0, a.x + typical)
        if len(lp) < 2:
            continue
        lane = Lane(index=index, left=left_id, right=right_id,
                    points=list(lp) + list(reversed(rp)),
                    left_points=lp, right_points=rp, status=status,
                    width_ref=rx - lx, center_ref=0.5 * (lx + rx))
        if kind == "closed" and a.x <= ego_x < b.x:
            lane.is_ego = True
            ego_lane = (lane, a, b)
        lanes.append(lane)
        index += 1

    closed = sum(1 for lane in lanes if lane.status != "partial")
    visible = len(lanes)
    topology = LaneTopology(
        boundaries=all_boundaries, lanes=lanes, unknown_gaps=unknown_gaps,
        closed_lane_count=closed, visible_lane_count=visible,
        lane_count=visible, left_edge_known=left_edge,
        right_edge_known=right_edge, reference_y=reference_y,
        ego_reference_norm=ego_x)
    if left_edge and right_edge and not unknown_gaps:
        topology.total_lane_count = visible
    if ego_lane is None:
        near = [item for item in merged if abs(item.x - ego_x) < 0.15 * typical]
        topology.ego_near_boundary = bool(near)
        topology.reason = ("ego on a boundary (lane change)" if near
                           else "ego not inside a closed lane interval")
        return topology
    lane, a, b = ego_lane
    width = b.x - a.x
    center = 0.5 * (a.x + b.x)
    if min(ego_x - a.x, b.x - ego_x) < STRADDLE_RATIO * width:
        # Ego reference sits on a boundary (lane change / straddling): the
        # lane index is ambiguous, so it is not reported.
        lane.is_ego = False
        topology.ego_near_boundary = True
        topology.reason = "ego straddles a boundary (lane change)"
        return topology
    topology.valid = True
    topology.ego_lane_index = lane.index
    topology.lanes_left_of_ego = lane.index
    topology.lanes_right_of_ego = visible - lane.index - 1
    topology.ego_center_norm = center
    topology.ego_offset_norm = ego_x - center
    topology.ego_offset_ratio = (ego_x - center) / max(1e-6, width)
    topology.ego_near_boundary = abs(topology.ego_offset_ratio) > 0.35
    states = [a.boundary.state, b.boundary.state]
    topology.status = "observed" if all(s == "observed" for s in states) else "tracked"
    topology.observed = topology.status == "observed"
    topology.confidence = float(min(a.boundary.confidence, b.boundary.confidence))
    return topology


def single_frame_topology(candidates: list[dict]) -> LaneTopology:
    """Topology of one frame without history (diagnostics, unit tests)."""
    items = []
    for index, cand in enumerate(candidates):
        if cand["quality"] < SPAWN_QUALITY:
            continue
        xs = sample_candidate(cand)
        if np.count_nonzero(~np.isnan(xs)) < 2:
            continue
        items.append((Boundary(
            boundary_id=index, kind=cand["kind"], category=cand["category"],
            style=cand["style"], points=_points(xs),
            confidence=cand["quality"]), xs))
    items = _dedupe_frame(items)
    return build_topology(items)


def _dedupe_frame(items):
    """Merge same-kind candidates of one frame that describe one line."""
    items = sorted(items, key=lambda item: -item[0].confidence)
    kept = []
    for boundary, xs in items:
        duplicate = False
        for other, oxs in kept:
            if other.kind != boundary.kind:
                continue
            cost = _curve_cost(xs, oxs, min_common=2)
            if cost is not None and cost < DUPLICATE_COST:
                fill = np.isnan(oxs) & ~np.isnan(xs)
                oxs[fill] = xs[fill]
                duplicate = True
                break
        if not duplicate:
            kept.append((boundary, xs.copy()))
    return kept


# ---------------------------------------------------------------------------
# Cross-frame boundary tracks
# ---------------------------------------------------------------------------
@dataclass
class _Track:
    track_id: int
    kind: str
    xs: np.ndarray
    seen_s: np.ndarray                  # per-sample last observation time
    first_s: float
    last_s: float
    hits: int = 1
    missed: int = 0
    quality: float = 0.0
    velocity: float = 0.0               # lateral shift per AI frame
    categories: dict = field(default_factory=dict)
    styles: dict = field(default_factory=dict)

    def predicted(self) -> np.ndarray:
        return self.xs + self.velocity

    def vote(self, cand: dict) -> None:
        weight = max(0.05, float(cand["quality"]))
        self.categories[cand["category"]] = self.categories.get(cand["category"], 0.0) + weight
        self.styles[cand["style"]] = self.styles.get(cand["style"], 0.0) + weight

    def category(self) -> int:
        return int(max(self.categories, key=self.categories.get))

    def style(self) -> str:
        known = {k: v for k, v in self.styles.items() if k != "unknown"}
        if known and max(known.values()) >= 0.3 * sum(self.styles.values()):
            return max(known, key=known.get)
        return "unknown"


class LaneTopologyTracker:
    """Boundary tracks + confirmed lane topology for one video generation."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.generation: int | None = None
        self.last_media_s = -1.0
        self.tracks: list[_Track] = []
        self.next_id = 1
        self.stable: LaneTopology | None = None
        self.stable_media_s = -1.0
        self.pending_sig: tuple | None = None
        self.pending_hits = 0
        self.ai_dt = 0.1                 # smoothed media time per AI frame
        self.miss_window = MAX_MISSED_S

    # -- association -------------------------------------------------------
    def _associate(self, candidates: list[dict], media_s: float) -> None:
        samples = [sample_candidate(c) for c in candidates]
        pairs = []
        for ti, track in enumerate(self.tracks):
            pred = track.predicted()
            for ci, cand in enumerate(candidates):
                if cand["kind"] != track.kind:
                    continue
                cost = _curve_cost(pred, samples[ci])
                if cost is not None and cost < MATCH_COST:
                    pairs.append((cost, ti, ci))
        pairs.sort()
        assigned: dict[int, list[int]] = {}
        taken = set()
        for cost, ti, ci in pairs:
            if ci in taken:
                continue
            # A track may absorb several candidates only when they are
            # pieces of one line at different heights (no y overlap).
            ok = True
            for other in assigned.get(ti, []):
                overlap = (min(candidates[ci]["y_bottom"], candidates[other]["y_bottom"])
                           - max(candidates[ci]["y_top"], candidates[other]["y_top"]))
                if overlap > 0.03:
                    ok = False
                    break
            if not ok:
                continue
            assigned.setdefault(ti, []).append(ci)
            taken.add(ci)

        for ti, track in enumerate(self.tracks):
            pred = track.predicted()
            if ti not in assigned:
                track.missed += 1
                track.xs = pred
                track.velocity *= 0.6
                track.quality *= 0.85
                continue
            new = np.full_like(pred, np.nan)
            for ci in assigned[ti]:
                xs = samples[ci]
                fill = ~np.isnan(xs) & (np.isnan(new) | (candidates[ci]["quality"] > 0.5))
                new[fill] = xs[fill]
                track.vote(candidates[ci])
                track.quality = 0.6 * track.quality + 0.4 * float(candidates[ci]["quality"])
            common = ~np.isnan(new) & ~np.isnan(pred)
            if common.any():
                shift = float(np.median(new[common] - track.xs[common]))
                track.velocity = 0.5 * track.velocity + 0.5 * shift
            blended = pred.copy()
            blended[common] = 0.4 * pred[common] + 0.6 * new[common]
            only_new = ~np.isnan(new) & np.isnan(pred)
            blended[only_new] = new[only_new]
            observed = ~np.isnan(new)
            track.seen_s[observed] = media_s
            # Forget heights that have not been observed for a while.
            stale = (media_s - track.seen_s) > self.miss_window
            blended[stale] = np.nan
            track.xs = blended
            track.hits += 1
            track.missed = 0
            track.last_s = media_s

        for ci, cand in enumerate(candidates):
            if ci in taken:
                continue
            if cand["quality"] < SPAWN_QUALITY or cand["y_bottom"] < SPAWN_MIN_Y_BOTTOM:
                continue
            xs = samples[ci]
            if np.count_nonzero(~np.isnan(xs)) < 3:
                continue
            # Same-frame duplicate of an existing track: do not spawn.
            if any(t.kind == cand["kind"] and (_curve_cost(t.xs, xs, 2) or 1.0) < DUPLICATE_COST
                   for t in self.tracks):
                continue
            track = _Track(
                track_id=self.next_id, kind=cand["kind"], xs=xs.copy(),
                seen_s=np.where(np.isnan(xs), -1e9, media_s),
                first_s=media_s, last_s=media_s, quality=float(cand["quality"]))
            track.vote(cand)
            self.next_id += 1
            self.tracks.append(track)
            taken.add(ci)

        # Drop expired tracks and merge tracks that converged onto one line.
        self.tracks = [t for t in self.tracks
                       if (media_s - t.last_s <= self.miss_window
                           or t.missed <= MAX_MISSED_FRAMES)
                       and t.missed <= 4 * MAX_MISSED_FRAMES
                       and np.count_nonzero(~np.isnan(t.xs)) >= 2]
        self.tracks.sort(key=lambda t: (-t.hits, t.track_id))
        kept: list[_Track] = []
        for track in self.tracks:
            twin = next((k for k in kept if k.kind == track.kind and
                         (_curve_cost(k.xs, track.xs, 2) or 1.0) < DUPLICATE_COST), None)
            if twin is None:
                kept.append(track)
                continue
            fill = np.isnan(twin.xs) & ~np.isnan(track.xs)
            twin.xs[fill] = track.xs[fill]
            twin.seen_s = np.maximum(twin.seen_s, track.seen_s)
            for key, value in track.categories.items():
                twin.categories[key] = twin.categories.get(key, 0.0) + value
            for key, value in track.styles.items():
                twin.styles[key] = twin.styles.get(key, 0.0) + value
        self.tracks = kept

    def confirmed_boundaries(self) -> list[tuple[Boundary, np.ndarray]]:
        result = []
        for track in self.tracks:
            if track.hits < CONFIRM_FRAMES:
                continue
            result.append((Boundary(
                boundary_id=track.track_id, kind=track.kind,
                category=track.category(), style=track.style(),
                points=_points(track.xs), confidence=track.quality,
                state="observed" if track.missed == 0 else "tracked",
                hits=track.hits, missed=track.missed,
                first_seen_s=track.first_s, last_seen_s=track.last_s),
                track.xs.copy()))
        return result

    # -- public ------------------------------------------------------------
    def update(self, candidates: list[dict] | None, media_s: float,
               generation: int, frame_id: int = 0) -> LaneTopology:
        if self.generation != generation:
            self.reset()
            self.generation = generation
        if (self.last_media_s >= 0.0
                and (media_s < self.last_media_s - 0.05
                     or media_s - self.last_media_s > RESET_GAP_S)):
            self.reset()
            self.generation = generation
        if self.last_media_s >= 0.0 and media_s > self.last_media_s:
            step = media_s - self.last_media_s
            self.ai_dt = 0.7 * self.ai_dt + 0.3 * min(step, RESET_GAP_S)
        self.miss_window = max(MAX_MISSED_S, MAX_MISSED_FRAMES * self.ai_dt)
        self.last_media_s = media_s

        self._associate(list(candidates or []), media_s)
        current = build_topology(self.confirmed_boundaries())
        if current.valid and current.status == "observed":
            current.observed = True

        signature = current.signature()
        if self.stable is None or signature == self.stable.signature():
            self.pending_sig, self.pending_hits = None, 0
            if current.valid or self.stable is None:
                self.stable = current
                self.stable_media_s = media_s
            return current
        # A different lane signature: confirm it over several frames. While
        # pending, report the last confirmed topology, explicitly stale.
        if signature == self.pending_sig:
            self.pending_hits += 1
        else:
            self.pending_sig, self.pending_hits = signature, 1
        age_s = media_s - self.stable_media_s
        if self.pending_hits >= TOPOLOGY_CONFIRM or (
                age_s > self.miss_window and self.pending_hits >= 2):
            self.pending_sig, self.pending_hits = None, 0
            self.stable = current
            self.stable_media_s = media_s
            return current
        # Shallow copy: only top-level flags change on the held result.
        held = copy.copy(self.stable)
        held.observed = False
        held.stale = True
        held.age_ms = round(age_s * 1000.0, 1)
        if held.valid:
            held.status = "pending"
            held.reason = f"new signature {signature[1:]} unconfirmed"
        return held

    def last_ego_lane(self) -> int | None:
        return self.stable.ego_lane_index if self.stable else None
