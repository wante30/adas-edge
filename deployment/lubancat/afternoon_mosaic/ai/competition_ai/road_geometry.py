"""Low-level mask -> road geometry helpers.

Coordinates are normalized to the source image: (0,0) top-left, (1,1)
bottom-right. Letterbox padding and scale are removed explicitly so Windows
crop frames and RKNN full frames produce the same normalized geometry.

Only NumPy / OpenCV / stdlib are allowed here.
"""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

LANE_BACKGROUND = 8
LANE_CROSSWALK = 0
LANE_FOREGROUND = (1, 2, 3, 4, 5, 6, 7)
LANE_NAMES = {
    0: "crosswalk",
    1: "double other",
    2: "double white",
    3: "double yellow",
    4: "road curb",
    5: "single other",
    6: "single white",
    7: "single yellow",
    8: "background",
}
STYLE_SOLID = 0
STYLE_DASHED = 1
STYLE_BACKGROUND = 2


def model_to_source(u, v, mask_shape, letterbox, source_width, source_height):
    """Map mask pixel centers to normalized source coordinates."""
    mask_h, mask_w = mask_shape
    scale = float(letterbox.get("scale", 0.0) or 0.0)
    if scale <= 1e-6:
        scale = 1.0
    pad_x = float(letterbox.get("pad_x", 0.0))
    pad_y = float(letterbox.get("pad_y", 0.0))
    model_w = 512.0
    model_h = 288.0
    u_arr = np.asarray(u, dtype=np.float32)
    v_arr = np.asarray(v, dtype=np.float32)
    model_x = (u_arr + 0.5) * (model_w / mask_w)
    model_y = (v_arr + 0.5) * (model_h / mask_h)
    x_norm = ((model_x - pad_x) / scale) / max(float(source_width), 1.0)
    y_norm = ((model_y - pad_y) / scale) / max(float(source_height), 1.0)
    return x_norm, y_norm


def _runs(row: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([0], row.astype(np.int8), [0]))
    changes = np.flatnonzero(np.diff(padded))
    return [(int(changes[i]), int(changes[i + 1]))
            for i in range(0, len(changes), 2)]


def _pick_component(binary: np.ndarray, prefer_direct: np.ndarray | None = None):
    """Pick the drivable component connected to the ego/bottom-center area."""
    binary = (binary > 0).astype(np.uint8)
    if not binary.any():
        return None
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    height, width = binary.shape
    bottom = labels[int(height * 0.72):, :]
    best = None
    best_score = -1.0
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 24:
            continue
        bottom_overlap = int(np.count_nonzero(bottom == index))
        center_x = float(centroids[index][0])
        score = (area + 4.0 * bottom_overlap
                 - 0.5 * abs(center_x - width * 0.5))
        if prefer_direct is not None and prefer_direct.any():
            direct_overlap = int(np.count_nonzero(
                (labels == index) & (prefer_direct > 0)))
            score += 0.6 * direct_overlap
        if score > best_score:
            best_score = score
            best = index
    if best is None:
        return None
    return (labels == best).astype(np.uint8)


def drivable_surface(drivable: np.ndarray):
    """Return (boundary component, ego component, used_direct).

    The boundary component uses direct+alternative because the road can be
    split by a vehicle or classified as alternative. The ego component uses
    direct only when available: that is the surface the vehicle may occupy.
    """
    direct = (drivable == 0).astype(np.uint8)
    alternative = (drivable == 1).astype(np.uint8)
    combined = ((direct | alternative) > 0).astype(np.uint8)
    # Close only tiny holes; do not bridge vehicles or barriers. The whole
    # combined mask is the boundary candidate: scanline_bounds() bridges a
    # short central gap but rejects disconnected roadside blobs.
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    boundary = combined
    ego = _pick_component(direct)
    used_direct = ego is not None
    if ego is None:
        ego = boundary
    return boundary, ego, used_direct


def scanline_bounds(component: np.ndarray, levels: Iterable[float],
                    ego_u: float, min_run: int = 4,
                    bridged_gap: float = 0.0,
                    bridge_center: tuple[float, float] = (0.28, 0.72)):
    """Sample left/right road boundaries at normalized image y levels.

    bridged_gap is a normalized width. A gap is only bridged when both sides
    have a real run and the gap midpoint is near the ego; this recovers a road
    split by a taxi without merging a distant sidewalk.
    """
    height, width = component.shape
    result = []
    for level in levels:
        row = int(round(float(level) * (height - 1)))
        row = int(np.clip(row, 0, height - 1))
        runs = _runs(component[row])
        runs = [(a, b) for a, b in runs if b - a >= min_run]
        if not runs:
            continue
        index = min(range(len(runs)),
                    key=lambda i: abs((runs[i][0] + runs[i][1]) * 0.5 - ego_u))
        a, b = runs[index]
        if bridged_gap > 0.0:
            gap_px = bridged_gap * width
            for direction in (1, -1):
                cursor = index + direction
                while 0 <= cursor < len(runs):
                    na, nb = runs[cursor]
                    if direction > 0:
                        gap = na - b
                        midpoint = (b + na) * 0.5 / width
                    else:
                        gap = a - nb
                        midpoint = (a + nb) * 0.5 / width
                    if gap <= 2.0 or (
                            gap <= gap_px
                            and bridge_center[0] <= midpoint <= bridge_center[1]
                            and (nb - na) >= min_run):
                        if direction > 0:
                            b = nb
                        else:
                            a = na
                        cursor += direction
                        continue
                    break
        result.append({
            "y": float(level),
            "left": float(a) / width,
            "right": float(b) / width,
            "center": float(a + b) * 0.5 / width,
            "width": float(b - a) / width,
        })

    # Interpolate up to two consecutive missing scan rows. This bridges a
    # short vehicle occlusion but never extrapolates past the observed road.
    levels = [float(value) for value in levels]
    by_level = {round(item["y"], 5): item for item in result}
    if len(by_level) >= 2:
        existing = sorted(by_level)
        for index, level in enumerate(levels):
            key = round(level, 5)
            if key in by_level:
                continue
            before = next((levels[j] for j in range(index - 1, max(-1, index - 3), -1)
                           if round(levels[j], 5) in by_level), None)
            after = next((levels[j] for j in range(index + 1, min(len(levels), index + 3))
                          if round(levels[j], 5) in by_level), None)
            if before is None or after is None:
                continue
            a = by_level[round(before, 5)]
            b = by_level[round(after, 5)]
            span = max(1e-6, after - before)
            t = (level - before) / span
            by_level[key] = {
                "y": level,
                "left": a["left"] + (b["left"] - a["left"]) * t,
                "right": a["right"] + (b["right"] - a["right"]) * t,
                "center": a["center"] + (b["center"] - a["center"]) * t,
                "width": a["width"] + (b["width"] - a["width"]) * t,
            }
        result = sorted(by_level.values(), key=lambda item: item["y"], reverse=True)
    return result


def _smooth(values: list[float], window: int = 3) -> list[float]:
    if len(values) <= 2:
        return values
    out = list(values)
    half = window // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out[i] = float(sum(values[lo:hi]) / (hi - lo))
    return out


def smooth_bounds(samples: list[dict]) -> list[dict]:
    if not samples:
        return samples
    samples = sorted(samples, key=lambda item: item["y"], reverse=True)
    lefts = _smooth([item["left"] for item in samples])
    rights = _smooth([item["right"] for item in samples])
    centers = _smooth([item["center"] for item in samples])
    for item, left, right, center in zip(samples, lefts, rights, centers):
        item["left"], item["right"], item["center"] = left, right, center
        item["width"] = max(0.0, right - left)
    return samples


def lane_markings(lane_cat: np.ndarray, lane_style: np.ndarray,
                  mask_shape, letterbox, source_width: int, source_height: int,
                  max_total: int = 12):
    height, width = lane_cat.shape
    found = []
    for category in LANE_FOREGROUND:
        binary = (lane_cat == category).astype(np.uint8)
        if not binary.any():
            continue
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        per_class = []
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            if area < 10:
                continue
            ys, xs = np.where(labels == index)
            if len(ys) < 3:
                continue
            rows = np.linspace(int(ys.min()), int(ys.max()), 6).astype(int)
            points = []
            for row in rows:
                columns = xs[ys == row]
                if columns.size == 0:
                    continue
                u = float(np.mean(columns))
                x_norm, y_norm = model_to_source(
                    u, row, mask_shape, letterbox, source_width, source_height)
                if 0.0 <= float(x_norm) <= 1.0 and 0.0 <= float(y_norm) <= 1.0:
                    points.append((float(x_norm), float(y_norm)))
            if len(points) < 2:
                continue
            style_values = lane_style[labels == index]
            solid = int(np.count_nonzero(style_values == STYLE_SOLID))
            dashed = int(np.count_nonzero(style_values == STYLE_DASHED))
            foreground = solid + dashed
            if foreground >= 3 and solid >= 0.6 * foreground:
                style = "solid"
            elif foreground >= 3 and dashed >= 0.6 * foreground:
                style = "dashed"
            else:
                style = "unknown"
            per_class.append({
                "category": int(category),
                "category_name": LANE_NAMES.get(category, "unknown"),
                "style": style,
                "double": category in (1, 2, 3),
                "confidence": round(min(1.0, area / 90.0), 3),
                "points": points,
                "area": area,
            })
        per_class.sort(key=lambda item: item["area"], reverse=True)
        found.extend(per_class[:3])
    found.sort(key=lambda item: item["area"], reverse=True)
    for item in found:
        item.pop("area", None)
    return found[:max_total]


# ---------------------------------------------------------------------------
# Longitudinal boundary candidates
#
# lane_cat is the only source of marking geometry. drivable is NOT used to
# clip candidates: drivable is "where ego may drive", not the extent of the
# carriageway, and lines of neighbouring lanes routinely lie outside it.
#
# Pipeline (mask pixel space, 256x144 is isotropic with the 16:9 source):
#   1. per-row runs of marking pixels (paint and curb kept separate);
#   2. runs linked row to row into strands; a strand ends at any branch or
#      merge so two converging lines never share a strand;
#   3. horizontal blobs (stop lines, crosswalk stripes misclassified as
#      paint) are rejected by run width vs. strand height;
#   4. strands linked across vertical gaps when one straight/quadratic model
#      explains both (residual), their directions agree and the gap is
#      allowed for the observed line style (dashes may skip long gaps);
#   5. each chain becomes one BoundaryCandidate with category, style and a
#      quality score. Short isolated noise is dropped; nothing is invented.
# ---------------------------------------------------------------------------
PAINT_CLASSES = (1, 2, 3, 5, 6, 7)
CURB_CLASSES = (4,)
CANDIDATE_MIN_Y = 0.50          # rows above this are too close to horizon
GAP_ROWS_DASHED = 34            # max vertical gap (mask rows) inside a dash chain
GAP_ROWS_SOLID = 14             # max gap for solid / unknown lines
GAP_ROWS_LONG = 30              # max gap when both pieces are long (occlusion)


def _row_runs(row: np.ndarray) -> list[tuple[int, int]]:
    if not row.any():
        return []
    return _runs(row)


def _strands(binary: np.ndarray, v_min: int) -> list[dict]:
    """Link per-row runs into non-branching strands, scanning bottom-up."""
    height = binary.shape[0]
    finished: list[dict] = []
    active: list[dict] = []
    for v in range(height - 1, v_min - 1, -1):
        runs = _row_runs(binary[v])
        # Candidates: active strands whose last run is at most 2 rows below.
        live = [item for item in active if item["last_v"] - v <= 2]
        for item in active:
            if item not in live:
                finished.append(item)
        matches_run: dict[int, list[int]] = {i: [] for i in range(len(runs))}
        matches_str: dict[int, list[int]] = {i: [] for i in range(len(live))}
        for ri, (a, b) in enumerate(runs):
            for si, item in enumerate(live):
                pa, pb = item["last_run"]
                if a <= pb + 1 and b >= pa - 1:
                    matches_run[ri].append(si)
                    matches_str[si].append(ri)
        next_active = []
        used = set()
        for ri, (a, b) in enumerate(runs):
            linked = matches_run[ri]
            if len(linked) == 1 and len(matches_str[linked[0]]) == 1:
                item = live[linked[0]]
                item["rows"].append((v, a, b))
                item["last_v"] = v
                item["last_run"] = (a, b)
                next_active.append(item)
                used.add(linked[0])
            else:
                next_active.append({"rows": [(v, a, b)], "last_v": v,
                                    "last_run": (a, b)})
        for si, item in enumerate(live):
            if si not in used:
                finished.append(item)
        active = next_active
    finished.extend(active)
    return finished


def _fit(us: np.ndarray, vs: np.ndarray, weights: np.ndarray | None = None,
         degree: int = 1):
    """Least-squares u(v); returns (coef, rms)."""
    if vs.size < 2 or float(np.ptp(vs)) < 1e-6:
        return None, float("inf")
    degree = min(degree, max(1, len(np.unique(vs)) - 1))
    coef = np.polyfit(vs, us, degree, w=weights)
    rms = float(np.sqrt(np.mean((np.polyval(coef, vs) - us) ** 2)))
    return coef, rms


def _strand_features(strand: dict, lane_cat: np.ndarray,
                     lane_style: np.ndarray) -> dict | None:
    rows = strand["rows"]
    vs = np.array([r[0] for r in rows], dtype=np.float64)
    a = np.array([r[1] for r in rows], dtype=np.float64)
    b = np.array([r[2] for r in rows], dtype=np.float64)
    us = (a + b - 1) * 0.5
    widths = b - a
    pixels = int(widths.sum())
    height = int(vs.max() - vs.min() + 1)
    mean_width = float(widths.mean())
    # Horizontal blob: wide runs over very few rows (stop line, crosswalk).
    if height <= 3 and mean_width >= 7:
        return None
    if mean_width > 4.0 * height and mean_width >= 10:
        return None
    coef, rms = _fit(us, vs)
    slope = float(coef[0]) if coef is not None else 0.0
    # For a thin line run width ~ thickness*|du/dv|. When centres barely move
    # but runs are wide the strand is horizontal, not vertical.
    if coef is not None and abs(slope) < 1.0 and mean_width > 6 and height < 8:
        return None
    if abs(slope) > 9.0:            # < ~6 degrees from horizontal
        return None
    cats = np.concatenate([lane_cat[int(v), int(x0):int(x1)]
                           for v, x0, x1 in rows])
    styles = np.concatenate([lane_style[int(v), int(x0):int(x1)]
                             for v, x0, x1 in rows])
    return {
        "vs": vs, "us": us, "w": widths.astype(np.float64),
        "v_top": float(vs.min()), "v_bot": float(vs.max()),
        "height": height, "pixels": pixels, "slope": slope,
        "coef": coef, "rms": rms,
        "cats": np.bincount(cats, minlength=9),
        "styles": np.bincount(styles, minlength=3),
    }


def _chain_points(chain: list[dict]):
    vs = np.concatenate([item["vs"] for item in chain])
    us = np.concatenate([item["us"] for item in chain])
    ws = np.concatenate([np.sqrt(np.maximum(item["w"], 1.0)) for item in chain])
    return us, vs, ws


def _dash_like(item: dict) -> bool:
    styles = item["styles"]
    return styles[STYLE_DASHED] > styles[STYLE_SOLID]


def _link_cost(lower: list[dict], upper: list[dict]) -> float | None:
    """Cost of joining chain `upper` (smaller v) above chain `lower`."""
    # Chains are ordered bottom -> top: the joint is between the top piece
    # of the lower chain and the bottom piece of the upper chain.
    low, up = lower[-1], upper[0]
    gap = low["v_top"] - up["v_bot"]
    if gap < -2:                            # overlapping in v: not a continuation
        return None
    long_pieces = (sum(i["height"] for i in lower) >= 6
                   and sum(i["height"] for i in upper) >= 6)
    dashed = _dash_like(low) or _dash_like(up)
    max_gap = GAP_ROWS_SOLID
    if dashed:
        max_gap = GAP_ROWS_DASHED
    if long_pieces:
        max_gap = max(max_gap, GAP_ROWS_LONG)
    # Gaps shrink toward the horizon; scale the allowance with depth.
    depth = max(0.25, (low["v_top"] - 0.50 * 143) / (0.5 * 143))
    if gap > max_gap * min(1.0, depth + 0.15):
        return None
    us, vs, ws = _chain_points(lower + upper)
    degree = 2 if float(np.ptp(vs)) >= 40 else 1
    coef, rms = _fit(us, vs, ws, degree)
    if coef is None:
        return None
    if rms > 1.6 + 0.02 * max(0.0, gap):
        return None
    # Direction agreement when both sides are long enough to have one.
    lo_us, lo_vs, lo_ws = _chain_points(lower)
    up_us, up_vs, up_ws = _chain_points(upper)
    if float(np.ptp(lo_vs)) >= 4 and float(np.ptp(up_vs)) >= 4:
        lo_c, _ = _fit(lo_us, lo_vs, lo_ws)
        up_c, _ = _fit(up_us, up_vs, up_ws)
        angle = abs(np.degrees(np.arctan(lo_c[0]) - np.arctan(up_c[0])))
        # A short dash has an unreliable own direction.
        short = min(float(np.ptp(lo_vs)), float(np.ptp(up_vs))) + 1.0
        if angle > 14.0 + 60.0 / short:
            return None
    # Prediction error of the lower chain's line at the upper chain's foot.
    lo_c, _ = _fit(lo_us, lo_vs, lo_ws) if float(np.ptp(lo_vs)) >= 2 else (None, 0)
    if lo_c is not None:
        predicted = float(np.polyval(lo_c, up["v_bot"]))
        actual = float(up["us"][np.argmax(up["vs"])])
        # Extrapolation error grows with the lever from the lower chain's
        # centre of mass to the joint.
        lever = float(np.mean(lo_vs)) - up["v_bot"]
        if abs(predicted - actual) > 2.5 + 0.12 * max(0.0, gap) + 0.15 * lever:
            return None
    return rms + 0.03 * max(0.0, gap)


def _link_pieces(pieces: list[dict]) -> list[list[dict]]:
    """Greedy gap linking. Each chain has one free top and one free bottom.

    Pairs are pre-filtered with each piece's own line (cheap), then the
    full _link_cost() validates the joined chain. Lowest cost links first;
    a link is re-validated against the current chains before it is applied,
    so a chain can never absorb two branches.
    """
    count = len(pieces)
    if count < 2:
        return [[p] for p in pieces]
    order = sorted(range(count), key=lambda i: -pieces[i]["v_bot"])
    pieces = [pieces[i] for i in order]
    pairs = []
    for i, low in enumerate(pieces):
        for j, up in enumerate(pieces):
            if i == j or up["v_bot"] >= low["v_bot"]:
                continue
            gap = low["v_top"] - up["v_bot"]
            if gap < -2 or gap > GAP_ROWS_DASHED:
                continue
            if low["coef"] is not None and low["height"] >= 3:
                predicted = float(np.polyval(low["coef"], up["v_bot"]))
            else:
                predicted = float(low["us"][np.argmin(low["vs"])])
            actual = float(up["us"][np.argmax(up["vs"])])
            if abs(predicted - actual) > 4.0 + 0.2 * max(0.0, gap):
                continue
            pairs.append((abs(predicted - actual) + 0.05 * max(0.0, gap), i, j))
    pairs.sort()
    chain_of = list(range(count))
    chains: dict[int, list[dict]] = {i: [pieces[i]] for i in range(count)}
    top_of = {i: i for i in range(count)}       # chain id -> top piece index
    bot_of = {i: i for i in range(count)}       # chain id -> bottom piece index
    for _, i, j in pairs:
        ci, cj = chain_of[i], chain_of[j]
        if ci == cj or top_of[ci] != i or bot_of[cj] != j:
            continue
        if _link_cost(chains[ci], chains[cj]) is None:
            continue
        chains[ci] = chains[ci] + chains[cj]
        top_of[ci] = top_of[cj]
        for piece_index in range(count):
            if chain_of[piece_index] == cj:
                chain_of[piece_index] = ci
        del chains[cj], top_of[cj], bot_of[cj]
    return list(chains.values())


def boundary_candidates(lane_cat: np.ndarray, lane_style: np.ndarray,
                        mask_shape, letterbox, source_width: int,
                        source_height: int) -> list[dict]:
    """Per-frame longitudinal boundary candidates in normalized source coords.

    Returns dicts: kind ('lane'|'curb'), category, style
    ('solid'|'dashed'|'unknown'), coef (x(y) polynomial, normalized), y_top,
    y_bottom, points [(x,y)], pieces, pixels, rms_px, quality.
    """
    height, width = lane_cat.shape
    v_min = int(round(CANDIDATE_MIN_Y * (height - 1)))
    # Pixel -> normalized mapping is affine; derive its coefficients once.
    x0, y0 = model_to_source(0.0, 0.0, mask_shape, letterbox, source_width, source_height)
    x1, y1 = model_to_source(1.0, 1.0, mask_shape, letterbox, source_width, source_height)
    ax, bx = float(x1 - x0), float(x0)
    ay, by = float(y1 - y0), float(y0)
    result = []
    for kind, classes in (("lane", PAINT_CLASSES), ("curb", CURB_CLASSES)):
        binary = np.isin(lane_cat, classes)
        if not binary.any():
            continue
        pieces = [f for f in (_strand_features(s, lane_cat, lane_style)
                              for s in _strands(binary, v_min)) if f is not None]
        chains = _link_pieces(pieces)
        for chain in chains:
            us, vs, ws = _chain_points(chain)
            span = float(vs.max() - vs.min() + 1)
            pixels = sum(item["pixels"] for item in chain)
            if span < 4 or pixels < 8:
                continue
            coef_px, rms = _fit(us, vs, ws, 1)
            if coef_px is None:
                continue
            degree = 1
            if span >= 40:
                quad, quad_rms = _fit(us, vs, ws, 2)
                # Curvature limit: lateral bend over the span <= ~span/6 px.
                bend = abs(float(quad[0])) * (span / 2.0) ** 2
                if quad_rms < 0.7 * rms and bend <= span / 6.0:
                    coef_px, rms, degree = quad, quad_rms, 2
            # Convert u(v) to x(y): u = (x-bx)/ax, v = (y-by)/ay.
            poly_v = np.poly1d(coef_px)
            ys_px = np.unique(vs)
            xs_norm = ax * poly_v(ys_px) + bx
            ys_norm = ay * ys_px + by
            coef = np.polyfit(ys_norm, xs_norm, min(degree, max(1, len(ys_norm) - 1)))
            cats = sum(item["cats"] for item in chain)
            styles = sum(item["styles"] for item in chain)
            category = int(np.argmax(cats[1:8]) + 1)
            fg = int(styles[STYLE_SOLID] + styles[STYLE_DASHED])
            if len(chain) >= 2 and _dash_like(chain[0]) | _dash_like(chain[-1]):
                style = "dashed"
            elif fg >= 4 and styles[STYLE_SOLID] >= 0.6 * fg:
                style = "solid"
            elif fg >= 4 and styles[STYLE_DASHED] >= 0.6 * fg:
                style = "dashed"
            else:
                style = "unknown"
            observed_rows = int(len(np.unique(vs)))
            coverage = observed_rows / span
            y_span = span * ay
            quality = float(np.clip(
                min(1.0, y_span / 0.18) * (0.6 + 0.4 * min(1.0, pixels / 60.0))
                * (1.0 if rms <= 1.0 else max(0.3, 1.0 / rms)), 0.0, 1.0))
            step = max(1, len(ys_px) // 12)
            points = [(float(x), float(y)) for x, y in
                      zip(xs_norm[::-1][::step], ys_norm[::-1][::step])]
            result.append({
                "kind": kind,
                "category": category,
                "category_name": LANE_NAMES.get(category, "unknown"),
                "style": style,
                "coef": [float(c) for c in coef],
                "y_top": float(ys_norm.min()),
                "y_bottom": float(ys_norm.max()),
                "points": points,
                "pieces": len(chain),
                "pixels": int(pixels),
                "coverage": round(float(coverage), 3),
                "rms_px": round(float(rms), 3),
                "quality": round(quality, 3),
            })
    _vanishing_point_check(result)
    return result


VP_TOLERANCE = 0.10        # normalized x error of a tangent at the VP height
SHORT_PIECE_Y = 0.06       # candidates shorter than this take the VP direction


def _tangent(candidate: dict):
    coef = np.asarray(candidate["coef"], dtype=np.float64)
    y = float(candidate["y_bottom"])
    slope = float(np.polyval(np.polyder(coef), y)) if coef.size > 1 else 0.0
    return float(np.polyval(coef, y)), y, slope


def _vanishing_point_check(candidates: list[dict]) -> None:
    """Lane boundaries of one road converge in perspective.

    The vanishing point is the robust median of pairwise intersections of
    strong candidate tangents. A candidate whose tangent misses it is not a
    longitudinal boundary of this road (arrow, stop line, turning mark,
    occluder edge); its quality is cut so it cannot start a track. With too
    few strong candidates nothing is changed.
    """
    strong = [c for c in candidates
              if c["quality"] >= 0.5 and c["y_bottom"] - c["y_top"] >= 0.08]
    points = []
    for i in range(len(strong)):
        xi, yi, si = _tangent(strong[i])
        for j in range(i + 1, len(strong)):
            xj, yj, sj = _tangent(strong[j])
            if abs(si - sj) < 0.25:
                continue
            # x = xi + si*(y-yi) = xj + sj*(y-yj)
            y = (xj - xi + si * yi - sj * yj) / (si - sj)
            x = xi + si * (y - yi)
            if 0.2 <= y <= 0.7 and -0.2 <= x <= 1.2:
                points.append((x, y))
    for c in candidates:
        c["vp_error"] = None
        c["direction"] = "fit"
    if len(points) < 1:
        return
    vx = float(np.median([p[0] for p in points]))
    vy = float(np.median([p[1] for p in points]))
    for c in candidates:
        # A short piece (single far dash) has no reliable own direction; in
        # perspective a longitudinal line passes through the vanishing
        # point, so take the line through its centroid and the VP.
        if c["y_bottom"] - c["y_top"] < SHORT_PIECE_Y and c["kind"] == "lane":
            xs = np.array([p[0] for p in c["points"]])
            ys = np.array([p[1] for p in c["points"]])
            cx, cy = float(xs.mean()), float(ys.mean())
            if cy - vy > 0.03:
                slope = (cx - vx) / (cy - vy)
                c["coef"] = [slope, cx - slope * cy]
                c["direction"] = "vanishing_point"
                c["vp_error"] = 0.0
                c["vp"] = (round(vx, 3), round(vy, 3))
                continue
        x, y, slope = _tangent(c)
        error = abs(x + slope * (vy - y) - vx)
        c["vp_error"] = round(error, 3)
        c["vp"] = (round(vx, 3), round(vy, 3))
        if error > VP_TOLERANCE + 0.5 * max(0.0, 0.70 - c["y_bottom"]):
            c["quality"] = round(c["quality"] * 0.3, 3)


def crosswalk_polygons(lane_cat: np.ndarray, mask_shape, letterbox,
                       source_width: int, source_height: int, max_total: int = 3):
    height, width = lane_cat.shape
    binary = (lane_cat == LANE_CROSSWALK).astype(np.uint8)
    if int(binary.sum()) < 12:
        return []
    # Sparse model stripes need horizontal closing before they form one
    # crosswalk. The wider kernel connects stripes within the same crossing;
    # groups that are still separate are merged by bbox proximity below.
    merged = cv2.dilate(binary, np.ones((3, 9), np.uint8), iterations=1)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(merged, 8)
    candidates = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        x, y, w, h = (int(stats[index, cv2.CC_STAT_LEFT]),
                      int(stats[index, cv2.CC_STAT_TOP]),
                      int(stats[index, cv2.CC_STAT_WIDTH]),
                      int(stats[index, cv2.CC_STAT_HEIGHT]))
        if area < 20 or w < max(4, int(width * 0.035)) or h < 2:
            continue
        if y + h * 0.5 < height * 0.15:
            continue
        candidates.append({
            "x0": x, "y0": y, "x1": x + w, "y1": y + h,
            "area": area,
        })

    if not candidates:
        return []

    # Merge stripe groups that share the same crossing height. A taxi in the
    # middle can split the visible stripes into two groups; for display we
    # bridge that gap, but keep observed/stale semantics in RoadScene.
    parent = list(range(len(candidates)))

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            overlap = max(0, min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]))
            min_height = max(1, min(a["y1"] - a["y0"], b["y1"] - b["y0"]))
            gap = max(a["x0"], b["x0"]) - min(a["x1"], b["x1"])
            if overlap >= 0.40 * min_height and gap <= width * 0.12:
                parent[find(i)] = find(j)

    groups: dict[int, list[dict]] = {}
    for index, item in enumerate(candidates):
        groups.setdefault(find(index), []).append(item)

    result = []
    for members in groups.values():
        x0 = min(item["x0"] for item in members)
        y0 = min(item["y0"] for item in members)
        x1 = max(item["x1"] for item in members)
        y1 = max(item["y1"] for item in members)
        area = sum(item["area"] for item in members)
        if (x1 - x0) < width * 0.045 or (y1 - y0) < 2:
            continue
        nx0, ny0 = model_to_source(float(x0), float(y0), mask_shape,
                                   letterbox, source_width, source_height)
        nx1, ny1 = model_to_source(float(x1), float(y1), mask_shape,
                                   letterbox, source_width, source_height)
        points = [(float(nx0), float(ny0)), (float(nx1), float(ny0)),
                  (float(nx1), float(ny1)), (float(nx0), float(ny1))]
        confidence = min(1.0, area / 180.0)
        if len(members) > 1:
            confidence *= 0.85
        result.append({
            "polygon": points,
            "bbox": (float(nx0), float(ny0), float(nx1), float(ny1)),
            "center_y": float((ny0 + ny1) * 0.5),
            "confidence": round(confidence, 3),
            "area": area,
        })
    result.sort(key=lambda item: item["area"], reverse=True)
    for item in result:
        item.pop("area", None)
    return result[:max_total]
