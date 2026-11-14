"""Numpy/OpenCV-only RoadSeg-RK post-processing (PC and RK3566 board)."""
from __future__ import annotations

import cv2
import numpy as np

COLORS = {  # BGR
    "other_drivable": (150, 190, 120),
    "direct_drivable": (95, 205, 135),
    "ego_lane": (80, 220, 80),
    "solid": (80, 80, 235),
    "dashed": (235, 220, 80),
    "crosswalk": (210, 90, 210),
}
ALPHA = {"other_drivable": 0.18, "direct_drivable": 0.24, "ego_lane": 0.40,
         "solid": 0.55, "dashed": 0.50, "crosswalk": 0.42}


def preprocess(frame_bgr, size):
    h, w = size[1], size[0]
    ih, iw = frame_bgr.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
    left, top = (w - nw) // 2, (h - nh) // 2
    canvas = np.full((h, w, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return rgb, dict(top=top, left=left, nw=nw, nh=nh, ih=ih, iw=iw)


def _fit_side(points):
    if len(points) < 6:
        return None
    ys = np.array([p[0] for p in points], np.float32)
    xs = np.array([p[1] for p in points], np.float32)
    for _ in range(2):
        coef = np.polyfit(ys, xs, 1)
        residual = np.abs(np.polyval(coef, ys) - xs)
        keep = residual <= max(2.0, 2.5 * np.median(residual))
        if keep.sum() < 6:
            return None
        ys, xs = ys[keep], xs[keep]
    coef = np.polyfit(ys, xs, 1)
    return coef, float(np.median(np.abs(np.polyval(coef, ys) - xs)))


def ego_lane(probs, threshold=0.5, min_conf=0.35):
    """Estimate an ego-lane region from direct/alt masks plus lane geometry.

    A verified ego lane requires a left/right boundary pair.  When that pair is
    missing, the direct component around the bottom-centre seed is returned as
    a low-confidence *direct prior*, not as a verified ego lane.
    """
    direct = probs["direct"] >= threshold
    drivable = direct | (probs["alt"] >= threshold)
    lane = (probs["solid"] >= threshold) | (probs["dashed"] >= threshold)
    h, w = direct.shape
    if direct.sum() < 0.01 * h * w:
        return None, 0.0, "no_direct"
    seed = None
    for y in range(h - 1, int(h * 0.78), -1):
        xs = np.flatnonzero(direct[y])
        if len(xs):
            x = int(xs[np.argmin(np.abs(xs - w / 2))])
            if abs(x - w / 2) < w * 0.28:
                seed = (x, y)
                break
    if seed is None:
        return None, 0.0, "no_bottom_seed"
    _, labels = cv2.connectedComponents(direct.astype(np.uint8), connectivity=8)
    component = labels == labels[seed[1], seed[0]]
    if component.sum() < 0.015 * h * w:
        return None, 0.0, "small_component"
    left_pts, right_pts = [], []
    lane_support = 0
    rows = 0
    for y in range(h - 2, int(h * 0.42), -1):
        xs = np.flatnonzero(component[y])
        if len(xs) < 8:
            continue
        rows += 1
        centre = float(xs.mean())
        lanes = np.flatnonzero(lane[y])
        left = lanes[lanes < centre - 6]
        right = lanes[lanes > centre + 6]
        left_pts.append((y, int(left.max()) if len(left) else int(xs.min())))
        right_pts.append((y, int(right.min()) if len(right) else int(xs.max())))
        lane_support += int(len(left) > 0) + int(len(right) > 0)
    left_fit, right_fit = _fit_side(left_pts), _fit_side(right_pts)
    if left_fit is None or right_fit is None:
        return component, 0.25, "direct_component"
    (lc, lres), (rc, rres) = left_fit, right_fit
    support = lane_support / max(1, 2 * rows)
    residual = (lres + rres) / 2
    confidence = max(0.0, min(1.0, support * 1.6) * max(0.0, 1 - residual / 14.0))
    if confidence < min_conf:
        return component, confidence, "direct_component"
    bottom = h - 1
    top = int(h * 0.45)
    lx_b, rx_b = np.polyval(lc, bottom), np.polyval(rc, bottom)
    lx_t, rx_t = np.polyval(lc, top), np.polyval(rc, top)
    if not (0 <= lx_b < rx_b < w and 0 <= lx_t < rx_t < w and (rx_b - lx_b) > w * 0.08):
        return component, confidence, "direct_component"
    poly = np.zeros((h, w), np.uint8)
    pts = [(int(np.clip(np.polyval(lc, y), 0, w - 1)), y) for y in range(bottom, top - 1, -1)]
    pts += [(int(np.clip(np.polyval(rc, y), 0, w - 1)), y) for y in range(top, bottom + 1)]
    cv2.fillPoly(poly, [np.array(pts, np.int32)], 1)
    ego = (poly > 0) & drivable
    if ego.sum() < 0.01 * h * w:
        return component, confidence, "direct_component"
    return ego, confidence, "lane_pair"


def _filter(mask, min_area, close=3):
    m = mask.astype(np.uint8)
    if close:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out.astype(bool)


def build_lowres_overlay(probs, threshold=0.5, use_ego=True, raw=False):
    h, w = probs["direct"].shape
    drivable = (probs["direct"] >= threshold) | (probs["alt"] >= threshold)
    solid = probs["solid"] >= threshold
    dashed = probs["dashed"] >= threshold
    crosswalk = probs["crosswalk"] >= threshold
    if not raw:
        solid = _filter(solid, 18, 3)
        dashed = _filter(dashed, 12, 3)
        crosswalk = _filter(crosswalk, 80, 5)
    ego, conf, reason = (None, 0.0, "disabled")
    if use_ego:
        ego, conf, reason = ego_lane(probs, threshold)
    color = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), np.float32)

    def paint(mask, key):
        color[mask] = COLORS[key]
        alpha[mask] = ALPHA[key]

    verified = ego is not None and reason == "lane_pair"
    if verified:
        paint(drivable, "other_drivable")
        paint(ego, "ego_lane")
    else:
        # Direct/alternative are still shown separately, but the direct region
        # is not promoted to a verified ego lane.
        paint(probs["direct"] >= threshold, "direct_drivable")
        paint(probs["alt"] >= threshold, "other_drivable")
    paint(dashed, "dashed")
    paint(solid, "solid")
    paint(crosswalk, "crosswalk")
    return color, alpha, dict(ego_available=bool(verified), ego_confidence=round(float(conf), 3),
                              ego_reason=reason, ego_pixels=int(ego.sum()) if ego is not None else 0,
                              drivable_pixels=int(drivable.sum()))


def compose(frame_bgr, probs, threshold=0.5, use_ego=True, raw=False):
    color, alpha, meta = build_lowres_overlay(probs, threshold, use_ego, raw)
    ih, iw = frame_bgr.shape[:2]
    color_n = cv2.resize(color, (iw, ih), interpolation=cv2.INTER_LINEAR)
    alpha_n = cv2.resize(alpha, (iw, ih), interpolation=cv2.INTER_LINEAR)
    out = cv2.blendLinear(frame_bgr.astype(np.float32), color_n, 1.0 - alpha_n, alpha_n)
    return np.clip(out, 0, 255).astype(np.uint8), meta
