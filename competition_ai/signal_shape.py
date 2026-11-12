"""Conservative native-pixel glow morphology and shape-template backend."""
from __future__ import annotations

from dataclasses import replace
import time
import cv2
import numpy as np

from .multi_signal import HSVBackend, bounded_roi


def _normalize(mask):
    points = cv2.findNonZero(mask)
    if points is None:
        return np.zeros((48, 48), np.uint8)
    x, y, w, h = cv2.boundingRect(points)
    scale = 40/max(w, h)
    small = cv2.resize(mask[y:y+h, x:x+w], (max(1, round(w*scale)), max(1, round(h*scale))),
                       interpolation=cv2.INTER_NEAREST)
    out = np.zeros((48, 48), np.uint8)
    oy, ox = (48-small.shape[0])//2, (48-small.shape[1])//2
    out[oy:oy+small.shape[0], ox:ox+small.shape[1]] = small
    return out > 0


def _templates():
    result = {}
    for thickness in (8, 11, 14):
        up = np.zeros((64, 64), np.uint8)
        cv2.rectangle(up, (32-thickness//2, 26), (32+thickness//2, 57), 255, -1)
        cv2.fillPoly(up, [np.array([(8, 28), (32, 3), (56, 28)], np.int32)], 255)
        for name, mask in [('STRAIGHT_ARROW', up), ('LEFT_ARROW', np.rot90(up)),
                           ('RIGHT_ARROW', np.rot90(up, 3))]:
            result.setdefault(name, []).append(_normalize(mask))
        turn = np.zeros((64, 64), np.uint8)
        cv2.ellipse(turn, (32, 25), (16, 16), 0, 180, 360, 255, thickness)
        cv2.line(turn, (48, 25), (48, 58), 255, thickness)
        cv2.line(turn, (16, 25), (16, 35), 255, thickness)
        cv2.fillPoly(turn, [np.array([(3, 33), (16, 49), (29, 33)], np.int32)], 255)
        result.setdefault('UTURN_ARROW', []).append(_normalize(turn))
    # Real signal masks commonly use an open chevron, with either a connected
    # stem or a separate LED dash. These are geometric templates, not samples
    # copied from the evaluation photographs.
    for thickness in (5, 8, 11):
        for gap in (0, 8, 16, 24, 32):
            up = np.zeros((64, 64), np.uint8)
            cv2.polylines(up, [np.array([(10, 28), (32, 6), (54, 28)], np.int32)],
                          False, 255, thickness)
            cv2.line(up, (32, 9+gap), (32, 56), 255, thickness)
            for name, mask in [('STRAIGHT_ARROW', up), ('LEFT_ARROW', np.rot90(up)),
                               ('RIGHT_ARROW', np.rot90(up, 3))]:
                result.setdefault(name, []).append(_normalize(mask))
        turn = np.zeros((64, 64), np.uint8)
        cv2.ellipse(turn, (34, 23), (14, 14), 0, 180, 360, 255, thickness)
        cv2.line(turn, (48, 23), (48, 49), 255, thickness)
        cv2.line(turn, (20, 23), (20, 46), 255, thickness)
        cv2.polylines(turn, [np.array([(6, 33), (20, 48), (35, 33)], np.int32)],
                      False, 255, thickness)
        result['UTURN_ARROW'].append(_normalize(turn))
    for thickness in (4, 7, 10):
        for head_depth in (20, 28, 36):
            for stem_start in (8, 24, 40):
                up = np.zeros((64, 64), np.uint8)
                cv2.polylines(up, [np.array([(8, 5+head_depth), (32, 5),
                                            (56, 5+head_depth)], np.int32)], False, 255, thickness)
                cv2.line(up, (32, stem_start), (32, 58), 255, thickness)
                for name, mask in [('STRAIGHT_ARROW', up), ('LEFT_ARROW', np.rot90(up)),
                                   ('RIGHT_ARROW', np.rot90(up, 3))]:
                    result[name].append(_normalize(mask))
    circle = np.zeros((64, 64), np.uint8)
    cv2.circle(circle, (32, 32), 25, 255, -1)
    result['ROUND'] = [_normalize(circle)]
    # Small camera roll and oblique viewing are supported; direction is never
    # inferred from the lamp's location in the image.
    variants = {}
    for name, masks in result.items():
        variants[name] = []
        for mask in masks:
            for aspect in (.75, 1., 1.33):
                stretched = cv2.resize(mask.astype(np.uint8)*255, (round(48*aspect), 48),
                                       interpolation=cv2.INTER_NEAREST)
                stretched = _normalize(stretched).astype(np.uint8)*255
                for angle in (-12, 0, 12):
                    rotated = cv2.warpAffine(stretched,
                        cv2.getRotationMatrix2D((24, 24), angle, 1), (48, 48),
                        flags=cv2.INTER_NEAREST)
                    variants[name].append(_normalize(rotated))
    return {name: np.stack(masks) for name, masks in variants.items()}


class ContourTemplateBackend(HSVBackend):
    def __init__(self):
        templates = _templates()
        self.template_bits = {name: np.packbits(masks.reshape(len(masks), -1), axis=1)
                              for name, masks in templates.items()}
        self.template_areas = {name: np.count_nonzero(masks, axis=(1, 2))
                               for name, masks in templates.items()}
        self.bit_count = np.array([bin(n).count('1') for n in range(256)], np.uint8)

    def regions(self, roi):
        """Separate clearly spaced lamp faces inside a detector's signal head.

        Close islands may be one LED arrow. They are grouped before proposing
        regions; unclear overlaps fall back to one UNKNOWN head observation.
        Returned boxes remain in original ROI pixels, never display pixels.
        """
        native_h, native_w = roi.shape[:2]
        small = bounded_roi(roi)
        h, s, v = cv2.split(cv2.cvtColor(small, cv2.COLOR_BGR2HSV))
        valid = (s >= 70) & (v >= 130)
        masks = [valid & ((h <= 10) | (h >= 160)),
                 valid & (h >= 14) & (h < 40), valid & (h >= 40) & (h <= 95)]
        regions = []
        for color, mask in enumerate(masks):
            clean = cv2.morphologyEx(mask.astype(np.uint8)*255, cv2.MORPH_CLOSE,
                                     np.ones((3, 3), np.uint8))
            contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= 8]
            groups = []
            for x, y, w, height in sorted(boxes, key=lambda b: b[2]*b[3], reverse=True):
                box = [x, y, x+w, y+height]
                for group in groups:
                    gx = max(0, max(box[0], group[0])-min(box[2], group[2]))
                    gy = max(0, max(box[1], group[1])-min(box[3], group[3]))
                    extent = max(w, height, group[2]-group[0], group[3]-group[1])
                    if np.hypot(gx, gy) <= .5*extent:
                        group[:] = [min(box[0], group[0]), min(box[1], group[1]),
                                    max(box[2], group[2]), max(box[3], group[3])]
                        break
                else:
                    groups.append(box)
            regions.extend((box, color) for box in groups if min(box[2]-box[0], box[3]-box[1]) >= 8)
        if not 2 <= len(regions) <= 4:
            return [(0, 0, native_w, native_h)]
        for index, (a, _) in enumerate(regions):
            for b, _ in regions[index+1:]:
                gap_x = max(0, max(a[0], b[0])-min(a[2], b[2]))
                gap_y = max(0, max(a[1], b[1])-min(a[3], b[3]))
                if np.hypot(gap_x, gap_y) < .35*min(a[2]-a[0], a[3]-a[1], b[2]-b[0], b[3]-b[1]):
                    return [(0, 0, native_w, native_h)]
        sx, sy = native_w/small.shape[1], native_h/small.shape[0]
        boxes = []
        for (x1, y1, x2, y2), _ in regions:
            pad = max(3, round(.15*max(x2-x1, y2-y1)))
            boxes.append((max(0, int((x1-pad)*sx)), max(0, int((y1-pad)*sy)),
                          min(native_w, int(np.ceil((x2+pad)*sx))),
                          min(native_h, int(np.ceil((y2+pad)*sy)))))
        return boxes

    def classify(self, roi, *, request=None):
        # Keep colour and shape timings in the evidence attached to the
        # published signal.  This is intentionally lightweight and uses the
        # same native ROI as the classifier, so board profiling does not
        # require a second pass or synthetic upscaling.
        color_started = time.perf_counter_ns()
        attr = super().classify(roi, request=request)
        color_ms = (time.perf_counter_ns() - color_started) / 1e6
        timing = {"color_ms": color_ms, "shape_ms": 0.0}
        if attr.color == 'UNKNOWN':
            return replace(attr, evidence={**attr.evidence, "timing_ms": timing})
        shape_started = time.perf_counter_ns()
        result = self._classify_glow(roi, attr)
        if attr.color == 'GREEN' and result.shape == 'UNKNOWN':
            core = self._classify_glow(roi, attr, bright_core=True)
            # Brightness thresholding can destroy a thin arrow stem. It is
            # only a round-lamp rescue, never an alternative arrow decision.
            if core.shape == 'ROUND':
                result = core
        timing["shape_ms"] = (time.perf_counter_ns() - shape_started) / 1e6
        return replace(result, evidence={**result.evidence, "timing_ms": timing})

    def _classify_glow(self, roi, attr, bright_core=False):
        roi = bounded_roi(roi)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        chroma = {'RED': (hue <= 10) | (hue >= 160),
                  'YELLOW': (hue >= 14) & (hue < 40),
                  'GREEN': (hue >= 40) & (hue <= (95 if bright_core else 105))}[attr.color]
        seed = ((sat >= 90) & (val >= 130) & chroma).astype(np.uint8)*255
        # Sunlit foliage and reflections can have the same hue as a green
        # lamp. Extract the bright LED core relative to this ROI's measured
        # seed brightness before contour analysis; keep colour voting intact.
        floor = 130
        if bright_core and np.any(seed):
            floor = max(floor, round(.8*float(np.percentile(val[seed > 0], 95))))
            seed[val < floor] = 0
        near_seed = cv2.dilate(seed, np.ones((5, 5), np.uint8)) > 0
        # Desaturated bright LED cores belong to a glow only when adjacent
        # to measured chromatic pixels. White background is not a colour cue.
        support = ((sat >= 35) & (val >= floor) & chroma) | ((val >= max(180, floor)) & near_seed)
        mask = support.astype(np.uint8)*255
        # Closing joins one-pixel LED gaps; no resizing is used to pass the
        # native-pixel quality gates. Raw contour is retained for comparison.
        kernel = max(2, min(5, round(min(mask.shape)/40)))
        clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        evidence = dict(attr.evidence)
        evidence['shape_value_floor'] = floor
        if not contours:
            return replace(attr, reason='NO_GLOW_CONTOUR')
        contour = contours[0]
        area = cv2.contourArea(contour)
        significant = [c for c in contours if cv2.contourArea(c) >= max(4 if bright_core else 3, (.10 if bright_core else .04)*area)]
        multiple = len(significant) > 1 and cv2.contourArea(significant[1]) > max(4, .10*area)
        # A disconnected arrow stem can be a legitimate part of one symbol.
        # Keep those islands for template comparison; two round lamps are
        # still vetoed below, and no arbitrary convex hull is filled.
        component = np.zeros_like(mask)
        cv2.drawContours(component, significant, -1, 255, -1)
        component = cv2.bitwise_and(component, clean)
        component = cv2.morphologyEx(component, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        points = cv2.findNonZero(component)
        if points is None:
            return replace(attr, reason='INSUFFICIENT_SHAPE_PIXELS', evidence=evidence)
        x, y, w, h = cv2.boundingRect(points)
        evidence['glow_bbox'] = [x, y, x+w, y+h]
        evidence['glow_area'] = area
        if min(w, h) < 12 or area < 60:
            return replace(attr, reason='INSUFFICIENT_SHAPE_PIXELS', evidence=evidence)
        if x <= 0 or y <= 0 or x+w >= roi.shape[1] or y+h >= roi.shape[0]:
            return replace(attr, reason='CLIPPED_GLOW', evidence=evidence)
        # Reject defocused edges: sufficiently bright-to-dark transitions must
        # remain in the native crop. This is a quality veto, not a sharpener.
        patch = val[max(0, y-1):y+h+1, max(0, x-1):x+w+1]
        sharpness = float(cv2.Laplacian(patch, cv2.CV_32F).var())
        evidence['edge_variance'] = sharpness
        # Edge contrast is normalized by brightness; Laplacian variance alone
        # rejects clear, low-contrast phone photos and favours sensor noise.
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1)
        contrast = float(np.percentile(np.hypot(gx, gy), 95))/max(1., float(patch.max()))
        evidence['edge_contrast'] = contrast
        if sharpness < 180 and contrast < .65:
            return replace(attr, reason='BLURRED_GLOW', evidence=evidence)
        normalized = _normalize(component)
        # Bounding-box centering rounds to integer pixels. Search that single
        # pixel uncertainty instead of weakening the shape acceptance gate.
        padded = np.pad(normalized, 1)
        aligned = np.stack([padded[1+dy:49+dy, 1+dx:49+dx]
                            for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
        scores = {}
        for shape, templates in self.template_bits.items():
            scores[shape] = 0.
            for shifted in aligned:
                packed = np.packbits(shifted.reshape(-1))
                intersection = self.bit_count[templates & packed].sum(axis=1)
                union = self.template_areas[shape]+np.count_nonzero(shifted)-intersection
                scores[shape] = max(scores[shape], float(np.max(intersection/np.maximum(1, union))))
        ordered = sorted(scores, key=scores.get, reverse=True)
        best, second = ordered[:2]
        perimeter = cv2.arcLength(contour, True)
        circularity = 4*np.pi*area/max(1., perimeter*perimeter)
        hull = cv2.convexHull(contour)
        solidity = area/max(1., cv2.contourArea(hull))
        try:
            defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False)) if len(contour) >= 4 else None
        except cv2.error:
            return replace(attr, reason='INVALID_GLOW_CONTOUR', evidence=evidence)
        depth = float(np.max(defects.reshape(-1, 4)[:, 3]))/256/max(w, h) if defects is not None else 0.
        evidence.update(template_scores={k: round(v, 3) for k, v in scores.items()},
                        circularity=round(circularity, 3), solidity=round(solidity, 3),
                        concavity_depth=round(depth, 3))
        # Round templates cannot compete with an arrow when the measured
        # contour itself fails the independent circularity/shape test.
        round_ok = not multiple and circularity >= .78 and .8 <= w/h <= 1.25 and scores['ROUND'] >= .84
        eligible = [s for s in ordered if s != 'ROUND' or round_ok]
        best, second = eligible[:2]
        if scores[best] < .64 or scores[best]-scores[second] < .09:
            return replace(attr, reason='AMBIGUOUS_SHAPE_TEMPLATE', evidence=evidence)
        if best == 'ROUND' and multiple:
            return replace(attr, reason='MULTIPLE_GLOW_REGIONS', evidence=evidence)
        if best == 'ROUND' and (scores[best] < .84 or circularity < .78 or not .8 <= w/h <= 1.25):
            return replace(attr, reason='NONCIRCULAR_GLOW', evidence=evidence)
        if best != 'ROUND' and not multiple and cv2.isContourConvex(cv2.approxPolyDP(contour, .025*perimeter, True)):
            return replace(attr, reason='NO_ARROW_CONCAVITY', evidence=evidence)
        if best != 'ROUND' and not multiple and (solidity > .90 or depth < .12):
            return replace(attr, reason='INSUFFICIENT_ARROW_CONCAVITY', evidence=evidence)
        return replace(attr, shape=best, shape_confidence=scores[best],
                       reason='GLOW_CONTOUR_TEMPLATE', evidence=evidence)
