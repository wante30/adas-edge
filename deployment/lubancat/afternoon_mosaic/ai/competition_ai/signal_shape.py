"""Conservative native-pixel glow morphology and shape-template backend."""
from __future__ import annotations

from dataclasses import replace
import cv2
import numpy as np

from .multi_signal import HSVBackend


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
    circle = np.zeros((64, 64), np.uint8)
    cv2.circle(circle, (32, 32), 25, 255, -1)
    result['ROUND'] = [_normalize(circle)]
    return result


class ContourTemplateBackend(HSVBackend):
    def __init__(self):
        self.templates = _templates()

    def classify(self, roi, *, request=None):
        attr = super().classify(roi)
        if attr.color == 'UNKNOWN':
            return attr
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        chroma = {'RED': (hue <= 10) | (hue >= 170),
                  'YELLOW': (hue >= 14) & (hue < 40),
                  'GREEN': (hue >= 40) & (hue <= 95)}[attr.color]
        mask = ((sat >= 90) & (val >= 130) & chroma).astype(np.uint8)*255
        # Closing joins one-pixel LED gaps; no resizing is used to pass the
        # native-pixel quality gates. Raw contour is retained for comparison.
        clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        evidence = dict(attr.evidence)
        if not contours:
            return replace(attr, reason='NO_GLOW_CONTOUR')
        contour = contours[0]
        area = cv2.contourArea(contour)
        if len(contours) > 1 and cv2.contourArea(contours[1]) > max(4, .10*area):
            return replace(attr, reason='MULTIPLE_GLOW_REGIONS', evidence=evidence)
        x, y, w, h = cv2.boundingRect(contour)
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
        if sharpness < 180:
            return replace(attr, reason='BLURRED_GLOW', evidence=evidence)
        component = np.zeros_like(mask)
        cv2.drawContours(component, [contour], -1, 255, -1)
        normalized = _normalize(component)
        scores = {}
        for shape, templates in self.templates.items():
            scores[shape] = max(float(np.logical_and(normalized, t).sum()) /
                                max(1, int(np.logical_or(normalized, t).sum())) for t in templates)
        ordered = sorted(scores, key=scores.get, reverse=True)
        best, second = ordered[:2]
        perimeter = cv2.arcLength(contour, True)
        circularity = 4*np.pi*area/max(1., perimeter*perimeter)
        evidence.update(template_scores={k: round(v, 3) for k, v in scores.items()},
                        circularity=round(circularity, 3))
        if scores[best] < .76 or scores[best]-scores[second] < .12:
            return replace(attr, reason='AMBIGUOUS_SHAPE_TEMPLATE', evidence=evidence)
        if best == 'ROUND' and (circularity < .78 or not .8 <= w/h <= 1.25):
            return replace(attr, reason='NONCIRCULAR_GLOW', evidence=evidence)
        if best != 'ROUND' and cv2.isContourConvex(cv2.approxPolyDP(contour, .025*perimeter, True)):
            return replace(attr, reason='NO_ARROW_CONCAVITY', evidence=evidence)
        return replace(attr, shape=best, shape_confidence=scores[best],
                       reason='GLOW_CONTOUR_TEMPLATE', evidence=evidence)
