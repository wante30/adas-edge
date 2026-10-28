from __future__ import annotations
try:
    from enum import StrEnum
except ImportError:  # Python 3.8/3.9 on RK3566 images
    from enum import Enum

    class StrEnum(str, Enum):
        pass
from typing import Any
import cv2
import numpy as np

class TrafficLightState(StrEnum):
    RED = "RED"; YELLOW = "YELLOW"; GREEN = "GREEN"; UNKNOWN = "UNKNOWN"

def _bbox(detection: Any) -> tuple[float, float, float, float] | None:
    if isinstance(detection, dict):
        b = detection.get("bbox") or detection.get("bbox_xyxy")
    else:
        b = [getattr(detection, k, None) for k in ("x1", "y1", "x2", "y2")]
    if b is None or len(b) != 4: return None
    return tuple(float(x) for x in b)

def classify_traffic_light(frame: np.ndarray, detection: Any, *, min_roi_size: int = 4, min_saturation: int = 70, min_value: int = 80, color_ratio_threshold: float = .08, min_confidence: float = .25, require_lamp_evidence: bool = False) -> TrafficLightState:
    """Conservative HSV classifier for one detector-provided traffic-light ROI."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3: return TrafficLightState.UNKNOWN
    if isinstance(detection, dict) and float(detection.get("confidence", 1.0)) < min_confidence: return TrafficLightState.UNKNOWN
    b = _bbox(detection)
    if b is None: return TrafficLightState.UNKNOWN
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in b)
    x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
    y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1 or x2-x1 < min_roi_size or y2-y1 < min_roi_size: return TrafficLightState.UNKNOWN
    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV); hch, sch, vch = cv2.split(hsv)
    if require_lamp_evidence:
        # A green leaf or a dim yellow sign can dominate the old HSV ratios.
        # For the daylight competition path require an illuminated coloured
        # patch together with dark housing. This is a candidate check, not a
        # learned signal/ego-lane classifier or a night-time accuracy claim.
        bright_color = ((sch >= min_saturation) & (vch >= 160)
                        & ((hch <= 10) | (hch >= 170) | ((hch >= 15) & (hch <= 95))))
        if (int(bright_color.sum()) < max(4, roi.shape[0] * roi.shape[1] * .015)
                or float((vch < 120).mean()) < .20):
            return TrafficLightState.UNKNOWN
    valid = (sch >= min_saturation) & (vch >= min_value)
    if int(valid.sum()) < max(2, min_roi_size): return TrafficLightState.UNKNOWN
    masks = {
        TrafficLightState.RED: (((hch <= 10) | (hch >= 170)) & valid),
        TrafficLightState.YELLOW: ((hch >= 15) & (hch <= 40) & valid),
        TrafficLightState.GREEN: ((hch >= 35) & (hch <= 95) & valid),
    }
    ratios = {state: float(mask.sum()) / float(valid.sum()) for state, mask in masks.items()}
    state, ratio = max(ratios.items(), key=lambda item: item[1])
    # Bright white glare or mixed lamps are deliberately UNKNOWN.
    ordered = sorted(ratios.values(), reverse=True)
    if ratio < color_ratio_threshold or (len(ordered) > 1 and ratio - ordered[1] < .03): return TrafficLightState.UNKNOWN
    return state

def classify_traffic_lights(frame: np.ndarray, detections: list[Any], **kwargs: Any) -> TrafficLightState:
    """Combine multiple signal ROIs conservatively for competition replay."""
    states = [classify_traffic_light(frame, detection, **kwargs) for detection in detections]
    if not states or TrafficLightState.UNKNOWN in states: return TrafficLightState.UNKNOWN
    if TrafficLightState.RED in states: return TrafficLightState.RED
    if TrafficLightState.YELLOW in states: return TrafficLightState.YELLOW
    return TrafficLightState.GREEN
