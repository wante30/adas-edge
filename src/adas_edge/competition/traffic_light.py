from __future__ import annotations

from collections import deque

from .policy import TrafficLightState


def classify_traffic_light(frame, bbox, *, min_saturation: int = 70, min_value: int = 80, threshold: float = 0.08) -> TrafficLightState:
    import cv2
    import numpy as np
    if frame is None or not hasattr(frame, "shape") or frame.ndim != 3:
        return TrafficLightState.UNKNOWN
    values = bbox.get("bbox", bbox.get("bbox_xyxy")) if isinstance(bbox, dict) else getattr(bbox, "bbox_xyxy", bbox)
    if values is None or len(values) != 4:
        return TrafficLightState.UNKNOWN
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [max(0, min(limit, int(round(value)))) for value, limit in zip(values, (width, height, width, height))]
    if x2 <= x1 or y2 <= y1:
        return TrafficLightState.UNKNOWN
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    valid = (saturation >= min_saturation) & (value >= min_value)
    if int(valid.sum()) == 0:
        return TrafficLightState.UNKNOWN
    masks = {
        TrafficLightState.RED: (((hue <= 10) | (hue >= 170)) & valid),
        TrafficLightState.YELLOW: ((hue >= 15) & (hue <= 40) & valid),
        TrafficLightState.GREEN: ((hue >= 35) & (hue <= 95) & valid),
    }
    ranked = sorted(((float(mask.sum()) / float(valid.sum()), state) for state, mask in masks.items()), reverse=True)
    if ranked[0][0] < threshold or (len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.03):
        return TrafficLightState.UNKNOWN
    return ranked[0][1]


def classify_traffic_lights(frame, detections, **kwargs):
    states = [classify_traffic_light(frame, item, **kwargs) for item in detections]
    if not states or TrafficLightState.UNKNOWN in states:
        return TrafficLightState.UNKNOWN
    if TrafficLightState.RED in states:
        return TrafficLightState.RED
    if TrafficLightState.YELLOW in states:
        return TrafficLightState.YELLOW
    return TrafficLightState.GREEN


class TrafficLightStabilizer:
    def __init__(self, window_size: int = 3, green_confirmations: int = 2, stale_timeout: int = 3) -> None:
        if window_size < 1 or green_confirmations < 1 or stale_timeout < 0:
            raise ValueError("stabilizer parameters must be positive")
        self.window = deque(maxlen=window_size)
        self.green_confirmations = green_confirmations
        self.stale_timeout = stale_timeout
        self.stable = TrafficLightState.UNKNOWN
        self.unknown_count = 0

    def update(self, state: TrafficLightState) -> TrafficLightState:
        state = TrafficLightState(state)
        self.window.append(state)
        if state is TrafficLightState.RED:
            self.stable, self.unknown_count = state, 0
            return self.stable
        if state is TrafficLightState.UNKNOWN:
            self.unknown_count += 1
            if self.unknown_count > self.stale_timeout:
                self.stable = TrafficLightState.UNKNOWN
            return self.stable
        self.unknown_count = 0
        if state is TrafficLightState.GREEN:
            if sum(item is TrafficLightState.GREEN for item in self.window) >= self.green_confirmations:
                self.stable = state
        elif state is TrafficLightState.YELLOW:
            self.stable = state
        return self.stable
