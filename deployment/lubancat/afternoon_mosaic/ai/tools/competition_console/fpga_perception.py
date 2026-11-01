"""Translate current visual detections to the FPGA's HUD V2 perception fields."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
import math

from competition_ai.scenario import Light
from competition_ai.traffic_light import TrafficLightState, classify_traffic_light


class TrafficCode(IntEnum):
    OFF = 0
    RED = 1
    GREEN = 2


class LaneCode(IntEnum):
    NONE = 0
    SOLID = 1
    DASHED = 2


MIN_SIGNAL_CONFIDENCE = 0.25
SIGNAL_DUPLICATE_IOU = 0.5


@dataclass(frozen=True)
class HudPerception:
    traffic: TrafficCode
    pedestrian: bool
    static_target: bool
    overspeed: bool
    lane_left: LaneCode
    lane_right: LaneCode
    lka_warning: bool
    traffic_evidence: str

    def ext_request(self) -> int:
        value = 1 | (int(self.traffic) << 1)
        value |= int(self.pedestrian) << 3
        value |= int(self.static_target) << 4
        value |= int(self.overspeed) << 5
        value |= int(self.lane_left) << 6
        value |= int(self.lane_right) << 8
        value |= int(self.lka_warning) << 10
        return value

    def to_dict(self) -> dict:
        values = asdict(self)
        values["traffic"] = self.traffic.name
        values["lane_left"] = self.lane_left.name
        values["lane_right"] = self.lane_right.name
        values["ext_request"] = self.ext_request()
        return values


def build_hud_perception(frame, snapshot) -> HudPerception:
    bound = getattr(snapshot, "signal", None)
    if (bound is not None and getattr(bound, "associated", False)
            and getattr(bound, "timestamp_ns", None) is not None):
        light = getattr(bound, "light", Light.UNKNOWN)
        if light == Light.RED:
            traffic, evidence = TrafficCode.RED, "USER_BOUND_RED_SIGNAL"
        elif light == Light.GREEN:
            traffic, evidence = TrafficCode.GREEN, "USER_BOUND_GREEN_SIGNAL"
        elif light == Light.YELLOW:
            traffic, evidence = TrafficCode.OFF, "USER_BOUND_YELLOW_UNSUPPORTED_BY_HUD_PROTOCOL"
        else:
            traffic, evidence = TrafficCode.OFF, "USER_BOUND_SIGNAL_COLOR_UNCONFIRMED"
        return HudPerception(
            traffic=traffic,
            pedestrian=any(item.get("class_name") == "person" for item in snapshot.main),
            static_target=False,
            overspeed=False,
            lane_left=LaneCode.NONE,
            lane_right=LaneCode.NONE,
            lka_warning=False,
            traffic_evidence=evidence,
        )

    raw_signals = []
    for detections in (getattr(snapshot, "specialist", ()), getattr(snapshot, "main", ())):
        for item in detections or ():
            if not isinstance(item, dict) or item.get("class_name") != "traffic light":
                continue
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if math.isfinite(confidence):
                raw_signals.append((confidence, item))

    signals = _deduplicate_signals(
        item for confidence, item in raw_signals if confidence >= MIN_SIGNAL_CONFIDENCE
    )
    traffic = TrafficCode.OFF
    if not raw_signals:
        evidence = "NO_SIGNAL_ROI"
    elif not signals:
        evidence = "NO_CONFIDENT_SIGNAL_ROI"
    else:
        states = [
            classify_traffic_light(
                frame, item, min_confidence=MIN_SIGNAL_CONFIDENCE,
                require_lamp_evidence=True,
            )
            for item in signals
        ]
        confirmed = [state for state in states if state != TrafficLightState.UNKNOWN]
        confirmed_colours = set(confirmed)
        if TrafficLightState.RED in confirmed_colours and TrafficLightState.GREEN in confirmed_colours:
            evidence = "CONFLICTING_CONFIRMED_SIGNAL_COLORS"
        elif TrafficLightState.YELLOW in confirmed_colours and len(confirmed_colours) > 1:
            evidence = "CONFLICTING_CONFIRMED_SIGNAL_COLORS"
        elif TrafficLightState.YELLOW in confirmed_colours:
            evidence = "YELLOW_UNSUPPORTED_BY_HUD_PROTOCOL"
        elif TrafficLightState.RED in confirmed_colours:
            traffic = TrafficCode.RED
            evidence = ("ONE_DETECTED_RED_SIGNAL" if len(confirmed) == 1
                        else "MULTIPLE_RED_SIGNALS_AGREE")
            if len(confirmed) < len(states):
                evidence += "_WITH_UNRESOLVED_CANDIDATES"
        elif TrafficLightState.GREEN in confirmed_colours:
            traffic = TrafficCode.GREEN
            evidence = ("ONE_DETECTED_GREEN_SIGNAL" if len(confirmed) == 1
                        else "MULTIPLE_GREEN_SIGNALS_AGREE")
            if len(confirmed) < len(states):
                evidence += "_WITH_UNRESOLVED_CANDIDATES"
        else:
            evidence = "SIGNAL_COLOR_UNCONFIRMED"

    return HudPerception(
        traffic=traffic,
        pedestrian=any(item.get("class_name") == "person" for item in snapshot.main),
        static_target=False,
        overspeed=False,
        lane_left=LaneCode.NONE,
        lane_right=LaneCode.NONE,
        lka_warning=False,
        traffic_evidence=evidence,
    )


def _deduplicate_signals(candidates):
    """Keep the strongest detection when models report overlapping lamp boxes."""
    ordered = sorted(candidates, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    unique = []
    boxes = []
    for item in ordered:
        raw_box = item.get("bbox") or item.get("bbox_xyxy")
        if raw_box is None or len(raw_box) != 4:
            continue
        try:
            box = tuple(float(value) for value in raw_box)
        except (TypeError, ValueError):
            continue
        x1, y1, x2, y2 = box
        if not all(math.isfinite(value) for value in box) or x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        duplicate = False
        for old_box in boxes:
            ox1, oy1, ox2, oy2 = old_box
            intersection = max(0.0, min(x2, ox2) - max(x1, ox1)) * max(
                0.0, min(y2, oy2) - max(y1, oy1)
            )
            old_area = (ox2 - ox1) * (oy2 - oy1)
            union = area + old_area - intersection
            if union > 0 and intersection / union >= SIGNAL_DUPLICATE_IOU:
                duplicate = True
                break
        if not duplicate:
            unique.append(item)
            boxes.append(box)
    return unique
