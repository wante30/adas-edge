"""Route intent and a display-only deterministic HMI state machine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import LaneBoundary, NavigationIntent

NAV_ACTIONS = {"STRAIGHT", "TURN_LEFT", "TURN_RIGHT", "CHANGE_LEFT", "CHANGE_RIGHT", "ARRIVE"}
LIGHT_STATES = {"UNKNOWN", "RED", "YELLOW", "GREEN"}


class NavigationState:
    def __init__(self, route: list[dict[str, Any]] | None = None):
        self.route = route or []
        for segment in self.route:
            if segment.get("action", "STRAIGHT") not in NAV_ACTIONS:
                raise ValueError(f"Unknown navigation action: {segment.get('action')}")

    @classmethod
    def from_file(cls, path: Path) -> "NavigationState":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def at(self, frame_id: int, timestamp_ms: int) -> NavigationIntent:
        for segment in self.route:
            if "start_frame" in segment:
                start, end, current = int(segment["start_frame"]), int(segment.get("end_frame", 2**63)), frame_id
            else:
                start, end, current = int(segment.get("start_ms", 0)), int(segment.get("end_ms", 2**63)), timestamp_ms
            if start <= current < end:
                distance = segment.get("distance_m")
                if distance is not None:
                    # Intent countdown is scripted, not odometry or GPS.
                    distance = max(0.0, float(distance) - max(0, current - start) * float(segment.get("countdown_per_unit", 0)))
                return NavigationIntent(segment.get("action", "STRAIGHT"), distance)
        return NavigationIntent()


def constrain_intent(navigation: NavigationIntent, traffic_light: str, lanes: dict[str, LaneBoundary], warning_flags: list[str]) -> tuple[str, list[str]]:
    """Does not control a car or replace competition_ai.CompetitionAdas."""

    warnings = list(dict.fromkeys(warning_flags))
    light = traffic_light.upper() if traffic_light.upper() in LIGHT_STATES else "UNKNOWN"
    if "PEDESTRIAN" in warnings or "COLLISION" in warnings:
        return "WARNING", warnings
    if light in {"RED", "YELLOW"}:
        flag = "RED_LIGHT" if light == "RED" else "YELLOW_LIGHT"
        return "STOP", list(dict.fromkeys(warnings + [flag]))
    if light == "UNKNOWN":
        return "STOP", list(dict.fromkeys(warnings + ["TRAFFIC_LIGHT_UNKNOWN"]))
    if navigation.action == "ARRIVE":
        return "STOP", warnings
    if navigation.action.startswith("CHANGE_"):
        side = "left" if navigation.action.endswith("LEFT") else "right"
        if lanes[side].style != "dashed":
            return "WARNING", list(dict.fromkeys(warnings + ["SOLID_LINE" if lanes[side].style == "solid" else "LANE_UNKNOWN"]))
    if navigation.action.endswith("LEFT"):
        return "LEFT", warnings
    if navigation.action.endswith("RIGHT"):
        return "RIGHT", warnings
    return "FORWARD", warnings
