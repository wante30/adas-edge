from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Mapping, Sequence


Action = Literal["STOP", "FORWARD", "LEFT", "RIGHT"]


@dataclass(frozen=True)
class PerceptionState:
    """Algorithm-neutral state consumed by ADAS policy."""

    traffic_light: str | None = None
    pedestrians_in_danger_zone: bool = False
    obstacle_in_path: bool = False
    lane_type: str | None = None
    requested_direction: str = "forward"
    vehicles: Sequence[Mapping[str, object]] = field(default_factory=tuple)


@dataclass(frozen=True)
class ControlCommand:
    action: Action
    warning_flags: tuple[str, ...]
    confidence: float
    timestamp: str


class RuleBasedDecisionEngine:
    """Conservative, explainable phase-one ADAS policy."""

    def decide(self, state: PerceptionState | Mapping[str, object]) -> ControlCommand:
        if isinstance(state, Mapping):
            state = PerceptionState(**state)

        warnings: list[str] = []
        action: Action = "FORWARD"

        if state.traffic_light == "red":
            warnings.append("RED_LIGHT")
        elif state.traffic_light == "unknown":
            warnings.append("TRAFFIC_LIGHT_STATE_UNKNOWN")
        if state.pedestrians_in_danger_zone:
            warnings.append("PEDESTRIAN_DANGER")
        if state.obstacle_in_path:
            warnings.append("COLLISION_WARNING")
        if warnings:
            action = "STOP"
        elif state.lane_type == "dashed" and state.requested_direction in {"left", "right"}:
            action = state.requested_direction.upper()  # type: ignore[assignment]
        elif state.lane_type == "solid" and state.requested_direction in {"left", "right"}:
            warnings.append("SOLID_LANE_CHANGE_FORBIDDEN")

        return ControlCommand(
            action=action,
            warning_flags=tuple(warnings),
            confidence=1.0 if not warnings else 0.9,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
