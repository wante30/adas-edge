from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .scene import SceneInput

CompetitionAction = Literal["STOP", "GO", "LEFT", "RIGHT", "WARNING", "BRAKE"]

@dataclass(frozen=True)
class CompetitionCommand:
    action: CompetitionAction
    warning_flags: tuple[str, ...] = ()

class CompetitionAdas:
    """Competition-specific deterministic policy, independent of UI and PCIe."""
    def decide(self, scene: SceneInput) -> CompetitionCommand:
        light = scene.traffic_light_state.lower()
        vehicle = scene.vehicle_command.lower()
        if scene.pedestrian_hazard:
            return CompetitionCommand("BRAKE", ("PEDESTRIAN_DANGER",))
        if scene.obstacle_hazard:
            return CompetitionCommand("BRAKE", ("COLLISION_WARNING",))
        if light == "red":
            return CompetitionCommand("STOP", () if vehicle == "stop" else ("RED_LIGHT",))
        if light == "yellow":
            return CompetitionCommand("STOP", ("YELLOW_LIGHT",))
        if light == "unknown":
            return CompetitionCommand("STOP", ("TRAFFIC_LIGHT_STATE_UNKNOWN",))
        requested = {"change_left": "LEFT", "change_right": "RIGHT"}.get(vehicle)
        if requested:
            if scene.lane_type.lower() == "dashed": return CompetitionCommand(requested)  # type: ignore[arg-type]
            return CompetitionCommand("WARNING", ("LANE_CHANGE_UNSAFE",) if scene.lane_type.lower() == "unknown" else ("SOLID_LANE_CHANGE_FORBIDDEN",))
        return CompetitionCommand("GO") if vehicle != "stop" else CompetitionCommand("STOP")
