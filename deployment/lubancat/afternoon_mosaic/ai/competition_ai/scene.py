from __future__ import annotations
from dataclasses import dataclass
try:
    from enum import StrEnum
except ImportError:  # Python 3.8/3.9 on RK3566 images
    from enum import Enum

    class StrEnum(str, Enum):
        pass

class LaneType(StrEnum): SOLID="solid"; DASHED="dashed"; UNKNOWN="unknown"
class NavigationCommand(StrEnum): LEFT="left"; RIGHT="right"; STRAIGHT="forward"; NONE="none"
class VehicleCommand(StrEnum): GO="go"; STOP="stop"; CHANGE_LEFT="change_left"; CHANGE_RIGHT="change_right"; NONE="none"

@dataclass(frozen=True)
class SceneInput:
    traffic_light_state: str = "unknown"
    lane_type: str = "unknown"
    navigation_command: str = "none"
    vehicle_command: str = "none"
    pedestrian_hazard: bool = False
    obstacle_hazard: bool = False

    def as_perception(self) -> dict[str, object]:
        direction = {NavigationCommand.LEFT: "left", NavigationCommand.RIGHT: "right", NavigationCommand.STRAIGHT: "forward"}.get(self.navigation_command, "forward")
        return {"traffic_light": self.traffic_light_state.lower(), "lane_type": self.lane_type.lower(), "requested_direction": direction, "pedestrians_in_danger_zone": self.pedestrian_hazard, "obstacle_in_path": self.obstacle_hazard}
