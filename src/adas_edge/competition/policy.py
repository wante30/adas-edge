from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag, StrEnum


class TrafficLightState(IntEnum):
    UNKNOWN = 0
    RED = 1
    YELLOW = 2
    GREEN = 3


class LaneState(IntEnum):
    UNKNOWN = 0
    SOLID = 1
    DASHED = 2


class AdasAction(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3


class WarningFlags(IntFlag):
    NONE = 0
    RED_LIGHT = 1 << 0
    YELLOW_LIGHT = 1 << 1
    TRAFFIC_LIGHT_UNKNOWN = 1 << 2
    PEDESTRIAN_DANGER = 1 << 3
    COLLISION_WARNING = 1 << 4
    SOLID_LANE_CHANGE_FORBIDDEN = 1 << 5
    LANE_CHANGE_UNSAFE = 1 << 6
    OVERSPEED = 1 << 7
    TURN_SIGNAL_REQUIRED = 1 << 8
    NAVIGATION_CONFLICT = 1 << 9


class NavigationCommand(StrEnum):
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    STRAIGHT = "straight"
    CHANGE_LEFT = "change_left"
    CHANGE_RIGHT = "change_right"


class TurnSignal(StrEnum):
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    HAZARD = "hazard"


@dataclass(frozen=True)
class SceneInput:
    traffic_light_state: TrafficLightState = TrafficLightState.UNKNOWN
    lane_state: LaneState = LaneState.UNKNOWN
    navigation_command: NavigationCommand = NavigationCommand.NONE
    requested_maneuver: NavigationCommand = NavigationCommand.NONE
    turn_signal: TurnSignal = TurnSignal.NONE
    current_speed_kph: float | None = None
    speed_limit_kph: float | None = None
    pedestrian_hazard: bool = False
    obstacle_hazard: bool = False
    forced_red_light_violation: bool = False


@dataclass(frozen=True)
class PolicyResult:
    action: AdasAction
    warning_flags: WarningFlags = WarningFlags.NONE


class CompetitionPolicy:
    def decide(self, scene: SceneInput) -> PolicyResult:
        warnings = WarningFlags.NONE
        if scene.pedestrian_hazard:
            warnings |= WarningFlags.PEDESTRIAN_DANGER
            return PolicyResult(AdasAction.STOP, warnings)
        if scene.obstacle_hazard:
            warnings |= WarningFlags.COLLISION_WARNING
            return PolicyResult(AdasAction.STOP, warnings)
        if scene.traffic_light_state is TrafficLightState.RED:
            warnings |= WarningFlags.RED_LIGHT
            if scene.forced_red_light_violation:
                warnings |= WarningFlags.RED_LIGHT
            return PolicyResult(AdasAction.STOP, warnings)
        if scene.traffic_light_state is TrafficLightState.YELLOW:
            return PolicyResult(AdasAction.STOP, WarningFlags.YELLOW_LIGHT)
        if scene.traffic_light_state is TrafficLightState.UNKNOWN:
            return PolicyResult(AdasAction.STOP, WarningFlags.TRAFFIC_LIGHT_UNKNOWN)
        maneuver = scene.requested_maneuver
        if maneuver in {NavigationCommand.CHANGE_LEFT, NavigationCommand.CHANGE_RIGHT}:
            expected = TurnSignal.LEFT if maneuver is NavigationCommand.CHANGE_LEFT else TurnSignal.RIGHT
            if scene.lane_state is LaneState.SOLID:
                warnings |= WarningFlags.SOLID_LANE_CHANGE_FORBIDDEN
            elif scene.lane_state is not LaneState.DASHED:
                warnings |= WarningFlags.LANE_CHANGE_UNSAFE
            if scene.turn_signal is not expected:
                warnings |= WarningFlags.TURN_SIGNAL_REQUIRED
            if warnings:
                action = AdasAction.FORWARD
            else:
                action = AdasAction.LEFT if expected is TurnSignal.LEFT else AdasAction.RIGHT
        elif scene.navigation_command is NavigationCommand.LEFT:
            action = AdasAction.LEFT
        elif scene.navigation_command is NavigationCommand.RIGHT:
            action = AdasAction.RIGHT
        else:
            action = AdasAction.FORWARD
        if scene.current_speed_kph is not None and scene.speed_limit_kph is not None and scene.current_speed_kph > scene.speed_limit_kph:
            warnings |= WarningFlags.OVERSPEED
        return PolicyResult(action, warnings)
