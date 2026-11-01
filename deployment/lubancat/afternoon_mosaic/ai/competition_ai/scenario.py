"""Explicit competition inputs; buttons describe simulation, not perception."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Mode(str, Enum):
    INTERSECTION = "intersection"
    ROAD = "road"
    NAVIGATION = "navigation"
    HAZARD = "hazard"


class Lane(str, Enum):
    LEFT = "left"
    STRAIGHT = "straight"
    RIGHT = "right"


class Motion(str, Enum):
    AUTO = "auto"
    STOP = "stop"
    GO = "go"


class Maneuver(str, Enum):
    AUTO = "auto"
    STOP = "stop"
    FORWARD = "forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    CHANGE_LEFT = "change_left"
    CHANGE_RIGHT = "change_right"
    BRAKE = "brake"


class Direction(str, Enum):
    NONE = "none"
    LEFT = "left"
    FORWARD = "forward"
    RIGHT = "right"


class Light(str, Enum):
    UNKNOWN = "UNKNOWN"
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class Boundary(str, Enum):
    UNKNOWN = "unknown"
    SOLID = "solid"
    DASHED = "dashed"


@dataclass(frozen=True)
class Controls:
    mode: Mode = Mode.INTERSECTION
    lane: Lane = Lane.STRAIGHT
    motion: Motion = Motion.AUTO
    request: Maneuver = Maneuver.AUTO
    signalized: bool = False  # additional gate for road/navigation scenes
    navigation: Direction = Direction.NONE
    navigation_valid: bool = False

    def __post_init__(self):
        for name, cls in (("mode", Mode), ("lane", Lane), ("motion", Motion),
                          ("request", Maneuver), ("navigation", Direction)):
            object.__setattr__(self, name, cls(getattr(self, name)))
        if self.request == Maneuver.BRAKE:
            raise ValueError("BRAKE is a decision, not a simulated request")


@dataclass(frozen=True)
class Perception:
    light: Light = Light.UNKNOWN
    signal_associated: bool = False
    signal_timestamp_ns: int | None = None
    left_boundary: Boundary = Boundary.UNKNOWN
    right_boundary: Boundary = Boundary.UNKNOWN
    lane_timestamp_ns: int | None = None
    pedestrian_hazard: bool = False
    obstacle_hazard: bool = False
    hazard_timestamp_ns: int | None = None
    valid_for_ms: int = 750

    def __post_init__(self):
        for name, cls in (("light", Light), ("left_boundary", Boundary), ("right_boundary", Boundary)):
            object.__setattr__(self, name, cls(getattr(self, name)))
        if not isinstance(self.valid_for_ms, int) or not 1 <= self.valid_for_ms <= 60000:
            raise ValueError("Invalid perception lifetime")
        for timestamp in (self.signal_timestamp_ns, self.lane_timestamp_ns, self.hazard_timestamp_ns):
            if timestamp is not None and (not isinstance(timestamp, int) or timestamp < 0):
                raise ValueError("Invalid source timestamp")


@dataclass(frozen=True)
class Decision:
    maneuver: Maneuver
    direction: Direction
    request_allowed: bool
    alarms: tuple[str, ...]
    reason: str
    simulated_motion: Motion
    navigation_hint: Direction = Direction.NONE

    @property
    def led_mask(self) -> int:
        # The screenshot defines LED0 left, LED1 forward, LED2 right.
        return {Direction.NONE: 0, Direction.LEFT: 1,
                Direction.FORWARD: 2, Direction.RIGHT: 4}[self.direction]
