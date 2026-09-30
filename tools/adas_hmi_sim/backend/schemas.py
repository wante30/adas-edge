"""Stable JSON protocol for the PC ADAS HMI simulator.

The browser only consumes :class:`SceneFrame`.  Detector adapters and the
scene builder can therefore change independently of the renderer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_value(cls, value: Any) -> "BBox":
        if isinstance(value, dict):
            value = [value.get(key, 0.0) for key in ("x1", "y1", "x2", "y2")]
        values = list(value or (0.0, 0.0, 0.0, 0.0))
        values += [0.0] * (4 - len(values))
        return cls(*(_float(item) for item in values[:4]))

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, self.y2)


@dataclass(frozen=True)
class WorldPoint:
    x: float
    z: float


@dataclass(frozen=True)
class Velocity:
    vx: float = 0.0
    vz: float = 0.0


@dataclass
class SceneObject:
    id: int
    class_name: str
    confidence: float
    bbox: BBox
    world: WorldPoint
    velocity: Velocity = field(default_factory=Velocity)
    alpha: float = 1.0
    track_age: int = 1
    pseudo_distance: bool = True

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = self.bbox.as_list()
        return data


@dataclass(frozen=True)
class LaneBoundary:
    points: list[WorldPoint]
    style: str = "unknown"
    confidence: float = 0.0
    source: str = "metadata"

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [asdict(point) for point in self.points],
            "style": self.style,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass(frozen=True)
class TrafficLight:
    state: str = "UNKNOWN"
    confidence: float = 0.0
    bbox: BBox | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"state": self.state, "confidence": self.confidence}
        if self.bbox is not None:
            data["bbox"] = self.bbox.as_list()
        return data


@dataclass(frozen=True)
class NavigationIntent:
    action: str = "STRAIGHT"
    distance_m: float | None = None
    source: str = "demo_route"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EgoState:
    speed: float = 0.0
    steering: float = 0.0
    action: str = "FORWARD"


@dataclass
class SceneFrame:
    frame_id: int
    timestamp_ms: int
    ego: EgoState
    lanes: dict[str, LaneBoundary]
    objects: list[SceneObject]
    traffic_light: TrafficLight
    navigation: NavigationIntent
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float | int] = field(default_factory=dict)
    source: str = "mock"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamp_ms": self.timestamp_ms,
            "ego": asdict(self.ego),
            "lanes": {name: lane.to_dict() for name, lane in self.lanes.items()},
            "objects": [obj.to_dict() for obj in self.objects],
            "traffic_light": self.traffic_light.to_dict(),
            "navigation": self.navigation.to_dict(),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SceneFrame":
        lanes = {
            name: LaneBoundary(
                points=[WorldPoint(_float(point.get("x")), _float(point.get("z"))) for point in data.get("points", [])],
                style=str(data.get("style", "unknown")),
                confidence=_float(data.get("confidence")),
                source=str(data.get("source", "metadata")),
            )
            for name, data in value.get("lanes", {}).items()
        }
        objects = []
        for item in value.get("objects", []):
            world = item.get("world", {})
            velocity = item.get("velocity", {})
            objects.append(
                SceneObject(
                    id=int(item.get("id", 0)),
                    class_name=str(item.get("class_name", "unknown")),
                    confidence=_float(item.get("confidence")),
                    bbox=BBox.from_value(item.get("bbox")),
                    world=WorldPoint(_float(world.get("x")), _float(world.get("z"))),
                    velocity=Velocity(_float(velocity.get("vx")), _float(velocity.get("vz"))),
                    alpha=_float(item.get("alpha"), 1.0),
                    track_age=int(item.get("track_age", 1)),
                    pseudo_distance=bool(item.get("pseudo_distance", True)),
                )
            )
        traffic = value.get("traffic_light", {})
        navigation = value.get("navigation", {})
        return cls(
            frame_id=int(value.get("frame_id", 0)),
            timestamp_ms=int(value.get("timestamp_ms", 0)),
            ego=EgoState(**{key: value.get("ego", {}).get(key, default) for key, default in (("speed", 0.0), ("steering", 0.0), ("action", "FORWARD"))}),
            lanes=lanes,
            objects=objects,
            traffic_light=TrafficLight(str(traffic.get("state", "UNKNOWN")), _float(traffic.get("confidence")), BBox.from_value(traffic["bbox"]) if traffic.get("bbox") else None),
            navigation=NavigationIntent(str(navigation.get("action", "STRAIGHT")), navigation.get("distance_m"), str(navigation.get("source", "demo_route"))),
            warnings=[str(item) for item in value.get("warnings", [])],
            metrics=dict(value.get("metrics", {})),
            source=str(value.get("source", "unknown")),
            metadata=dict(value.get("metadata", {})),
        )


def serialise_scene(scene: SceneFrame) -> dict[str, Any]:
    """Return a JSON-compatible scene payload for WebSocket clients."""

    return scene.to_dict()
