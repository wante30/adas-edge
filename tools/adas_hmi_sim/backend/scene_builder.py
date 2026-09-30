"""Replaceable 2D metadata to pseudo-world mapping; never precise ranging."""

from __future__ import annotations

from typing import Any

from .navigation_state import LIGHT_STATES, constrain_intent
from .object_tracker import ObjectTracker
from .schemas import BBox, EgoState, LaneBoundary, NavigationIntent, SceneFrame, TrafficLight, WorldPoint

RENDER_CLASSES = {"person", "car", "bus", "truck", "bicycle", "motorcycle"}


def normalise_detection(item: Any) -> dict[str, Any]:
    """Accept competition Detection, core Detection, or existing JSONL rows."""

    if not isinstance(item, dict):
        item = item.to_dict()
    bbox = item.get("bbox") or item.get("bbox_xyxy")
    if bbox is None:
        bbox = [item.get(key, 0) for key in ("x1", "y1", "x2", "y2")]
    return {"class_name": str(item.get("class_name", "unknown")), "confidence": float(item.get("confidence", 0)), "bbox": BBox.from_value(bbox)}


class SceneBuilder:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        tracking = self.config.get("tracking", {})
        self.tracker = ObjectTracker(tracking.get("iou_threshold", 0.18), tracking.get("ema_alpha", 0.35), tracking.get("max_missed", 4))

    def reset(self) -> None:
        self.tracker.reset()

    def bbox_to_world(self, bbox: BBox, class_name: str, width: int, height: int) -> WorldPoint:
        mapping = self.config.get("mapping", {})
        # Config focal lengths are referenced to a 1280x720 image.
        fx = float(mapping.get("fx", 900)) * width / 1280
        fy = float(mapping.get("fy", 900)) * height / 720
        cx = float(mapping.get("cx", 640)) * width / 1280
        cy = float(mapping.get("cy", 310)) * height / 720
        u, v = bbox.bottom_center
        heights = {"car": 1.5, "person": 1.7, "bus": 3.0, "truck": 2.8, "bicycle": 1.3, "motorcycle": 1.4}
        heights.update(mapping.get("class_heights", {}))
        size_depth = fy * float(heights.get(class_name, 1.5)) / max(4.0, bbox.height)
        ground_depth = fy * float(mapping.get("camera_height", 1.4)) / max(height * 0.025, v - cy)
        weight = float(mapping.get("height_weight", 0.65))
        z = weight * size_depth + (1 - weight) * ground_depth
        z = max(float(mapping.get("min_depth", 2.0)), min(float(mapping.get("max_depth", 70.0)), z))
        x = (u - cx) * z / max(1.0, fx)
        x = max(-12.0, min(12.0, x))
        return WorldPoint(round(x, 4), round(z, 4))

    @staticmethod
    def lanes(metadata: dict[str, Any] | None = None) -> dict[str, LaneBoundary]:
        lanes = {}
        for side, x, default_style in (("left", -1.8, "solid"), ("right", 1.8, "dashed")):
            data = (metadata or {}).get(side)
            if data is None:
                lanes[side] = LaneBoundary([WorldPoint(x, z) for z in (0, 10, 25, 45, 70)], default_style, 0.0, "mock")
                continue
            style = str(data.get("style", data.get("type", "unknown"))).lower()
            points = [WorldPoint(float(point["x"]), float(point["z"])) if isinstance(point, dict) else WorldPoint(float(point[0]), float(point[1])) for point in data.get("points", [])]
            lanes[side] = LaneBoundary(points or [WorldPoint(x, 0), WorldPoint(x, 70)], style if style in {"solid", "dashed", "unknown"} else "unknown", float(data.get("confidence", 0.0)), "metadata")
        return lanes

    def build(self, frame_id: int, timestamp_ms: int, detections: list[Any], width: int, height: int, *, traffic_light_state: str = "UNKNOWN", lane_metadata: dict[str, Any] | None = None, navigation: NavigationIntent | None = None, warning_flags: list[str] | None = None, detector_update: bool = True, source: str = "mock", speed: float = 42.0) -> SceneFrame:
        normalised = [normalise_detection(item) for item in detections]
        measurements = []
        for item in normalised:
            if item["class_name"] in RENDER_CLASSES and item["bbox"].height > 0 and item["bbox"].width > 0:
                measurements.append({**item, "world": self.bbox_to_world(item["bbox"], item["class_name"], width, height)})
        objects = self.tracker.update(measurements, timestamp_ms) if detector_update else self.tracker.snapshot()
        lane_state = self.lanes(lane_metadata)
        warnings = list(warning_flags or [])
        # Image danger zones are a display heuristic, not safety certification.
        for obj in objects:
            u, v = obj.bbox.bottom_center
            if obj.alpha >= 0.8 and width * 0.35 <= u <= width * 0.65 and v >= height * 0.75:
                if obj.class_name == "person":
                    warnings.append("PEDESTRIAN")
                elif obj.world.z < 7:
                    warnings.append("COLLISION")
        light = traffic_light_state.upper()
        light = light if light in LIGHT_STATES else "UNKNOWN"
        tl_boxes = [item for item in normalised if item["class_name"] == "traffic light"]
        tl = TrafficLight(light, max((item["confidence"] for item in tl_boxes), default=0.0), tl_boxes[0]["bbox"] if tl_boxes else None)
        nav = navigation or NavigationIntent()
        action, warnings = constrain_intent(nav, light, lane_state, warnings)
        return SceneFrame(frame_id, timestamp_ms, EgoState(0.0 if action == "STOP" else speed, -12.0 if action == "LEFT" else 12.0 if action == "RIGHT" else 0.0, action), lane_state, objects, tl, nav, warnings, {"object_count": len(objects), "track_count": self.tracker.count}, source, {"distance_mode": "PSEUDO_DISTANCE", "lane_mock": any(lane.source == "mock" for lane in lane_state.values()), "image_width": width, "image_height": height, "display_policy": "HMI_SIMULATION_ONLY"})
