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
        self.tracker = ObjectTracker(tracking.get("iou_threshold", 0.18), tracking.get("ema_alpha", 0.35), tracking.get("max_missed", 2))
        display = self.config.get("display", {})
        self.display_min_confidence = float(display.get("min_confidence", 0.28))
        self.display_max_objects = max(1, int(display.get("max_objects", 24)))
        self.display_max_lateral = float(display.get("max_lateral_m", 8.2))
        self.display_max_depth = float(display.get("max_depth_m", 58.0))

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

    def _display_measurements(self, measurements: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
        """Remove detector noise before it becomes a persistent 3D object.

        The detector is intentionally allowed to run at a low threshold for
        traffic-light recall.  That threshold is unsuitable for the polished
        HMI layer: one weak box can otherwise become a full-size car and a
        missed update can leave a small cluster of ghost cars on screen.
        This policy affects the display SceneFrame only; it does not alter the
        raw detector output or the board protocol.
        """
        if source not in {"front", "live"}:
            return measurements
        class_min = {"car": 0.30, "person": 0.34, "motorcycle": 0.32, "bicycle": 0.30, "bus": 0.25, "truck": 0.25}
        class_radius = {
            "car": (0.70, 1.35),
            "person": (0.42, 0.80),
            "motorcycle": (0.48, 0.90),
            "bicycle": (0.48, 0.90),
            "bus": (1.15, 1.80),
            "truck": (1.00, 1.70),
        }
        class_limits = {"car": 14, "person": 4, "motorcycle": 8, "bicycle": 6, "bus": 3, "truck": 4}
        candidates = [
            item for item in measurements
            if item["confidence"] >= max(self.display_min_confidence, class_min.get(item["class_name"], self.display_min_confidence))
            and abs(item["world"].x) <= self.display_max_lateral
            and 2.0 <= item["world"].z <= self.display_max_depth
        ]
        # Keep the strongest/nearest candidates first, then reject boxes that
        # collapse onto the same pseudo-world position.
        candidates.sort(key=lambda item: (float(item["confidence"]), -float(item["world"].z)), reverse=True)
        selected: list[dict[str, Any]] = []
        class_counts: dict[str, int] = {}
        for item in candidates:
            class_name = item["class_name"]
            if class_counts.get(class_name, 0) >= class_limits.get(class_name, self.display_max_objects):
                continue
            dx_limit, dz_limit = class_radius.get(class_name, (0.5, 0.9))
            duplicate = any(
                other["class_name"] == item["class_name"]
                and abs(other["world"].x - item["world"].x) <= dx_limit
                and abs(other["world"].z - item["world"].z) <= dz_limit
                for other in selected
            )
            if not duplicate:
                selected.append(item)
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            if len(selected) >= self.display_max_objects:
                break
        return selected

    @staticmethod
    def _lane_steering(lanes: dict[str, LaneBoundary]) -> float:
        """Estimate a display-only wheel angle from the measured lane curve."""
        left, right = lanes.get("left"), lanes.get("right")
        if left is None or right is None or len(left.points) < 3 or len(right.points) < 3:
            return 0.0
        def at(points: list[WorldPoint], z: float) -> float:
            return min(points, key=lambda point: abs(point.z - z)).x
        near = (at(left.points, 8.0) + at(right.points, 8.0)) * 0.5
        far = (at(left.points, 30.0) + at(right.points, 30.0)) * 0.5
        curvature = (far - near) / 22.0
        # A few degrees of wheel angle are useful in the cockpit view even
        # when the lane center is only slightly off-axis.  This is a display
        # cue, not a vehicle-control command.
        return max(-18.0, min(18.0, near * 6.0 + curvature * 90.0))

    @staticmethod
    def lanes(metadata: dict[str, Any] | None = None, *, allow_mock: bool = True) -> dict[str, LaneBoundary]:
        lanes = {}
        for side, x, default_style in (("left", -1.8, "solid"), ("right", 1.8, "dashed")):
            data = (metadata or {}).get(side)
            if data is None:
                if allow_mock:
                    lanes[side] = LaneBoundary([WorldPoint(x, z) for z in (0, 10, 25, 45, 70)], default_style, 0.0, "mock")
                else:
                    lanes[side] = LaneBoundary([], "unknown", 0.0, "metadata")
                continue
            style = str(data.get("style", data.get("type", "unknown"))).lower()
            points = [WorldPoint(float(point["x"]), float(point["z"])) if isinstance(point, dict) else WorldPoint(float(point[0]), float(point[1])) for point in data.get("points", [])]
            lanes[side] = LaneBoundary(points or [WorldPoint(x, 0), WorldPoint(x, 70)], style if style in {"solid", "dashed", "unknown"} else "unknown", float(data.get("confidence", 0.0)), "metadata")
        return lanes

    def build(self, frame_id: int, timestamp_ms: int, detections: list[Any], width: int, height: int, *, traffic_light_state: str = "UNKNOWN", traffic_light_decision_state: str | None = None, traffic_light_metadata: dict[str, Any] | None = None, lane_metadata: dict[str, Any] | None = None, navigation: NavigationIntent | None = None, warning_flags: list[str] | None = None, detector_update: bool = True, source: str = "mock", speed: float = 42.0) -> SceneFrame:
        normalised = [normalise_detection(item) for item in detections]
        measurements = []
        for item in normalised:
            if item["class_name"] in RENDER_CLASSES and item["bbox"].height > 0 and item["bbox"].width > 0:
                measurements.append({**item, "world": self.bbox_to_world(item["bbox"], item["class_name"], width, height)})
        measurements = self._display_measurements(measurements, source)
        objects = self.tracker.update(measurements, timestamp_ms) if detector_update else self.tracker.snapshot()
        if source in {"front", "live"}:
            objects = [obj for obj in objects if obj.alpha >= 0.35]
        lane_state = self.lanes(lane_metadata, allow_mock=source not in {"front", "live"})
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
        decision_light = (traffic_light_decision_state or light).upper()
        decision_light = decision_light if decision_light in LIGHT_STATES else "UNKNOWN"
        tl_boxes = [item for item in normalised if item["class_name"] == "traffic light"]
        if traffic_light_metadata is not None:
            accepted_bbox = traffic_light_metadata.get("bbox")
            tl = TrafficLight(light, float(traffic_light_metadata.get("confidence", 0.0)), BBox.from_value(accepted_bbox) if accepted_bbox is not None else None)
        else:
            tl = TrafficLight(light, max((item["confidence"] for item in tl_boxes), default=0.0), tl_boxes[0]["bbox"] if tl_boxes else None)
        nav = navigation or NavigationIntent()
        action, warnings = constrain_intent(nav, decision_light, lane_state, warnings)
        route_steering = -12.0 if action == "LEFT" else 12.0 if action == "RIGHT" else 0.0
        lane_steering = self._lane_steering(lane_state) if source in {"front", "live"} else 0.0
        steering = lane_steering if abs(lane_steering) > 2.0 else route_steering
        return SceneFrame(frame_id, timestamp_ms, EgoState(0.0 if action == "STOP" else speed, steering, action), lane_state, objects, tl, nav, warnings, {"object_count": len(objects), "track_count": self.tracker.count, "raw_detection_count": len(normalised)}, source, {"distance_mode": "PSEUDO_DISTANCE", "lane_mock": any(lane.source == "mock" for lane in lane_state.values()), "image_width": width, "image_height": height, "display_policy": "HMI_SIMULATION_ONLY", "display_filter": {"min_confidence": self.display_min_confidence, "max_objects": self.display_max_objects, "max_lateral_m": self.display_max_lateral, "stale_alpha_floor": 0.35}})
