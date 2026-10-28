from tools.adas_hmi_sim.backend.scene_builder import SceneBuilder
from tools.adas_hmi_sim.backend.schemas import BBox, SceneFrame


def test_bbox_to_pseudo_world_is_explainable_and_monotonic():
    builder = SceneBuilder({"mapping": {"fx": 900, "fy": 900, "cx": 640, "cy": 310}})
    near = builder.bbox_to_world(BBox(590, 250, 690, 450), "car", 1280, 720)
    far = builder.bbox_to_world(BBox(610, 180, 670, 260), "car", 1280, 720)
    assert near.z < far.z
    assert abs(near.x) < 0.2


def test_scene_builder_tracks_and_serialises_scene_frame():
    builder = SceneBuilder()
    first = builder.build(1, 50, [{"class_name": "car", "confidence": .9, "bbox": [500, 200, 580, 350]}], 960, 540, traffic_light_state="GREEN")
    second = builder.build(2, 100, [{"class_name": "car", "confidence": .92, "bbox": [502, 202, 582, 352]}], 960, 540, traffic_light_state="GREEN")
    assert first.objects[0].id == second.objects[0].id
    restored = SceneFrame.from_dict(second.to_dict())
    assert restored.objects[0].world.z == second.objects[0].world.z
    assert restored.metadata["distance_mode"] == "PSEUDO_DISTANCE"


def test_front_display_filters_weak_out_of_road_and_duplicate_tracks():
    builder = SceneBuilder({"display": {"min_confidence": 0.28, "max_objects": 24, "max_lateral_m": 8.2}})
    detections = [
        {"class_name": "car", "confidence": 0.92, "bbox": [430, 300, 500, 500]},
        {"class_name": "car", "confidence": 0.88, "bbox": [431, 301, 501, 501]},
        {"class_name": "car", "confidence": 0.12, "bbox": [50, 50, 80, 100]},
    ]
    scene = builder.build(1, 50, detections, 960, 540, source="front")
    assert len(scene.objects) == 1
    assert scene.metrics["raw_detection_count"] == 3
    assert scene.metadata["display_filter"]["stale_alpha_floor"] == 0.35


def test_front_lane_curve_produces_display_steering_cue():
    builder = SceneBuilder()
    lane = {
        "left": {"points": [{"x": -2.0, "z": 4}, {"x": -1.5, "z": 8}, {"x": 0.5, "z": 30}]},
        "right": {"points": [{"x": 2.0, "z": 4}, {"x": 2.5, "z": 8}, {"x": 4.5, "z": 30}]},
    }
    scene = builder.build(1, 50, [], 960, 540, lane_metadata=lane, source="front")
    assert scene.ego.steering > 0
