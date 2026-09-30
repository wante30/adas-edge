from tools.adas_hmi_sim.backend.schemas import BBox, EgoState, LaneBoundary, NavigationIntent, SceneFrame, SceneObject, TrafficLight, WorldPoint


def test_sceneframe_json_shape_is_renderer_facing():
    frame = SceneFrame(1002, 1234, EgoState(42, 0, "FORWARD"), {"left": LaneBoundary([WorldPoint(-1.8, 2), WorldPoint(-1.8, 30)], "solid", .9, "metadata")}, [SceneObject(3, "car", .91, BBox(420, 170, 580, 360), WorldPoint(-1.2, 11.5))], TrafficLight("RED", .8), NavigationIntent("STRAIGHT", 200), ["RED_LIGHT"], {"object_count": 1}, "replay")
    data = frame.to_dict()
    assert data["frame_id"] == 1002
    assert data["objects"][0]["bbox"] == [420, 170, 580, 360]
    assert data["objects"][0]["world"] == {"x": -1.2, "z": 11.5}
    assert data["traffic_light"]["state"] == "RED"
    assert data["navigation"]["action"] == "STRAIGHT"
