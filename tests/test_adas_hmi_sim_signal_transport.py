from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from tools.adas_hmi_sim.backend.navigation_state import NavigationState
from tools.adas_hmi_sim.backend.scene_builder import SceneBuilder
from tools.adas_hmi_sim.backend.server import SimulatorOptions, create_app
from tools.adas_hmi_sim.backend.video_pipeline import FramePacket


class OneFrameSource:
    def __init__(self, metadata):
        self.packet = FramePacket(1, 100, np.zeros((80, 120, 3), dtype=np.uint8), [], metadata)

    def next(self):
        packet, self.packet = self.packet, None
        return packet


def _scene(mode: str, metadata: dict):
    frontend = Path(__file__).resolve().parents[1] / "tools" / "adas_hmi_sim" / "frontend"
    options = SimulatorOptions(OneFrameSource(metadata), SceneBuilder(), NavigationState(), frontend, mode)
    with TestClient(create_app(options)) as client:
        with client.websocket_connect("/ws") as socket:
            return socket.receive_json()["scene"]


def test_mock_scripted_green_preserves_demo_action():
    scene = _scene("mock", {"traffic_light_state": "GREEN"})
    assert scene["traffic_light"]["state"] == "GREEN"
    assert scene["ego"]["action"] == "FORWARD"


def test_real_unassociated_green_is_displayed_without_go_permission():
    scene = _scene("front", {
        "traffic_light": {"state": "GREEN", "decision_state": "UNKNOWN", "association": "UNVERIFIED"},
    })
    assert scene["traffic_light"]["state"] == "GREEN"
    assert scene["ego"]["action"] == "STOP"
    assert scene["metadata"]["traffic_light"]["association"] == "UNVERIFIED"
    assert "TRAFFIC_LIGHT_UNKNOWN" in scene["warnings"]


def test_real_source_without_association_does_not_fall_back_to_green_action():
    scene = _scene("front", {"traffic_light_state": "GREEN"})
    assert scene["ego"]["action"] == "STOP"


def test_front_partial_lane_metadata_does_not_inject_mock_boundary():
    scene = _scene("front", {"lanes": {"left": {"points": [{"x": -2, "z": 4}], "confidence": 0.5}}})
    assert scene["lanes"]["left"]["source"] == "metadata"
    assert scene["lanes"]["right"]["source"] == "metadata"
    assert scene["lanes"]["right"]["points"] == []
    assert scene["metadata"]["lane_mock"] is False
