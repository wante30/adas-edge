from tools.adas_hmi_sim.backend.navigation_state import NavigationState, constrain_intent
from tools.adas_hmi_sim.backend.scene_builder import SceneBuilder


def test_navigation_interval_and_red_light_constraint():
    state = NavigationState([{"start_ms": 0, "end_ms": 8000, "action": "STRAIGHT", "distance_m": 100}, {"start_ms": 8000, "end_ms": 12000, "action": "TURN_LEFT", "distance_m": 40}])
    assert state.at(0, 2000).action == "STRAIGHT"
    assert state.at(0, 9000).action == "TURN_LEFT"
    lanes = SceneBuilder.lanes()
    action, warnings = constrain_intent(state.at(0, 9000), "RED", lanes, [])
    assert action == "STOP"
    assert "RED_LIGHT" in warnings


def test_change_on_solid_line_becomes_warning():
    state = NavigationState([{"start_ms": 0, "end_ms": 100, "action": "CHANGE_LEFT"}])
    action, warnings = constrain_intent(state.at(0, 1), "GREEN", SceneBuilder.lanes(), [])
    assert action == "WARNING"
    assert "SOLID_LINE" in warnings
