from adas_edge.decision.engine import RuleBasedDecisionEngine


def test_red_light_overrides_forward_request() -> None:
    engine = RuleBasedDecisionEngine()

    command = engine.decide(
        {"traffic_light": "red", "requested_direction": "forward"}
    )

    assert command.action == "STOP"
    assert "RED_LIGHT" in command.warning_flags


def test_green_light_permits_forward_motion() -> None:
    command = RuleBasedDecisionEngine().decide(
        {"traffic_light": "green", "requested_direction": "forward"}
    )

    assert command.action == "FORWARD"
    assert command.warning_flags == ()


def test_pedestrian_overrides_green_light() -> None:
    command = RuleBasedDecisionEngine().decide(
        {
            "traffic_light": "green",
            "pedestrians_in_danger_zone": True,
            "requested_direction": "forward",
        }
    )

    assert command.action == "STOP"
    assert "PEDESTRIAN_DANGER" in command.warning_flags


def test_dashed_lane_permits_requested_lane_change() -> None:
    command = RuleBasedDecisionEngine().decide(
        {"lane_type": "dashed", "requested_direction": "left"}
    )

    assert command.action == "LEFT"


def test_solid_lane_blocks_requested_lane_change() -> None:
    command = RuleBasedDecisionEngine().decide(
        {"lane_type": "solid", "requested_direction": "right"}
    )

    assert command.action == "FORWARD"
    assert "SOLID_LANE_CHANGE_FORBIDDEN" in command.warning_flags


def test_obstacle_in_path_raises_collision_warning_and_stops() -> None:
    command = RuleBasedDecisionEngine().decide(
        {"traffic_light": "green", "obstacle_in_path": True}
    )

    assert command.action == "STOP"
    assert "COLLISION_WARNING" in command.warning_flags


def test_unknown_detected_traffic_light_fails_safe() -> None:
    command = RuleBasedDecisionEngine().decide(
        {"traffic_light": "unknown", "requested_direction": "forward"}
    )

    assert command.action == "STOP"
    assert "TRAFFIC_LIGHT_STATE_UNKNOWN" in command.warning_flags


def test_independent_hazards_are_all_reported() -> None:
    command = RuleBasedDecisionEngine().decide(
        {"traffic_light": "red", "pedestrians_in_danger_zone": True, "obstacle_in_path": True}
    )

    assert command.action == "STOP"
    assert set(command.warning_flags) == {"RED_LIGHT", "PEDESTRIAN_DANGER", "COLLISION_WARNING"}
