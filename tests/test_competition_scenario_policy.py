from dataclasses import replace

import pytest

from competition_ai.scenario import Boundary, Controls, Direction, Lane, Light, Maneuver, Mode, Motion, Perception
from competition_ai.scenario_policy import decide


def observed(light=Light.GREEN, **kwargs):
    return Perception(light, True, 0, **kwargs)


@pytest.mark.parametrize("lane,maneuver,led", [(Lane.LEFT, Maneuver.TURN_LEFT, 1),
                                             (Lane.STRAIGHT, Maneuver.FORWARD, 2),
                                             (Lane.RIGHT, Maneuver.TURN_RIGHT, 4)])
def test_green_selected_lane_turns_are_not_lane_changes(lane, maneuver, led):
    result = decide(Controls(lane=lane), observed(left_boundary=Boundary.SOLID,
                    right_boundary=Boundary.SOLID, lane_timestamp_ns=0), 0)
    assert result.maneuver == maneuver and result.led_mask == led and result.request_allowed


@pytest.mark.parametrize("motion,alarms", [(Motion.STOP, ()), (Motion.AUTO, ()), (Motion.GO, ("RED_LIGHT",))])
def test_red_light_stop_and_forced_go_alarm_are_independent(motion, alarms):
    result = decide(Controls(motion=motion), observed(Light.RED), 0)
    assert result.maneuver == Maneuver.STOP and result.led_mask == 0 and result.alarms == alarms
    assert result.simulated_motion == (Motion.GO if motion == Motion.GO else Motion.STOP)


def test_unassociated_green_and_expired_signal_never_grant_go():
    assert not decide(Controls(), replace(observed(), signal_associated=False), 0).request_allowed
    assert not decide(Controls(), observed(), 750_000_000).request_allowed
    assert not decide(Controls(), replace(observed(), signal_timestamp_ns=1), 0).request_allowed


def test_road_without_lights_can_move_and_change_only_on_the_correct_dashed_boundary():
    controls = Controls(mode=Mode.ROAD, request=Maneuver.CHANGE_LEFT)
    perception = Perception(left_boundary=Boundary.DASHED, right_boundary=Boundary.SOLID, lane_timestamp_ns=0)
    result = decide(controls, perception, 0)
    assert result.maneuver == Maneuver.CHANGE_LEFT and result.led_mask == 1
    blocked = decide(replace(controls, request=Maneuver.CHANGE_RIGHT), perception, 0)
    assert blocked.maneuver == Maneuver.FORWARD and blocked.led_mask == 2
    assert not blocked.request_allowed and blocked.alarms == ("SOLID_LANE",)
    assert decide(Controls(mode=Mode.ROAD), Perception(), 0).maneuver == Maneuver.FORWARD


def test_lane_change_unknown_or_expired_keeps_forward_with_reason():
    controls = Controls(mode=Mode.ROAD, request=Maneuver.CHANGE_LEFT)
    assert decide(controls, Perception(), 0).alarms == ("UNKNOWN_LANE",)
    old = Perception(left_boundary=Boundary.DASHED, lane_timestamp_ns=0)
    assert not decide(controls, old, 750_000_000).request_allowed


def test_pedestrian_priority_beats_green_navigation_and_manual_go():
    controls = Controls(mode=Mode.NAVIGATION, motion=Motion.GO,
                        navigation=Direction.LEFT, navigation_valid=True)
    perception = observed(pedestrian_hazard=True, hazard_timestamp_ns=0)
    result = decide(controls, perception, 0)
    assert result.maneuver == Maneuver.BRAKE and result.led_mask == 0
    assert result.alarms == ("PEDESTRIAN_DANGER",) and result.navigation_hint == Direction.LEFT
    assert result.simulated_motion == Motion.GO
    assert decide(controls, perception, 750_000_000).maneuver == Maneuver.STOP


def test_navigation_hint_does_not_implicitly_turn_or_require_nonexistent_green():
    controls = Controls(mode=Mode.NAVIGATION, navigation=Direction.RIGHT, navigation_valid=True)
    result = decide(controls, Perception(), 0)
    assert result.maneuver == Maneuver.FORWARD and result.navigation_hint == Direction.RIGHT
    assert decide(replace(controls, request=Maneuver.TURN_RIGHT), Perception(), 0).maneuver == Maneuver.TURN_RIGHT
    assert decide(replace(controls, signalized=True), Perception(), 0).maneuver == Maneuver.STOP


def test_wrong_scene_or_lane_and_manual_stop_remain_explicit():
    assert decide(Controls(mode=Mode.ROAD, request=Maneuver.TURN_LEFT), Perception(), 0).alarms == ("WRONG_SCENE",)
    assert decide(Controls(request=Maneuver.TURN_RIGHT), observed(), 0).alarms == ("WRONG_LANE",)
    assert decide(Controls(motion=Motion.STOP), observed(), 0).maneuver == Maneuver.STOP
    with pytest.raises(ValueError):
        Controls(request="brake")
    with pytest.raises(ValueError):
        Controls(mode="bad")
