from dataclasses import replace

import pytest

from competition_ai.scenario import Boundary, Controls, Direction, Lane, Light, Maneuver, Mode, Motion, Perception
from competition_ai.scenario_policy import decide
from competition_io.hud_v1 import Icon, Register, Request
from competition_io.scenario_hud import MockScenarioHud, project_scenario


def projection(controls, perception, *, now_ns=0, source_ns=0, decision=None):
    return project_scenario(controls, perception, decision or decide(controls, perception, now_ns),
                            now_ns=now_ns, source_frame_id=1, source_timestamp_ns=source_ns)


def green():
    return Perception(Light.GREEN, True, 0)


@pytest.mark.parametrize("lane,state,blink,led", [
    (Lane.LEFT, 0x103, 1, 1), (Lane.STRAIGHT, 0x101, 0, 2), (Lane.RIGHT, 0x105, 2, 4),
])
def test_direction_is_not_a_copy_of_competition_led_bits(lane, state, blink, led):
    item = projection(Controls(lane=lane), green())
    assert (item.state, item.blink) == (state, blink)
    assert item.sidecar['competition_led_mask'] == led
    assert not item.sidecar['legacy_adst_generated']
    if lane == Lane.STRAIGHT:
        assert not item.state & (Icon.LEFT | Icon.RIGHT)
        assert 'straight_arrow' in item.sidecar['hud1_unrepresented']


def test_turn_and_lane_change_keep_distinct_semantics_with_same_visual_direction():
    turn = projection(Controls(lane=Lane.LEFT), green())
    change = projection(Controls(mode=Mode.ROAD, request=Maneuver.CHANGE_LEFT),
                        Perception(left_boundary=Boundary.DASHED, lane_timestamp_ns=0))
    assert (turn.state, turn.blink) == (change.state, change.blink)
    assert turn.sidecar['checked_decision']['maneuver'] == 'turn_left'
    assert change.sidecar['checked_decision']['maneuver'] == 'change_left'


def test_normal_red_stop_has_no_direction_and_forced_red_has_warning():
    red = Perception(Light.RED, True, 0)
    stopped = projection(Controls(), red)
    forced = projection(Controls(motion=Motion.GO), red)
    assert (stopped.state, stopped.blink) == (0x101, 0)
    assert (forced.state, forced.blink) == (0x109, 4)
    assert forced.sidecar['checked_decision']['maneuver'] == 'stop'
    assert forced.sidecar['checked_decision']['simulated_motion'] == 'go'
    assert forced.sidecar['competition_led_mask'] == 0


@pytest.mark.parametrize('kind', ['pedestrian_hazard', 'obstacle_hazard'])
def test_hazard_priority_warns_without_encoding_brake_as_a_legacy_turn(kind):
    item = projection(Controls(), replace(green(), **{kind: True, 'hazard_timestamp_ns': 0}))
    assert (item.state, item.blink) == (0x109, 4)
    assert item.sidecar['checked_decision']['maneuver'] == 'brake'
    assert item.sidecar['projected_direction'] == Direction.NONE


@pytest.mark.parametrize('boundary,expected_alarm', [(Boundary.SOLID, 'SOLID_LANE'), (Boundary.UNKNOWN, 'UNKNOWN_LANE')])
def test_blocked_lane_change_cannot_send_a_direction_icon(boundary, expected_alarm):
    item = projection(Controls(mode=Mode.ROAD, request=Maneuver.CHANGE_LEFT),
                      Perception(left_boundary=boundary, lane_timestamp_ns=0))
    assert (item.state, item.blink) == (0x109, 4)
    assert item.sidecar['checked_decision']['alarms'] == (expected_alarm,)
    assert not item.sidecar['checked_decision']['request_allowed']


def test_unknown_signal_and_navigation_are_warning_not_permission():
    for controls in (Controls(), Controls(mode=Mode.NAVIGATION)):
        item = projection(controls, Perception())
        assert (item.state, item.blink) == (0x109, 4)
        assert item.sidecar['competition_led_mask'] == 0


@pytest.mark.parametrize('now_ns,source_ns,reason', [
    (0, 1, 'source_timestamp_in_future'),
    (750_000_000, 0, 'source_expired'),
])
def test_source_envelope_failure_clears_a_previously_valid_left_icon(now_ns, source_ns, reason):
    controls = Controls(lane=Lane.LEFT)
    item = projection(controls, green(), now_ns=now_ns, source_ns=source_ns,
                      decision=decide(controls, green(), 0))
    assert (item.state, item.blink) == (0x109, 4)
    assert item.sidecar['validity'] == reason
    assert not item.sidecar['semantic_decision_valid']
    assert item.sidecar['competition_led_mask'] == 0


def test_decision_cannot_be_reused_with_different_input_or_future_observation():
    controls = Controls(lane=Lane.LEFT)
    old_decision = decide(controls, green(), 0)
    for observation in (Perception(Light.RED, True, 0), Perception(Light.GREEN, True, 1)):
        item = projection(controls, observation, decision=old_decision)
        assert item.sidecar['validity'] == 'decision_does_not_match_current_inputs'
        assert item.state == 0x109


def test_encoded_words_reach_mock_receiver_and_old_arrows_clear_on_next_state():
    sink = MockScenarioHud()
    for index, lane in enumerate((Lane.LEFT, Lane.STRAIGHT, Lane.RIGHT), 1):
        item = projection(Controls(lane=lane), green())
        receipt = sink.submit(item)
        word = int.from_bytes(bytes.fromhex(receipt['request_bytes_le_hex']), 'little')
        decoded = Request.decode(word)
        assert decoded == sink.simulator.applied
        assert decoded.sequence == index
        assert decoded.state == item.state and decoded.blink == item.blink
        assert receipt['receiver_verified'] and receipt['hardware_writes'] == 0
        assert receipt['transaction']['accepted'] and receipt['transaction']['applied']
        assert receipt['receiver_snapshot']['effective'] & 0xE == item.state & 0xE
    assert [offset for offset, _ in sink.simulator.writes] == [Register.REQUEST]*3


def test_timeout_does_not_claim_fpga_commit_or_send_a_second_request():
    sink = MockScenarioHud()
    sink.simulator.pixel_running = False
    receipt = sink.submit(projection(Controls(lane=Lane.LEFT), green()), timeout_ms=1)
    assert not receipt['receiver_verified']
    assert receipt['transaction']['phase'] == 'UNKNOWN'
    assert receipt['transaction']['accepted'] and not receipt['transaction']['applied']
    assert len(sink.simulator.writes) == 1


@pytest.mark.parametrize('field,value', [('source_frame_id', -1), ('now_ns', -1),
                                        ('source_timestamp_ns', 1.2), ('valid_for_ms', 0)])
def test_bad_envelope_rejected_before_any_transport(field, value):
    kwargs = {'now_ns': 0, 'source_frame_id': 1, 'source_timestamp_ns': 0, 'valid_for_ms': 750}
    kwargs[field] = value
    with pytest.raises(ValueError):
        project_scenario(Controls(), green(), decide(Controls(), green(), 0), **kwargs)
