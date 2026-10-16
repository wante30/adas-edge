from dataclasses import dataclass

import numpy as np

from competition_ai.scenario import Controls, Direction, Maneuver, Mode, Perception
from competition_ai.scenario_policy import decide
from tools.competition_console.navigation_session import NavigationSession, apply_navigation


@dataclass
class Observation:
    direction: str
    source_frame_id: int
    timestamp_ns: int
    valid: bool = True
    distance_m: float = 42.0
    inference_ms: float = 1.0


class Reader:
    def __init__(self):
        self.calls = []

    def update(self, image, frame_id, timestamp_ns):
        self.calls.append(frame_id)
        return Observation('right', frame_id, timestamp_ns)


def test_navigation_source_cadence_pause_replay_and_expiry():
    reader = Reader()
    session = NavigationSession(reader)
    image = np.zeros((8,8,3), np.uint8)
    for frame_id in range(30):
        session.advance(image, frame_id, round(frame_id/30*1e9))
    assert reader.calls == [0,10,20]
    session.advance(image,29,round(29/30*1e9))
    assert reader.calls == [0,10,20]
    assert session.current(1_000_000_000)['valid']
    assert not session.current(1_500_000_000)['valid']
    assert session.current(1_500_000_000)['direction'] == 'none'
    assert not session.current(0)['valid']
    assert not session.current(1_000_000_000,ended=True)['valid']
    session.advance(image,0,0)
    assert reader.calls == [0,10,20,0]


def test_navigation_hint_cannot_request_turn_or_bypass_lamp_gate():
    controls = Controls(mode=Mode.NAVIGATION,signalized=True)
    result = apply_navigation(controls,{'valid':True,'direction':'right'})
    assert result.navigation == Direction.RIGHT and result.navigation_valid
    assert result.request == Maneuver.AUTO and result.signalized
    decision = decide(result,Perception(),0)
    assert decision.maneuver == Maneuver.STOP
    assert decision.navigation_hint == Direction.RIGHT
    assert decision.alarms == ('SIGNAL_UNASSOCIATED',)


def test_uturn_and_unreadable_crop_do_not_become_left_or_stale_hint():
    controls = Controls(navigation=Direction.RIGHT,navigation_valid=True)
    for observation in ({'valid':True,'direction':'uturn'},
                        {'valid':False,'direction':'right'}):
        result = apply_navigation(controls,observation)
        assert result.navigation == Direction.NONE and not result.navigation_valid


def test_navigation_producer_cannot_claim_a_different_source_frame():
    class WrongFrame(Reader):
        def update(self,image,frame_id,timestamp_ns):
            return Observation('right',frame_id+1,timestamp_ns)
    session = NavigationSession(WrongFrame())
    session.advance(np.zeros((8,8,3),np.uint8),10,333_333_333)
    current = session.current(333_333_333)
    assert not current['valid'] and current['direction'] == 'none'
    assert current['reason'] == 'navigation_source_mismatch'
