from dataclasses import replace

import numpy as np

from competition_ai.scenario import Lane, Maneuver, Mode, Motion
from tools.competition_console.controls import Controller
from tools.competition_console.pipeline import FrontPipeline
from competition_ai.image_motion import ImageMotion
from competition_ai.scenario_policy import decide


class Detector:
    def __init__(self):
        self.calls = 0
    def detect(self, frame):
        self.calls += 1
        return [{'class_name':'car','bbox':[0,0,10,10]}],1.


def test_keyboard_controls_share_one_state_and_restart_clears_forced_go():
    controller = Controller()
    for key in ('1','g','d','q'):
        controller.key(key)
    assert controller.controls.lane == Lane.LEFT and controller.controls.motion == Motion.GO
    assert controller.controls.mode == Mode.ROAD and controller.controls.request == Maneuver.CHANGE_LEFT
    controller.key(' ')
    assert controller.paused and controller.key('.') == 'step'
    assert controller.key('r') == 'restart'
    assert controller.controls.motion == Motion.AUTO and not controller.paused
    assert controller.key(27) == 'quit'


def test_pipeline_cadence_pause_expiry_and_replay_reset():
    main,specialist = Detector(),Detector()
    pipeline = FrontPipeline(main,specialist)
    frame = np.zeros((108,192,3),np.uint8)
    for fid in range(30):
        pipeline.advance(frame,fid,round(fid/30*1e9))
    assert main.calls == 10 and specialist.calls == 3
    pipeline.advance(frame,29,round(29/30*1e9))
    assert main.calls == 10
    old = pipeline.snapshot(Lane.STRAIGHT,2_000_000_000)
    assert old.main == [] and old.specialist == [] and not old.signal.associated
    pipeline.advance(frame,0,0)
    assert main.calls == 11 and specialist.calls == 4
    assert pipeline.last_frame == 0


def test_integrated_pipeline_event_drives_policy_brake_and_expires(monkeypatch):
    class Approaching:
        calls = 0
        def detect(self, frame):
            size = 28+self.calls*2.2
            self.calls += 1
            return [{'class_name':'car','confidence':.9,'bbox':[160-size/2,128-size*.65,160+size/2,128]}],1.
    pipeline = FrontPipeline(Approaching(),Detector(),hazards=True)
    monkeypatch.setattr(pipeline.hazard_engine.background,'update',lambda *args:ImageMotion(valid=True))
    frame = np.zeros((180,320,3),np.uint8)
    for fid in range(30):
        now = round(fid/30*1e9)
        pipeline.advance(frame,fid,now)
    snapshot = pipeline.snapshot(Lane.STRAIGHT,now)
    assert snapshot.hazards.obstacle_hazard
    decision = decide(Controller().controls,snapshot.as_perception(),now)
    assert decision.maneuver == Maneuver.BRAKE
    assert 'COLLISION_WARNING' in decision.alarms
    assert not pipeline.snapshot(Lane.STRAIGHT,now+300_000_000).as_perception().obstacle_hazard
