from dataclasses import replace

import numpy as np
import time

from competition_ai.scenario import Lane, Light, Maneuver, Mode, Motion, Perception
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


def test_mode_tabs_cannot_silently_bypass_an_unbound_intersection():
    controller = Controller()
    assert controller.controls.signalized
    for key in ('d', 'n', 'h', 'i'):
        controller.key(key)
        decision = decide(controller.controls, Perception(), 0)
        assert controller.controls.signalized
        assert decision.maneuver == Maneuver.STOP
        assert decision.alarms == ('SIGNAL_UNASSOCIATED',)


def test_explicit_no_signal_scene_preserves_intersection_gate_and_replay_default():
    controller = Controller()
    controller.key('t')  # The intersection mode cannot disable its signal gate.
    assert controller.controls.signalized
    controller.key('d')
    controller.key('t')
    assert not controller.controls.signalized
    assert decide(controller.controls, Perception(), 0).maneuver == Maneuver.FORWARD
    controller.key('i')
    assert decide(controller.controls, Perception(), 0).maneuver == Maneuver.STOP
    controller.key('r')
    assert controller.controls.signalized and controller.controls.mode == Mode.INTERSECTION


def test_mode_change_keeps_bound_red_gate_and_cancels_binding():
    controller = Controller()
    controller.key('b')
    controller.key('g')
    controller.key('d')
    assert not controller.binding
    decision = decide(controller.controls, Perception(Light.RED, True, 0), 0)
    assert decision.alarms == ('RED_LIGHT',) and decision.maneuver == Maneuver.STOP
    assert decision.simulated_motion == Motion.GO


def test_view_toggles_and_cancel_do_not_change_driving_controls():
    controller = Controller()
    original = controller.controls
    controller.key('v')
    controller.key('f')
    controller.key('c')
    controller.key('b')
    assert not controller.overlay and not controller.diagnostics and not controller.bev and controller.binding
    controller.key('b')
    assert not controller.binding
    controller.key('b')
    assert controller.key(27) is None and not controller.binding
    controller.key('f')
    assert controller.key(27) is None and not controller.diagnostics
    controller.key('c')
    assert controller.key(27) is None and not controller.bev
    assert controller.key(27) == 'quit'
    assert controller.controls == original


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


def test_pipeline_cadence_can_use_arrival_clock_when_source_timestamps_jitter():
    main, specialist = Detector(), Detector()
    pipeline = FrontPipeline(main, specialist, main_hz=5., specialist_hz=1.)
    frame = np.zeros((108,192,3), np.uint8)
    for fid in range(50):
        # The PGVF producer clock is quantized just below 200 ms between AI
        # requests, while the sidecar receives those requests at a stable 5 Hz.
        source_ns = round(fid * 199.6e6)
        schedule_ns = fid * 200_000_000
        pipeline.advance(frame, fid, source_ns,
                         schedule_timestamp_ns=schedule_ns)
    assert main.calls == 50
    assert specialist.calls == 10


def test_pipeline_async_hazards_do_not_block_main_detector(monkeypatch):
    main, specialist = Detector(), Detector()
    pipeline = FrontPipeline(main, specialist, main_hz=10., specialist_hz=1.,
                             hazards=True, hazards_async=True)
    calls = []
    original_update = pipeline.hazard_engine._engine.update

    def slow_update(*args, **kwargs):
        time.sleep(.05)
        calls.append(args[2])
        return original_update(*args, **kwargs)

    monkeypatch.setattr(pipeline.hazard_engine._engine, 'update', slow_update)
    frame = np.zeros((108,192,3), np.uint8)
    started = time.perf_counter()
    for fid in range(4):
        pipeline.advance(frame, fid, fid * 100_000_000)
    elapsed = time.perf_counter() - started
    deadline = time.perf_counter() + 1.0
    while not calls and time.perf_counter() < deadline:
        time.sleep(.01)
    assert elapsed < .04
    assert calls
    pipeline.hazard_engine.close()


def test_pipeline_can_decouple_hazard_cadence_from_main_detector():
    main, specialist = Detector(), Detector()
    pipeline = FrontPipeline(main, specialist, main_hz=10., specialist_hz=1.,
                             hazards=True, hazard_hz=5.)
    frame = np.zeros((108,192,3), np.uint8)
    for fid in range(20):
        pipeline.advance(frame, fid, fid * 50_000_000)
    assert main.calls == 10
    assert len(pipeline.hazard_engine.timings) == 5


def test_pipeline_can_run_main_for_each_rate_limited_request():
    main, specialist = Detector(), Detector()
    pipeline = FrontPipeline(main, specialist, main_hz=10., specialist_hz=1.,
                             main_every_request=True)
    frame = np.zeros((108,192,3), np.uint8)
    for fid in range(12):
        pipeline.advance(frame, fid, round(fid * 100_000_000))
    assert main.calls == 12


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
