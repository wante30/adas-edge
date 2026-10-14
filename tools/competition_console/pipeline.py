"""PC inference plus explicitly calibrated lamp observations for one stream."""
from __future__ import annotations

from dataclasses import dataclass

from competition_ai.bound_signal import BoundSignal, SignalObservation
from competition_ai.hazard_events import HazardEvents, HazardObservation
from competition_ai.scenario import Lane, Perception


@dataclass(frozen=True)
class FrontSnapshot:
    main: list
    specialist: list
    signal: SignalObservation
    main_age_ms: float | None
    specialist_age_ms: float | None
    binding_source: str
    hazards: HazardObservation = HazardObservation()

    def as_perception(self):
        return Perception(self.signal.light,self.signal.associated,self.signal.timestamp_ns,
                          pedestrian_hazard=self.hazards.pedestrian_hazard,
                          obstacle_hazard=self.hazards.obstacle_hazard,
                          hazard_timestamp_ns=self.hazards.timestamp_ns)


class FrontPipeline:
    def __init__(self, main, specialist, *, main_hz=10., specialist_hz=3., hazards=False, hazard_path=None):
        if main_hz <= 0 or specialist_hz <= 0:
            raise ValueError('Inference cadences must be positive')
        self.main_detector, self.specialist_detector = main,specialist
        self.main_hz, self.specialist_hz = main_hz,specialist_hz
        self.main_timings, self.specialist_timings = [],[]
        self.hazard_engine = HazardEvents(**({'path':hazard_path} if hazard_path is not None else {})) if hazards else None
        self.reset()

    def reset(self):
        self.main, self.specialist = [],[]
        self.last_main, self.last_specialist, self.last_frame = None,None,None
        self.bindings = {}
        if self.hazard_engine is not None:
            self.hazard_engine.reset()

    def bind(self, frame, bbox, lanes, *, source='USER_SELECTED_HORIZONTAL_RYG_HEAD'):
        signal = BoundSignal(frame,bbox)
        if not signal.active:
            raise ValueError('Selected housing has insufficient visual contrast')
        if not lanes:
            raise ValueError('Specify at least one simulated lane')
        for lane in lanes:
            self.bindings[Lane(lane)] = (signal,source)

    @staticmethod
    def due(now,last,hz):
        return last is None or now-last >= 1./hz-1e-6

    def advance(self, frame, frame_id, timestamp_ns):
        if self.last_frame is not None and frame_id < self.last_frame:
            self.reset()
        if self.last_frame == frame_id:
            return  # paused keyboard input cannot re-infer or refresh data
        now = timestamp_ns/1e9
        main_due = self.due(now,self.last_main,self.main_hz)
        if main_due:
            self.main,elapsed = self.main_detector.detect(frame)
            self.main_timings.append(elapsed)
            self.last_main = now
            if self.hazard_engine is not None:
                self.hazard_engine.update(frame,self.main,frame_id,timestamp_ns)
        if self.due(now,self.last_specialist,self.specialist_hz):
            height,width = frame.shape[:2]
            self.specialist,elapsed = self.specialist_detector.detect(frame[:round(height*.60)])
            self.specialist_timings.append(elapsed)
            self.last_specialist = now
        seen = set()
        for signal,_ in self.bindings.values():
            if id(signal) in seen:
                continue
            seen.add(id(signal))
            signal.update(frame,frame_id,timestamp_ns,classify=main_due)
        self.last_frame = frame_id

    def snapshot(self, lane, now_ns):
        binding = self.bindings.get(Lane(lane))
        signal = binding[0].current(now_ns) if binding else SignalObservation()
        now = now_ns/1e9
        ma = (now-self.last_main)*1000 if self.last_main is not None else None
        sa = (now-self.last_specialist)*1000 if self.last_specialist is not None else None
        return FrontSnapshot(self.main if ma is not None and 0 <= ma < 750 else [],
                             self.specialist if sa is not None and 0 <= sa < 400 else [],
                             signal,ma,sa,binding[1] if binding else 'UNBOUND',
                             self.hazard_engine.current(now_ns) if self.hazard_engine else HazardObservation())
