"""PC inference plus explicitly calibrated lamp observations for one stream."""
from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Condition, RLock, Thread

from competition_ai.bound_signal import BoundSignal, SignalObservation
from competition_ai.hazard_events import HazardEvents, HazardObservation
from competition_ai.scenario import Lane, Light, Perception


class _AsyncHazardEngine:
    """Run stateful hazard analysis without stalling the detector cadence.

    Hazard tracking is stateful, so it stays on one worker in source order.
    The pending slot is latest-only: a slow hazard pass cannot build an
    unbounded queue of stale frames behind the main detector.
    """

    def __init__(self, engine):
        self._engine = engine
        self._condition = Condition()
        self._state_lock = RLock()
        self._pending = None
        self._stopping = False
        self._worker = Thread(target=self._run, name='hazard-events', daemon=True)
        self._worker.start()

    @property
    def timings(self):
        return self._engine.timings

    @property
    def background(self):
        return self._engine.background

    def reset(self):
        with self._state_lock:
            with self._condition:
                self._pending = None
            self._engine.reset()

    def update(self, frame, detections, frame_id, timestamp_ns, **kwargs):
        # Detection rows are replaced on every main pass; copy the container
        # so the worker sees a stable snapshot while the next pass runs.
        task = (frame, list(detections), frame_id, timestamp_ns, kwargs)
        with self._condition:
            if self._stopping:
                return
            self._pending = task
            self._condition.notify()

    def current(self, now_ns):
        # HazardEvents publishes an immutable observation at the end of each
        # update. Reading that reference does not need to block the detector.
        return self._engine.current(now_ns)

    def close(self):
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify()

    def _run(self):
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                task = self._pending
                self._pending = None
            frame, detections, frame_id, timestamp_ns, kwargs = task
            with self._state_lock:
                self._engine.update(frame, detections, frame_id, timestamp_ns, **kwargs)


@dataclass(frozen=True)
class FrontSnapshot:
    main: list
    specialist: list
    signal: SignalObservation
    main_age_ms: float | None
    specialist_age_ms: float | None
    binding_source: str
    hazards: HazardObservation = HazardObservation()
    sample_timestamp_ns: int | None = None

    def at(self, timestamp_ns):
        if self.sample_timestamp_ns is None:
            return self
        elapsed_ms = max(0., (timestamp_ns-self.sample_timestamp_ns)/1e6)
        main_age = self.main_age_ms+elapsed_ms if self.main_age_ms is not None else None
        specialist_age = self.specialist_age_ms+elapsed_ms if self.specialist_age_ms is not None else None
        signal = self.signal
        if signal.timestamp_ns is not None and not 0 <= timestamp_ns-signal.timestamp_ns < 400_000_000:
            signal = replace(signal, light=Light.UNKNOWN, associated=False,
                             reason='observation_expired')
        hazards = self.hazards
        if hazards.reason != 'disabled' and (
                hazards.timestamp_ns is None
                or not 0 <= timestamp_ns-hazards.timestamp_ns < 250_000_000):
            hazards = replace(hazards, timestamp_ns=None, events=(), tracks=(),
                              reason='observation_expired', lead_id=None)
        return replace(self,
                       main=self.main if main_age is not None and 0 <= main_age < 750 else [],
                       specialist=self.specialist if specialist_age is not None and 0 <= specialist_age < 400 else [],
                       signal=signal, main_age_ms=main_age,
                       specialist_age_ms=specialist_age, hazards=hazards,
                       sample_timestamp_ns=timestamp_ns)

    def as_perception(self):
        return Perception(self.signal.light,self.signal.associated,self.signal.timestamp_ns,
                          pedestrian_hazard=self.hazards.pedestrian_hazard,
                          obstacle_hazard=self.hazards.obstacle_hazard,
                          hazard_timestamp_ns=self.hazards.timestamp_ns)


class FrontPipeline:
    def __init__(self, main, specialist, *, main_hz=10., specialist_hz=3., hazards=False,
                 hazard_path=None, hazards_async=False, main_every_request=False,
                 hazard_hz=None):
        if main_hz <= 0 or specialist_hz <= 0:
            raise ValueError('Inference cadences must be positive')
        if hazard_hz is not None and hazard_hz <= 0:
            raise ValueError('Hazard cadence must be positive when enabled')
        self.main_detector, self.specialist_detector = main,specialist
        self.main_hz, self.specialist_hz = main_hz,specialist_hz
        self.main_every_request = bool(main_every_request)
        self.hazard_hz = float(hazard_hz if hazard_hz is not None else main_hz)
        self.main_timings, self.specialist_timings = [],[]
        hazard_engine = (HazardEvents(**({'path':hazard_path} if hazard_path is not None else {}))
                         if hazards else None)
        self.hazard_engine = (_AsyncHazardEngine(hazard_engine)
                              if hazard_engine is not None and hazards_async
                              else hazard_engine)
        self.reset()

    def reset(self):
        self.main, self.specialist = [],[]
        self.last_main, self.last_specialist, self.last_frame = None,None,None
        self.next_main_schedule = None
        self.next_specialist_schedule = None
        self.next_hazard_schedule = None
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

    def advance(self, frame, frame_id, timestamp_ns, *, schedule_timestamp_ns=None):
        if self.last_frame is not None and frame_id < self.last_frame:
            self.reset()
        if self.last_frame == frame_id:
            return  # paused keyboard input cannot re-infer or refresh data
        source_now = timestamp_ns/1e9
        schedule_now = (timestamp_ns if schedule_timestamp_ns is None
                        else schedule_timestamp_ns)/1e9
        main_due = (self.main_every_request or self.next_main_schedule is None or
                    schedule_now + 1e-6 >= self.next_main_schedule)
        if main_due:
            self.main,elapsed = self.main_detector.detect(frame)
            self.main_timings.append(elapsed)
            self.last_main = source_now
            period = 1. / self.main_hz
            if self.next_main_schedule is None:
                self.next_main_schedule = schedule_now + period
            else:
                while self.next_main_schedule <= schedule_now + 1e-6:
                    self.next_main_schedule += period
            hazard_due = (self.hazard_engine is not None and
                          (self.next_hazard_schedule is None or
                           schedule_now + 1e-6 >= self.next_hazard_schedule))
            if hazard_due:
                self.hazard_engine.update(frame,self.main,frame_id,timestamp_ns)
                period = 1. / self.hazard_hz
                if self.next_hazard_schedule is None:
                    self.next_hazard_schedule = schedule_now + period
                else:
                    while self.next_hazard_schedule <= schedule_now + 1e-6:
                        self.next_hazard_schedule += period
        specialist_due = (self.specialist_detector is not None and
                          (self.next_specialist_schedule is None or
                           schedule_now + 1e-6 >= self.next_specialist_schedule))
        if specialist_due:
            height,width = frame.shape[:2]
            self.specialist,elapsed = self.specialist_detector.detect(frame[:round(height*.60)])
            self.specialist_timings.append(elapsed)
            self.last_specialist = source_now
            period = 1. / self.specialist_hz
            if self.next_specialist_schedule is None:
                self.next_specialist_schedule = schedule_now + period
            else:
                while self.next_specialist_schedule <= schedule_now + 1e-6:
                    self.next_specialist_schedule += period
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
        sa = ((now-self.last_specialist)*1000
              if self.specialist_detector is not None and self.last_specialist is not None
              else None)
        return FrontSnapshot(self.main if ma is not None and 0 <= ma < 750 else [],
                             self.specialist if sa is not None and 0 <= sa < 400 else [],
                             signal,ma,sa,binding[1] if binding else 'UNBOUND',
                             self.hazard_engine.current(now_ns) if self.hazard_engine else HazardObservation(),
                             now_ns)
