"""Bounded latest-task attribute worker; never waits in the AI/Qt frame loop."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import threading
import time

from .multi_signal import MultiSignalTracker, unknown_directions


def expire_result(result, now_ns, *, available=True, reason=""):
    result = copy.deepcopy(result)
    result['published_at_ns'] = now_ns
    for signal in result['signals']:
        signal['age_ms'] = (now_ns-signal['timestamp_ns'])/1e6
        if not available or not signal['timestamp_ns'] <= now_ns < signal['expires_at_ns']:
            signal.update(color='UNKNOWN', shape='UNKNOWN', valid=False,
                          reason=reason or ('EXPIRED' if available else 'SOURCE_UNAVAILABLE'),
                          association='UNASSOCIATED', ego_relevant='UNKNOWN')
    # Association must be recomputed against observation expiry on every read.
    result['directions'] = unknown_directions(reason or 'NO_ASSOCIATION_EVIDENCE')
    return result


class LatestSignalWorker:
    def __init__(self, hz=2., tracker=None, jsonl_path=None):
        if not 0 < hz <= 2:
            raise ValueError('attribute frequency must be > 0 and <= 2 Hz')
        self.tracker = tracker or MultiSignalTracker()
        self.period = 1/hz
        self.condition = threading.Condition()
        self.pending = None
        self.result = None
        self.context = None
        self.stopping = False
        self.replaced = 0
        self.completed = 0
        self.errors = 0
        self.last_ms = None
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.thread = threading.Thread(target=self._run, name='signal-attributes', daemon=True)
        self.thread.start()

    def submit(self, frame, candidates, *, frame_id, timestamp_ns, context):
        with self.condition:
            if self.stopping:
                return
            if tuple(context) != self.context:
                self.context = tuple(context)
                self.result = None
            if self.pending is not None:
                self.replaced += 1
            # Source frame arrays are immutable after capture. Retain a single
            # reference instead of copying the full Q4 on each AI request.
            self.pending = (frame, list(candidates), frame_id, timestamp_ns, tuple(context))
            self.condition.notify()

    def current(self, now_ns, *, context, available=True, reason=''):
        with self.condition:
            result = self.result if tuple(context) == self.context else None
            metrics = dict(completed=self.completed, replaced=self.replaced,
                           errors=self.errors, attribute_ms=self.last_ms, hz=1/self.period)
        if result is None:
            result = dict(schema_version=1, source_id='Q4', context=list(context), frame_id=-1,
                          timestamp_ns=0, signals=[], debug_only=True, ttl_ms=self.tracker.ttl_ns/1e6,
                          rejections=[reason or 'WAITING_FOR_FRESH_DETECTION'])
        result = expire_result(result, now_ns, available=available, reason=reason)
        result['metrics'] = metrics
        return result

    def close(self):
        with self.condition:
            self.stopping = True
            self.pending = None
            self.condition.notify_all()
        self.thread.join(timeout=2)

    def _run(self):
        next_start = 0.
        while True:
            with self.condition:
                while not self.stopping:
                    delay = next_start-time.monotonic()
                    if self.pending is not None and delay <= 0:
                        break
                    self.condition.wait(timeout=max(.001, delay) if self.pending is not None else None)
                if self.stopping:
                    return
                frame, candidates, fid, ns, context = self.pending
                self.pending = None
            started = time.perf_counter()
            next_start = time.monotonic()+self.period
            try:
                now = time.monotonic_ns()
                if not 0 <= now-ns < 500_000_000:
                    # Slow YOLO must not turn historical boxes into current attributes.
                    result = dict(schema_version=1, source_id='Q4', context=list(context),
                                  frame_id=fid, timestamp_ns=ns, signals=[],
                                  directions=unknown_directions('INPUT_EXPIRED'),
                                  rejections=['INPUT_EXPIRED'], debug_only=True)
                else:
                    self.tracker.update(frame, candidates, frame_id=fid, timestamp_ns=ns, context=context)
                    result = self.tracker.snapshot(time.monotonic_ns(), context=context)
                elapsed = (time.perf_counter()-started)*1000
                with self.condition:
                    if context == self.context:
                        self.result = result
                    self.completed += 1
                    self.last_ms = elapsed
                if self.jsonl_path:
                    self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.jsonl_path.open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(dict(result, attribute_ms=elapsed), allow_nan=False)+'\n')
            except Exception as exc:
                with self.condition:
                    self.errors += 1
                    self.result = None
                print('SIGNAL_ATTRIBUTE_ERROR '+repr(exc), flush=True)
