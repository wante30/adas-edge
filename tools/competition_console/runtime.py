"""Bounded recording and source-clock playback for the native cockpit."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full, Queue
from threading import Event, Thread
import time

import cv2
import numpy as np

from tools.competition_console.latest_frame_slot import LatestFrameSlot


class LatestFrameWorker:
    """Run one task at a time and reject submissions while work is pending."""
    def __init__(self, target):
        self.target = target
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='cockpit-inference')
        self.future = None
        self.closed = False

    @property
    def busy(self):
        return self.future is not None

    def submit(self, *args, **kwargs):
        if self.closed:
            raise RuntimeError('Inference worker is closed')
        if self.future is not None:
            return False
        self.future = self.executor.submit(self.target, *args, **kwargs)
        return True

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        future, self.future = self.future, None
        return future.result()

    def wait(self):
        if self.future is None:
            return None
        future, self.future = self.future, None
        return future.result()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.future is not None:
                self.future.result()
                self.future = None
        finally:
            self.executor.shutdown(wait=True)


class PrefetchedCapture:
    """Bounded, ordered prefetch for file replay; seeks stop and discard old reads.

    The caller remains responsible for source frame IDs. Unlike a live-camera
    latest-frame queue this class preserves every file frame, including EOF.
    """
    def __init__(self, capture, *, capacity=2):
        if capacity < 1: raise ValueError('Queue capacity must be positive')
        self.capture, self.capacity = capture, capacity
        self.queue = Queue(maxsize=capacity)
        self.stop = Event()
        self.worker = None
        self.error = None
        self.closed = False

    def _run(self):
        try:
            while not self.stop.is_set():
                result = self.capture.read()
                while not self.stop.is_set():
                    try:
                        self.queue.put(result, timeout=.05)
                        break
                    except Full:
                        continue
                if not result[0]: return
        except BaseException as exc:
            self.error = exc

    def _stop(self):
        self.stop.set()
        if self.worker is not None: self.worker.join()
        self.worker = None

    def read(self):
        if self.closed: raise RuntimeError('Video capture is closed')
        if self.worker is None:
            self.stop.clear()
            self.worker = Thread(target=self._run, name='cockpit-video-decoder', daemon=True)
            self.worker.start()
        while True:
            if self.error is not None:
                raise RuntimeError('Video decoding failed') from self.error
            try:
                return self.queue.get(timeout=.05)
            except Empty:
                if not self.worker.is_alive(): return False, None

    def set(self, prop, value):
        if self.closed: raise RuntimeError('Video capture is closed')
        self._stop()
        self.queue = Queue(maxsize=self.capacity)
        self.error = None
        if not self.capture.set(prop, value):
            raise ValueError('Video seek failed; source frame identity cannot be guaranteed')
        return True

    def get(self, prop):
        if self.worker is not None:
            raise RuntimeError('Read capture metadata before starting prefetch')
        return self.capture.get(prop)

    def isOpened(self): return not self.closed and self.capture.isOpened()

    def release(self):
        if not self.closed:
            self._stop()
            self.capture.release()
            self.queue = Queue(maxsize=self.capacity)
            self.closed = True


class PlaybackPacer:
    """Use cumulative deadlines so a slow inference frame adds no permanent drift.

    This never skips video frames or inference. Pause/replay resets the wall-clock
    anchor. The first rendered frame anchors playback after model cold start.
    """
    def __init__(self, fps, *, clock=time.perf_counter):
        if fps <= 0:
            raise ValueError('FPS must be positive')
        self.fps, self.clock = fps, clock
        self.reset()

    def reset(self, fps=None):
        if fps is not None:
            if fps <= 0: raise ValueError('FPS must be positive')
            self.fps = fps
        self.anchor = self.first_frame = self.last_frame = None
        self.lateness_ms = []

    def delay_ms(self, frame_id, *, paused=False, unpaced=False):
        now = self.clock()
        if paused:
            self.anchor = self.first_frame = self.last_frame = None
            return 30
        if self.anchor is None or (self.last_frame is not None and frame_id <= self.last_frame):
            self.anchor, self.first_frame = now, frame_id
        target = self.anchor + (frame_id-self.first_frame)/self.fps
        self.lateness_ms.append(max(0., (now-target)*1000))
        self.last_frame = frame_id
        return 1 if unpaced else max(1, int((target+1/self.fps-now)*1000))

    def summary(self):
        values = self.lateness_ms
        return {'measured_frames': len(values), 'first_frame_anchors_after_cold_start': True,
                'p50_ms': float(np.median(values)) if values else None,
                'p95_ms': float(np.percentile(values, 95)) if values else None,
                'max_ms': max(values) if values else None,
                'last_ms': values[-1] if values else None}


class BoundedVideoWriter:
    """Encode on one worker with backpressure; every submitted frame is retained.

    Frame copies protect against caller reuse. A full queue blocks the producer;
    it never drops frames or allows memory/recording latency to grow unbounded.
    Errors are raised on write/release, including failures on the final frame.
    """
    def __init__(self, path, fourcc, fps, size, *, capacity=3, writer_factory=None):
        if capacity < 1: raise ValueError('Queue capacity must be positive')
        self.queue = Queue(maxsize=capacity)
        self.ready = Event()
        self.error = None
        self.closed = False
        self.count = 0
        self.encode_ms = []
        self.peak_queued = 0
        self.factory = writer_factory or cv2.VideoWriter
        self.worker = Thread(target=self._run, args=(path, fourcc, fps, size),
                             name='cockpit-video-encoder', daemon=True)
        self.worker.start()
        self.ready.wait()
        self._check()

    def _check(self):
        if self.error is not None:
            raise RuntimeError('Native recording encoder failed') from self.error

    def _run(self, *args):
        writer = None
        try:
            writer = self.factory(*args)
            if not writer.isOpened():
                raise RuntimeError('Cannot open video writer')
            self.ready.set()
            while True:
                image = self.queue.get()
                try:
                    if image is None: break
                    start = time.perf_counter()
                    writer.write(image)
                    self.encode_ms.append((time.perf_counter()-start)*1000)
                    self.count += 1
                finally:
                    self.queue.task_done()
        except BaseException as exc:
            self.error = exc
        finally:
            if writer is not None:
                try: writer.release()
                except BaseException as exc: self.error = self.error or exc
            self.ready.set()

    def isOpened(self):
        return self.ready.is_set() and self.error is None and not self.closed

    def _put(self, item):
        while True:
            self._check()
            try:
                self.queue.put(item, timeout=.05)
                self.peak_queued = max(self.peak_queued, self.queue.qsize())
                return
            except Full:
                continue

    def write(self, image):
        if self.closed: raise RuntimeError('Recording is closed')
        self._put(image.copy())

    def release(self):
        if not self.closed:
            self.closed = True
            try:
                self._put(None)
            finally:
                self.worker.join()
        self._check()

    def summary(self):
        return {'submitted_frames_encoded': self.count, 'queue_capacity': self.queue.maxsize,
                'peak_queued': self.peak_queued, 'frames_dropped': 0,
                'encode_total_s': sum(self.encode_ms)/1000,
                'encode_p95_ms': float(np.percentile(self.encode_ms, 95)) if self.encode_ms else None}
