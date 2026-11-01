#!/usr/bin/env python3
"""Small ByteTrack-compatible tracker for the 10 Hz Windows front stream.

No third-party tracker dependency.  It keeps the production detection dict
(x/y/width/height/className/confidence) intact and adds only trackId.
Vehicle-family association is isolated from person/two-wheel families.
"""
from __future__ import annotations

import numpy as np
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # RK3566 image has no SciPy; greedy IoU is sufficient
    linear_sum_assignment = None


def family(cls: str) -> str:
    if cls in ("car", "truck", "bus"):
        return "vehicle"
    if cls in ("motorcycle", "bicycle"):
        return "two_wheel"
    return cls


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    return inter / max(1e-9, (ax2-ax1)*(ay2-ay1) +
                       (bx2-bx1)*(by2-by1) - inter)


class _Track:
    next_id = 1

    def __init__(self, det: dict):
        self.id = _Track.next_id
        _Track.next_id += 1
        self.family = family(det["className"])
        self.class_name = det["className"]
        self.time_since_update = 0
        self.hits = 0
        self.kf = None
        x, y, w, h = det["x"], det["y"], det["width"], det["height"]
        self.bbox = (x, y, x + w, y + h)
        self._init_filter(x, y, w, h)
        self.hits = 1

    def _init_filter(self, x, y, w, h):
        import cv2
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0
        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.01
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        z = np.array([[x + w / 2], [y + h / 2], [w], [h]], dtype=np.float32)
        self.kf.statePost = np.vstack([z, np.zeros((4, 1), dtype=np.float32)])

    def predict(self,dt=1.):
        for i in range(4):self.kf.transitionMatrix[i,i+4]=dt
        self.kf.predict()
        self.time_since_update += 1
        s = self.kf.statePre
        cx, cy, w, h = (float(s[i, 0]) for i in range(4))
        w, h = max(2.0, w), max(2.0, h)
        self.bbox = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def update(self, det: dict):
        import cv2
        x, y, w, h = det["x"], det["y"], det["width"], det["height"]
        z = np.array([[x + w / 2], [y + h / 2], [w], [h]], dtype=np.float32)
        self.kf.correct(z)
        self.class_name = det["className"]
        self.bbox = (x, y, x + w, y + h)
        self.time_since_update = 0
        self.hits += 1


class ByteTracker:
    """Two-stage Kalman/IoU tracker, tuned for the 10 Hz front stream."""

    def __init__(self, track_buffer: int = 30, match_thresh: float = 0.8,
                 high_thresh: float = 0.5, low_thresh: float = 0.1,
                 new_track_thresh: float = 0.55):
        self.tracks: list[_Track] = []
        self.buffer = track_buffer
        self.match_thresh = match_thresh
        self.high = high_thresh
        self.low = low_thresh
        self.new_thresh = new_track_thresh
        self.last_timestamp_ns=None
        self.retention_ns=600000000

    @staticmethod
    def _cost(track: _Track, det: dict) -> float:
        if track.family != family(det["className"]):
            return 2.0
        return 1.0 - box_iou(track.bbox, (
            det["x"], det["y"], det["x"] + det["width"], det["y"] + det["height"]))

    def _associate(self, tracks, dets, max_cost):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        # Vectorized camera-local IoU avoids Python N*M loops on the RK3566.
        a=np.asarray([t.bbox for t in tracks],dtype=np.float32)
        b=np.asarray([(d["x"],d["y"],d["x"]+d["width"],d["y"]+d["height"]) for d in dets],dtype=np.float32)
        inter=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-np.maximum(a[:,None,:2],b[None,:,:2])).prod(axis=2)
        area_a=np.maximum(0,a[:,2:]-a[:,:2]).prod(axis=1)
        area_b=np.maximum(0,b[:,2:]-b[:,:2]).prod(axis=1)
        cost=1-inter/np.maximum(1e-9,area_a[:,None]+area_b[None,:]-inter)
        tf=np.asarray([t.family for t in tracks]);df=np.asarray([family(d["className"]) for d in dets])
        cost[tf[:,None]!=df[None,:]]=2.
        if linear_sum_assignment is not None:
            rows, cols = linear_sum_assignment(cost)
            pairs = list(zip(rows, cols))
        else:
            order=np.argsort(cost,axis=None)
            pairs=((int(k//len(dets)),int(k%len(dets))) for k in order if cost.flat[k]<=max_cost)
        matched, used_t, used_d = [], set(), set()
        for i, j in pairs:
            if i in used_t or j in used_d:
                continue
            if cost[i, j] <= max_cost:
                matched.append((tracks[i], dets[j]))
                used_t.add(i)
                used_d.add(j)
        return matched, [i for i in range(len(tracks)) if i not in used_t], \
               [j for j in range(len(dets)) if j not in used_d]

    def update(self, dets: list[dict], timestamp_ns=None) -> list[dict]:
        if timestamp_ns is not None:
            if self.last_timestamp_ns is not None and timestamp_ns<=self.last_timestamp_ns:return []
            dt=.1 if self.last_timestamp_ns is None else min(.6,(timestamp_ns-self.last_timestamp_ns)/1e9)
            self.last_timestamp_ns=timestamp_ns
            self.tracks=[t for t in self.tracks if timestamp_ns-(getattr(t,"last_seen_ns",None) or timestamp_ns)<=self.retention_ns]
        else:dt=1.
        for track in self.tracks:
            track.predict(dt)
        high = [d for d in dets if d["confidence"] >= self.high]
        low = [d for d in dets if self.low <= d["confidence"] < self.high]
        output = []

        matched, unmatched_tracks, unmatched_high = self._associate(
            self.tracks, high, self.match_thresh)
        for track, det in matched:
            track.update(det)
            det["trackId"] = track.id
            track.last_seen_ns=timestamp_ns
            det["velocity"]=[float(track.kf.statePost[i,0]) for i in range(4,8)]
            output.append(det)

        remaining = [self.tracks[i] for i in unmatched_tracks]
        matched_low, unmatched_remaining, _ = self._associate(
            remaining, low, 0.5)
        for track, det in matched_low:
            track.update(det)
            det["trackId"] = track.id
            track.last_seen_ns=timestamp_ns
            det["velocity"]=[float(track.kf.statePost[i,0]) for i in range(4,8)]
            output.append(det)

        for det in [high[i] for i in unmatched_high]:
            if det["confidence"] >= self.new_thresh:
                track = _Track(det)
                self.tracks.append(track)
                det["trackId"] = track.id
                track.last_seen_ns=timestamp_ns
                det["velocity"]=[0.,0.,0.,0.]
                output.append(det)

        # Standard ByteTrack: an unmatched low-score box is not promoted to
        # a new track. It may still be a detector output, but it has no stable
        # identity for the 3D scene yet.
        self.tracks = [t for t in self.tracks
                       if timestamp_ns is not None or t.time_since_update <= self.buffer]
        return output
