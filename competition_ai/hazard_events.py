"""Experimental front-camera events from real, source-timed detections.

Image expansion is a collision cue, not measured lead-car deceleration.
The displayed path is an explicit image-space assumption, not detected lanes.
Vehicle traffic lights never establish a pedestrian signal violation.
"""
from dataclasses import dataclass, field
import math
import time

import cv2
import numpy as np

from .image_motion import BackgroundMotion, ImageMotion
from .image_tracks import ImageTracker, iou

VEHICLES = {'car', 'truck', 'bus'}
DEFAULT_PATH = ((.48, .48), (.52, .48), (.73, .88), (.27, .88))


@dataclass(frozen=True)
class HazardObservation:
    timestamp_ns: int | None = None
    events: tuple[dict, ...] = ()
    tracks: tuple[dict, ...] = ()
    path: tuple = DEFAULT_PATH
    motion: ImageMotion = field(default_factory=ImageMotion)
    reason: str = 'disabled'
    lead_id: int | None = None
    sudden_braking: str = 'UNKNOWN_NO_EGO_SPEED_OR_CALIBRATED_RANGE'
    pedestrian_red_violation: str = 'UNKNOWN_NO_ASSOCIATED_PEDESTRIAN_SIGNAL'

    @property
    def pedestrian_hazard(self):
        return any(e['kind'] == 'PEDESTRIAN_CROSSING_PATH' for e in self.events)

    @property
    def obstacle_hazard(self):
        return any(e['kind'] == 'LEAD_RAPID_CLOSING' for e in self.events)


def trend(history, index, *, corrected=False):
    data = list(history)
    ts = np.array([(row[0]-data[0][0])/1e9 for row in data])
    values = np.array([row[index]-(row[4] if corrected else 0.) for row in data])
    if len(ts) < 6 or ts[-1] < .5:
        return 0., 0.
    centered = ts-ts.mean()
    slope = float(centered @ (values-values.mean()) / max(float(centered @ centered), 1e-9))
    residual = values-(values.mean()+slope*centered)
    variation = float(np.square(values-values.mean()).sum())
    fit = 1.-float(np.square(residual).sum())/variation if variation > 1e-9 else 0.
    return slope, fit


def paired_red_lamps(frame, box):
    x1, y1, x2, y2 = map(round, box)
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.shape[0] < 25 or crop.shape[1] < 40:
        return 0.
    crop = cv2.resize(crop, (100, 64))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 168)) & (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 165)
    left, right = float(red[25:51, 5:35].mean()), float(red[25:51, 65:95].mean())
    return min(left, right) if min(left, right) > max(left, right)*.3 else 0.


class HazardEvents:
    def __init__(self, path=DEFAULT_PATH):
        polygon = np.asarray(path, dtype=np.float32)
        if polygon.shape != (4, 2) or not np.isfinite(polygon).all() or (polygon < 0).any() or (polygon > 1).any():
            raise ValueError('Path must contain four finite normalized image points')
        if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < .005:
            raise ValueError('Path must be a convex, nondegenerate polygon')
        self.path = tuple(tuple(float(v) for v in point) for point in polygon)
        self.polygon = polygon
        self.timings = []
        self.reset()

    def reset(self):
        self.tracker, self.background = ImageTracker(), BackgroundMotion()
        self.last_frame, self.last_ns, self.blocked_until = None, None, 0
        self.confirmations = {}
        self.observation = HazardObservation(path=self.path, reason='warming_up')

    def inside(self, x, y):
        return cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False) >= 0

    def update(self, frame, detections, frame_id, timestamp_ns, *, motion=None):
        if self.last_frame == frame_id and self.last_ns == timestamp_ns:
            return self.observation
        if not isinstance(timestamp_ns, int) or timestamp_ns < 0:
            raise ValueError('Invalid source timestamp')
        if self.last_ns is not None and (timestamp_ns <= self.last_ns or timestamp_ns-self.last_ns > 300_000_000):
            self.reset()
        started = time.perf_counter()
        self.last_frame, self.last_ns = frame_id, timestamp_ns
        motion = motion if motion is not None else self.background.update(frame, detections, timestamp_ns)
        height, width = frame.shape[:2]
        items = []
        riders = [d['bbox'] for d in detections if d['class_name'] in ('motorcycle', 'bicycle')]
        for item in detections:
            name, box = item['class_name'], item['bbox']
            if name not in VEHICLES | {'person'} or item.get('confidence', 0.) < .4:
                continue
            if len(box) != 4 or not all(math.isfinite(v) for v in box):
                continue
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                continue
            # Large shallow bonnet detections and people riding bikes are not
            # evidence of a lead vehicle or a pedestrian on foot.
            if name in VEHICLES and y2 > height*.84 and (x2-x1)/(y2-y1) > 2.7:
                continue
            if name == 'person' and any(iou(box, rider) > .18 for rider in riders):
                continue
            items.append(item)
        tracks = self.tracker.update(items, timestamp_ns, frame.shape, motion)
        if not motion.stable:
            self.blocked_until = timestamp_ns+500_000_000
            self.confirmations.clear()
            for track in self.tracker.tracks.values():
                track.history.clear()
                track.lamp_history.clear()
        trusted = motion.stable and timestamp_ns >= self.blocked_until
        candidates = [t for t in tracks if t.name in VEHICLES and self.inside(t.history[-1][1], t.box[3]/height)] if trusted else []
        lead = max(candidates, key=lambda t: t.box[3], default=None)
        events, evidence, active_confirmations = [], [], {}
        for track in tracks:
            history = list(track.history)
            info = {'id': track.track_id, 'class_name': track.name, 'bbox': track.box,
                    'observations': len(history), 'in_assumed_path': False}
            if history:
                _, x, foot, size, _ = history[-1]
                info['in_assumed_path'] = self.inside(x, foot)
            if trusted and len(history) >= 6:
                lateral, lateral_fit = trend(history, 1, corrected=True)
                expansion, expansion_fit = trend(history, 3)
                growth = size/max(history[0][3], 1e-9)
                tau = size/expansion if expansion > 0 else None
                info.update(lateral_frame_widths_per_s=lateral, expansion_fit=expansion_fit,
                            visual_tau_s=tau, apparent_scale_growth=growth)
                kind = None
                if (track is lead and foot > .57 and size > .035 and tau is not None
                        and tau < 2.5 and expansion_fit > .85 and growth > 1.25):
                    kind = 'LEAD_RAPID_CLOSING'
                if (track.name == 'person' and info['in_assumed_path'] and foot > .6
                        and abs(lateral) > .025 and lateral_fit > .7):
                    kind = 'PEDESTRIAN_CROSSING_PATH'
                if kind:
                    key = (track.track_id, kind)
                    count = self.confirmations.get(key, 0)+1
                    active_confirmations[key] = count
                    if count >= 2:
                        events.append({'kind': kind, 'track_id': track.track_id, 'bbox': track.box,
                                       'source': 'IMAGE_TRAJECTORY_EXPERIMENTAL',
                                       'visual_tau_s': tau if kind == 'LEAD_RAPID_CLOSING' else None})
            if track is lead:
                lamp = paired_red_lamps(frame, track.box)
                track.lamp_history.append((timestamp_ns, lamp))
                before = [v for ts, v in track.lamp_history if .4 <= (timestamp_ns-ts)/1e9 <= 1.2]
                info['paired_red_lamp_fraction'] = lamp
                # Lamp appearance alone has no effect on the brake decision.
                if before and lamp > .025 and lamp-min(before) > .025:
                    info['brake_lamp_rise_candidate'] = True
                    if any(e['kind'] == 'LEAD_RAPID_CLOSING' and e['track_id'] == track.track_id for e in events):
                        events.append({'kind': 'SUSPECTED_BRAKING_WITH_CLOSING', 'track_id': track.track_id,
                                       'bbox': track.box, 'source': 'PAIRED_RED_LAMP_RISE_PLUS_IMAGE_EXPANSION'})
            evidence.append(info)
        self.confirmations = active_confirmations
        self.observation = HazardObservation(timestamp_ns, tuple(events), tuple(evidence), self.path, motion,
                                             'tracking' if trusted else 'camera_motion_or_unconfirmed_background',
                                             lead.track_id if lead else None)
        self.timings.append((time.perf_counter()-started)*1000)
        return self.observation

    def current(self, now_ns):
        observed = self.observation
        if observed.timestamp_ns is None or not 0 <= now_ns-observed.timestamp_ns < 250_000_000:
            return HazardObservation(path=self.path, reason='observation_expired')
        return observed
