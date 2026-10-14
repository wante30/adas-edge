"""Bounded, conservative same-class image tracks; no metric 3D positions."""
from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np


def iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection/max(union, 1e-9)


@dataclass
class ImageTrack:
    track_id: int
    name: str
    box: list
    timestamp_ns: int
    history: deque = field(default_factory=lambda: deque(maxlen=18))
    lamp_history: deque = field(default_factory=lambda: deque(maxlen=18))

    def observe(self, item, timestamp_ns, shape, motion):
        self.box, self.timestamp_ns = list(item['bbox']), timestamp_ns
        height, width = shape[:2]
        x1, y1, x2, y2 = self.box
        # Correct apparent lateral displacement for background translation.
        offset = self.history[-1][4] if self.history else 0.
        zoom = self.history[-1][5] if self.history else 0.
        if self.history and motion.valid:
            dt = (timestamp_ns-self.history[-1][0])/1e9
            offset += motion.dx*dt
            zoom += motion.scale_rate*dt
        self.history.append((timestamp_ns, (x1+x2)/2/width, y2/height,
                             math.sqrt((x2-x1)*(y2-y1))/width, offset, zoom))


class ImageTracker:
    def __init__(self):
        self.tracks, self.next_id = {}, 1

    def update(self, items, timestamp_ns, shape, motion):
        self.tracks = {k: t for k, t in self.tracks.items()
                       if 0 < timestamp_ns-t.timestamp_ns <= 250_000_000}
        tracks = list(self.tracks.values())
        scores = np.zeros((len(tracks), len(items)))
        for i, track in enumerate(tracks):
            for j, item in enumerate(items):
                if track.name == item['class_name']:
                    scores[i, j] = iou(track.box, item['bbox'])
        assigned = set()
        current = []
        for i, track in enumerate(tracks):
            if not items:
                break
            j = int(scores[i].argmax())
            score = scores[i, j]
            # Mutual best match and a margin reject ambiguous crowd/occlusion.
            others = np.delete(scores[:, j], i)
            alternatives = np.delete(scores[i], j)
            if (score < .3 or int(scores[:, j].argmax()) != i or j in assigned
                    or (len(others) and score-float(others.max()) < .08)
                    or (len(alternatives) and score-float(alternatives.max()) < .08)):
                continue
            track.observe(items[j], timestamp_ns, shape, motion)
            assigned.add(j)
            current.append(track)
        for j, item in enumerate(items):
            if j not in assigned:
                track = ImageTrack(self.next_id, item['class_name'], [], timestamp_ns)
                self.next_id += 1
                track.observe(item, timestamp_ns, shape, motion)
                self.tracks[track.track_id] = track
                current.append(track)
        return current
