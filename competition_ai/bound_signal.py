"""Track a user/scene-bound lamp housing; colours always come from pixels."""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .scenario import Light


def classify_horizontal_head(frame, bbox) -> Light:
    """Colour measurement for an explicitly bound horizontal R/Y/G housing.

    The layout is a scene calibration, not an inferred lamp type. Restricting
    colour evidence to bulb slots keeps surrounding leaves out of the vote.
    Unsupported/ambiguous layouts return UNKNOWN.
    """
    x1,y1,x2,y2 = [round(x) for x in bbox]
    roi = frame[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)]
    if roi.size == 0 or roi.shape[1]/max(1,roi.shape[0]) < 2:
        return Light.UNKNOWN
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h,s,v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    if (v < 120).mean() < .25:
        return Light.UNKNOWN
    yy,xx = np.indices(h.shape)
    height,width = h.shape
    colours = []
    for index,colour in enumerate((Light.RED, Light.YELLOW, Light.GREEN)):
        cx,cy = width*(.22+.28*index), height*.5
        slot = ((xx-cx)/max(2,width*.12))**2 + ((yy-cy)/max(2,height*.28))**2 <= 1
        hue = ((h <= 12) | (h >= 165)) if colour == Light.RED else ((h >= 15) & (h <= 35)) if colour == Light.YELLOW else ((h >= 40) & (h <= 95))
        mask = (slot & hue & (s >= 80) & (v >= 160)).astype(np.uint8)
        count,_,stats,_ = cv2.connectedComponentsWithStats(mask,8)
        largest = int(stats[1:,cv2.CC_STAT_AREA].max()) if count > 1 else 0
        if largest >= max(4, int(slot.sum()*.08)):
            colours.append(colour)
    return colours[0] if len(colours) == 1 else Light.UNKNOWN


@dataclass(frozen=True)
class SignalObservation:
    light: Light = Light.UNKNOWN
    raw_light: Light = Light.UNKNOWN
    bbox: tuple[float, float, float, float] | None = None
    source_frame_id: int | None = None
    timestamp_ns: int | None = None
    associated: bool = False
    tracking_score: float = 0.
    reason: str = "unbound"
    source: str = "BOUND_HORIZONTAL_RYG_SLOT_HSV"


class BoundSignal:
    """Local multiscale housing tracking, with immediate rejection on loss.

    This does not infer traffic rules or which lane a housing controls. The
    caller supplies an explicit competition-scene binding. No ground-truth
    colour or time schedule is consumed by this tracker.
    """

    def __init__(self, frame, bbox, *, scale_width=960, confirm_green=2):
        height, width = frame.shape[:2]
        self.scale = min(1., scale_width/width)
        self.shape = (round(width*self.scale), round(height*self.scale))
        box = np.asarray(bbox, dtype=np.float64)
        if (box.shape != (4,) or not np.isfinite(box).all() or box[0] < 0 or box[1] < 0
                or box[2] > width or box[3] > height or box[2]-box[0] < 12 or box[3]-box[1] < 8):
            raise ValueError("Select a complete visible lamp housing inside the video")
        self.box = box*self.scale
        self.aspect = (box[2]-box[0])/(box[3]-box[1])
        self.gray = self._gray(frame)
        self.template = self._patch(self.gray)
        self.active = self.template is not None and float(self.template.std()) >= 8
        self.last_frame = None
        self.candidate, self.confirmed = Light.UNKNOWN, 0
        self.confirm_green = max(1, int(confirm_green))
        self.observation = SignalObservation(bbox=tuple(float(x) for x in box),
                                             associated=self.active, reason="awaiting_colour" if self.active else "no_housing_features")

    def _gray(self, frame):
        return cv2.cvtColor(cv2.resize(frame, self.shape), cv2.COLOR_BGR2GRAY)

    def _patch(self, gray):
        x1,y1,x2,y2 = [round(x) for x in self.box]
        crop = gray[max(0,y1):min(gray.shape[0],y2), max(0,x1):min(gray.shape[1],x2)]
        if crop.size == 0:
            return None
        return cv2.resize(crop, (96, 40))

    def _lost(self, reason):
        self.active = False
        self.observation = SignalObservation(bbox=tuple(float(x/self.scale) for x in self.box), reason=reason)
        return self.observation

    def update(self, frame, frame_id: int, timestamp_ns: int, *, classify=True):
        if self.last_frame == frame_id:
            return self.observation  # pause/buttons cannot confirm the same image twice
        if self.last_frame is not None and frame_id < self.last_frame:
            return self._lost("replay_jump_requires_rebinding")
        if not self.active:
            return self.observation
        gray = self._gray(frame)
        score = 1.
        if self.last_frame is not None:
            x1,y1,x2,y2 = self.box
            width,height = x2-x1,y2-y1
            sx1,sy1 = max(0,round(x1-width*.6)), max(0,round(y1-height))
            sx2,sy2 = min(self.shape[0],round(x2+width*.6)), min(self.shape[1],round(y2+height))
            search = gray[sy1:sy2,sx1:sx2]
            best = None
            for zoom in (.96,1.,1.04):
                tw = max(8,round(width*zoom))
                th = max(5,round(tw/self.aspect))
                if th > search.shape[0] or tw > search.shape[1]:
                    continue
                template = cv2.resize(self.template,(tw,th))
                _,value,_,position = cv2.minMaxLoc(cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED))
                if best is None or value > best[0]:
                    best = (value,position,tw,th)
            if best is None or not np.isfinite(best[0]) or best[0] < .55:
                return self._lost("housing_appearance_mismatch")
            score,position,tw,th = best
            x1,y1 = sx1+position[0],sy1+position[1]
            self.box = np.array([x1,y1,x1+tw,y1+th],dtype=float)
        self.gray, self.last_frame = gray, frame_id
        box = tuple(float(x/self.scale) for x in self.box)
        if not classify:
            self.observation = replace(self.observation, bbox=box, tracking_score=score)
            return self.observation
        raw = classify_horizontal_head(frame, box)
        self.confirmed = self.confirmed+1 if raw == self.candidate else 1
        self.candidate = raw
        # RED/YELLOW stop immediately. GREEN requires distinct source images.
        light = raw if raw != Light.GREEN or self.confirmed >= self.confirm_green else Light.UNKNOWN
        self.observation = SignalObservation(light, raw, box, frame_id, timestamp_ns, True, score,
                                             "observed" if light != Light.UNKNOWN else "colour_unknown_or_confirming")
        return self.observation

    def current(self, now_ns: int, max_age_ms=400.) -> SignalObservation:
        observed = self.observation
        if observed.timestamp_ns is None:
            return observed
        if not 0 <= now_ns-observed.timestamp_ns < max_age_ms*1_000_000:
            return replace(observed, light=Light.UNKNOWN, associated=False, reason="observation_expired")
        return observed
