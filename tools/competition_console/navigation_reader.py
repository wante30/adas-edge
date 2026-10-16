"""Pixel-only reader for the rectified 112739 navigation-screen layout.

This is a video-layout-specific reader, not a general map/navigation engine.
The top-left instruction arrow and its adjacent distance are required together;
map polylines and the lane-choice arrows below the card are not inputs. Frame
IDs/timestamps only identify observations and expire them, never select answers.

OCR API: https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/usage/
RapidOCR 3.9.2 uses PP-OCRv6 small recognition on ONNX Runtime CPU. The detector
is not run: a small, geometrically located instruction line is recognized.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import re
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class NavigationObservation:
    direction: str = 'none'
    distance_m: float | None = None
    road_name: str = ''
    valid: bool = False
    confidence: float = 0.
    reason: str = 'NO_NAVIGATION_OBSERVATION'
    source_frame_id: int = -1
    timestamp_ns: int = 0
    raw_text: list[str] = field(default_factory=list)
    inference_ms: float = 0.

    def to_dict(self) -> dict:
        return asdict(self)


def parse_distance(text: str) -> float | None:
    """Parse only a leading distance from the calibrated metre instruction line."""
    text = text.strip().replace(' ', '')
    match = re.match(r'^(\d{1,4}(?:\.\d+)?)(公里|千米|km|米|m)?(?:[.。·,，])?(.*)$', text, re.I)
    if not match:
        return None
    number, unit, rest = match.groups()
    # Avoid interpreting a clock, ETA or an unfinished decimal as a distance.
    if rest.startswith((':', '：', '/', '%')) or any(token in rest for token in ('分钟', '小时', '到达')):
        return None
    if rest and not re.fullmatch(r'[\u4e00-\u9fff]+', rest):
        return None  # e.g. a fifth digit must not be silently truncated
    if '.' in number and unit is None:
        return None
    value = float(number) * (1000. if unit and unit.lower() in ('公里', '千米', 'km') else 1.)
    return value if 0 <= value <= 20000 else None


def classify_arrow(mask: np.ndarray) -> tuple[str, float]:
    """Classify an isolated dark instruction glyph, not an arbitrary map crop."""
    ink = mask > 0
    h, w = ink.shape
    if h < 16 or w < 6 or not .2 < w/h < 1.7 or not .12 < ink.mean() < .65:
        return 'none', 0.
    top = np.where(ink[:max(1, h//2)])[1]
    bottom = np.where(ink[-max(1, h//4):])[1]
    if len(top) < 5 or len(bottom) < 5:
        return 'none', 0.
    band = ink[round(h*.4):round(h*.78)]
    total = int(band.sum())
    if total < 8:
        return 'none', 0.
    left_mass = float(band[:, :max(1, w//3)].sum()/total)
    right_mass = float(band[:, 2*w//3:].sum()/total)
    top_center, bottom_center = float(top.mean()/w), float(bottom.mean()/w)
    shift = top_center-bottom_center
    # A U-turn has two legs in its middle, whereas a left turn has one shaft.
    # Verify a gap between them so a filled rectangle cannot become a U-turn.
    column_mass = band.sum(axis=0)
    middle_gap = float(column_mass[max(1, w//3):max(2, 2*w//3)].min(initial=h))
    if w/h < 1.05 and left_mass > .22 and right_mass > .22 and middle_gap < len(band)*.3 and bottom_center > .58:
        return 'uturn', min(.98, .80+min(left_mass, right_mass)*.35)
    if shift > .20 and bottom_center < .38 and left_mass > .55 and right_mass < .18:
        return 'right', min(.98, .75+shift*.4)
    if shift < -.19 and bottom_center > .62 and right_mass > .55 and left_mass < .18:
        return 'left', min(.98, .75-shift*.4)
    # Straight is supported geometrically but absent from the reviewed video
    # samples; its regression checks are synthetic and must be reported as such.
    if abs(shift) < .12 and .35 < bottom_center < .65:
        top_span = np.ptp(top)+1
        bottom_span = np.ptp(bottom)+1
        if top_span > bottom_span*1.8 and left_mass < .2 and right_mass < .2:
            return 'forward', .85
    return 'none', 0.


class NavigationReader:
    """Caller schedules updates, e.g. 2 Hz; current() checks the source-clock TTL."""
    def __init__(self, *, ttl_ms: int = 1250, ocr=None):
        if not isinstance(ttl_ms, int) or not 1 <= ttl_ms <= 10000:
            raise ValueError('ttl_ms must be 1..10000')
        self.ttl_ms = ttl_ms
        self._ocr = ocr
        self.reset()

    def reset(self) -> None:
        self._last = NavigationObservation()

    def current(self, timestamp_ns: int) -> NavigationObservation:
        age = timestamp_ns-self._last.timestamp_ns
        if self._last.valid and (age < 0 or age >= self.ttl_ms*1_000_000):
            return replace(self._last, direction='none', distance_m=None, road_name='',
                           valid=False, confidence=0., reason='NAVIGATION_EXPIRED_OR_CLOCK_ROLLBACK')
        return self._last

    def _engine(self):
        if self._ocr is None:
            from rapidocr import RapidOCR
            self._ocr = RapidOCR(params={
                'Global.log_level': 'warning', 'Global.use_det': False, 'Global.use_cls': False,
                'EngineConfig.onnxruntime.intra_op_num_threads': 2,
                'EngineConfig.onnxruntime.inter_op_num_threads': 1,
            })
        return self._ocr

    @staticmethod
    def _arrow(nav: np.ndarray):
        # Fixed geometry of the rectified screen, not source-video coordinates.
        # Main instruction is above the row of lane arrows and left of distance.
        crop = nav[:180, :115]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if float(np.percentile(gray, 80)) < 65 or float(np.std(gray)) < 12:
            return None
        candidates = []
        # Photographed screen brightness and card position change. Local
        # contrast isolates glyphs without relying on a frame/time schedule.
        for offset in (17, 27):
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                           cv2.THRESH_BINARY_INV, 41, offset)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
            for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
                if (area < 40 or not 18 <= h <= 65 or x <= 0 or y <= 12 or y+h >= len(gray)
                        or x >= 90 or x+w >= gray.shape[1]):
                    continue
                mask = np.uint8(labels[y:y+h, x:x+w] == label)*255
                direction, confidence = classify_arrow(mask)
                if direction != 'none':
                    box = (int(x), int(y), int(w), int(h))
                    line = NavigationReader._digits(nav, box)
                    if line is not None:
                        candidates.append((int(area), float(confidence), direction, box, line))
        # A larger primary instruction wins over the smaller lane-choice row.
        # Every candidate must also have a separately isolated numeric line.
        return sorted(candidates, key=lambda item: item[0], reverse=True)[0] if candidates else None

    @staticmethod
    def _digits(nav: np.ndarray, box):
        """Crop only the large contiguous digits, excluding tiny unit/road text.

        OCR of the whole compact card can turn '14 米路名' into '1440'. This
        component-height gate prevents the road name entering recognition.
        Bare digits have metre semantics only in this reviewed video layout.
        """
        x, y, w, h = box
        left, top = x+w+5, max(0, y-5)
        crop = nav[top:min(180, y+round(h*.7)+13), left:250]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY_INV, 31, 13)
        _, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        components = [(int(a), int(b), int(c), int(d)) for a,b,c,d,area in stats[1:]
                      if area >= 16 and max(10, h*.27) <= d <= h*1.1
                      and c >= 2 and c <= d*1.8 and a > 0 and b > 0
                      and b+d < len(gray) and a+c < gray.shape[1]]
        starts = [s for s in components if s[0] < 30]
        if not starts:
            return None
        first = min(starts, key=lambda s: s[0])
        # Never silently drop a clipped leading digit and recognize its suffix
        # (observed failures: 126 -> 26 and 24 -> 4). A taller glyph touching
        # the crop edge before the chosen first character invalidates the line.
        for a,b,c,d,area in stats[1:]:
            if (a < first[0] and area >= 12 and d >= max(8, first[3]*.65)
                    and c <= d*1.8 and b < first[1]+first[3] and b+d > first[1]
                    and (a == 0 or b == 0 or b+d >= len(gray))):
                return None
        # The number is in the upper part of the instruction, not its road name.
        if first[1]+first[3]/2 > h*.95:
            return None
        group = [first]
        for item in sorted(components, key=lambda s: s[0]):
            if item[0] <= first[0]:
                continue
            # Text on the road-name row can have an x coordinate between two
            # digits. Skip that other row instead of truncating the number.
            if abs(item[1]+item[3]/2-first[1]-first[3]/2) > first[3]*.35:
                continue
            previous = group[-1]
            gap = item[0]-(previous[0]+previous[2])
            if gap > max(6, first[3]*.45):
                break
            if (item[3] < first[3]*.8 or item[3] > first[3]*1.25
                    or abs(item[1]+item[3]/2-first[1]-first[3]/2) > first[3]*.25):
                if item[1] <= first[1]+first[3]*.25:
                    return None  # adjacent top-aligned ink may be a lost digit
                break
            group.append(item)
        gx, gy = min(s[0] for s in group), min(s[1] for s in group)
        right, bottom = max(s[0]+s[2] for s in group), max(s[1]+s[3] for s in group)
        if right-gx > 110:
            return None
        return crop[max(0, gy-3):min(len(crop), bottom+3), max(0, gx-3):right+3]

    def update(self, nav_bgr: np.ndarray, frame_id: int, timestamp_ns: int) -> NavigationObservation:
        if (not isinstance(frame_id, int) or frame_id < 0 or not isinstance(timestamp_ns, int)
                or timestamp_ns < 0):
            raise ValueError('Frame ID and source timestamp must be nonnegative integers')
        if frame_id == self._last.source_frame_id:
            return self.current(timestamp_ns)  # duplicate cannot re-infer or renew
        if frame_id < self._last.source_frame_id or timestamp_ns < self._last.timestamp_ns:
            self.reset()
        start = time.perf_counter()
        base = NavigationObservation(source_frame_id=frame_id, timestamp_ns=timestamp_ns)
        def finish(**changes):
            self._last = replace(base, inference_ms=(time.perf_counter()-start)*1000, **changes)
            return self._last
        if (not isinstance(nav_bgr, np.ndarray) or nav_bgr.dtype != np.uint8
                or nav_bgr.shape != (540, 960, 3)):
            return finish(reason='RECTIFIED_NAVIGATION_IMAGE_REQUIRED_960X540_BGR')
        found = self._arrow(nav_bgr)
        if found is None:
            return finish(reason='NO_TOP_INSTRUCTION_ARROW_OR_NOT_NAVIGATION_VIEW')
        _, arrow_score, direction, _, number_line = found
        if number_line.size == 0:
            return finish(reason='NO_DISTANCE_REGION')
        # Cubic upsampling helps blurred photographed digits. No temporal value
        # interpolation, map coordinate, frame index or route schedule is used.
        enlarged = cv2.resize(number_line, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        try:
            result = self._engine()(enlarged, use_det=False, use_cls=False)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            return finish(reason=f'OCR_UNAVAILABLE_{type(exc).__name__}')
        texts = list(result.txts or ())
        scores = list(result.scores or ())
        if len(texts) != 1 or len(scores) != 1 or scores[0] < .80:
            return finish(reason='DISTANCE_OCR_UNCONFIRMED', raw_text=texts)
        distance = parse_distance(texts[0])
        if distance is None:
            return finish(reason='DISTANCE_TEXT_NOT_VALID', raw_text=texts)
        # No road name is filled from a known route: low-resolution text is left
        # empty unless a future separately validated reader provides it.
        return finish(direction=direction, distance_m=distance, valid=True,
                      confidence=min(arrow_score, float(scores[0])), raw_text=texts,
                      reason='PIXEL_ARROW_AND_OCR_DISTANCE_VIDEO_LAYOUT_V1')
