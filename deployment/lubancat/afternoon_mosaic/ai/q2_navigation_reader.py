"""Lightweight Q2 navigation OCR without full-card text detection.

The Q2 feed is a photographed navigation screen.  The white instruction card
moves with the source screen, so the card is located with OpenCV, the arrow is
classified geometrically, and only the distance digits are passed to RapidOCR
in recognition-only mode.
"""
from __future__ import annotations

import logging
import re
import time

import cv2
import numpy as np

from tools.competition_console.navigation_reader import (
    NavigationObservation,
    NavigationReader,
    classify_arrow,
    parse_distance,
)

CARD_ROI = (45, 105, 460, 300)
FALLBACK_CARD = (45, 105, 415, 195)
ARROW_MIN_CONFIDENCE = 0.55


class Q2NavigationReader(NavigationReader):
    def __init__(self, *, ttl_ms: int = 2000, ocr=None):
        super().__init__(ttl_ms=min(max(int(ttl_ms), 500), 10000), ocr=ocr)
        self.ttl_ms = int(ttl_ms)

    def _engine(self):
        if self._ocr is None:
            from rapidocr import RapidOCR

            logging.addLevelName(logging.ERROR, "ERROR")
            self._ocr = RapidOCR(params={
                "Global.log_level": "error",
                "Global.use_det": False,
                "Global.use_cls": False,
                "EngineConfig.onnxruntime.intra_op_num_threads": 2,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            })
        return self._ocr

    def warmup(self) -> None:
        engine = self._engine()
        engine(np.full((32, 160, 3), 255, np.uint8),
               use_det=False, use_cls=False)

    @staticmethod
    def _locate_card(nav_bgr: np.ndarray):
        gray = cv2.cvtColor(nav_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(nav_bgr, cv2.COLOR_BGR2HSV)
        white = ((gray > 165) & (hsv[:, :, 1] < 70)).astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0
        height, width = nav_bgr.shape[:2]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if (area < 45000 or area > 130000
                    or w < 300 or w > 520 or h < 140 or h > 260
                    or not 1.4 <= w / max(1, h) <= 2.8
                    or x > width * 0.62 or y > height * 0.68):
                continue
            score = float(area) - abs((w / max(1, h)) - 1.9) * 1200.0
            if score > best_score:
                best_score = score
                best = (int(x), int(y), int(w), int(h))
        if best is not None:
            return best
        x0, y0, x1, y1 = CARD_ROI
        return (x0, y0, x1 - x0, y1 - y0)

    @staticmethod
    def _arrow_in_card(card: np.ndarray):
        height, width = card.shape[:2]
        roi_top = int(height * 0.18)
        roi_left = int(width * 0.02)
        roi = card[roi_top:int(height * 0.92),
                   roi_left:int(width * 0.48)]
        if roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        found = []
        for offset in (17, 27):
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV, 41, offset)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
            for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
                if (area < 45 or not 18 <= h <= 120 or not 10 <= w <= 115
                        or not 0.20 <= w / max(1, h) <= 1.75):
                    continue
                component = labels[y:y + h, x:x + w] == label
                mean_saturation = float(hsv[y:y + h, x:x + w, 1][component].mean())
                mean_value = float(hsv[y:y + h, x:x + w, 2][component].mean())
                # The blue navigation badge is not the instruction arrow.
                if mean_saturation > 120.0 or mean_value > 140.0:
                    continue
                mask = np.uint8(component) * 255
                direction, confidence = classify_arrow(mask)
                if direction == "none" or confidence < ARROW_MIN_CONFIDENCE:
                    continue
                found.append((int(area), float(confidence), direction,
                              (int(x + roi_left), int(y + roi_top), int(w), int(h))))
        if not found:
            return None
        return sorted(found, key=lambda item: (item[0], item[1]),
                      reverse=True)[0]

    def update(self, nav_bgr: np.ndarray, frame_id: int,
               timestamp_ns: int) -> NavigationObservation:
        if (not isinstance(frame_id, int) or frame_id < 0
                or not isinstance(timestamp_ns, int) or timestamp_ns < 0):
            raise ValueError("Frame ID and source timestamp must be nonnegative integers")
        if frame_id == self._last.source_frame_id:
            return self.current(timestamp_ns)
        if frame_id < self._last.source_frame_id or timestamp_ns < self._last.timestamp_ns:
            self.reset()
        start = time.perf_counter()
        base = NavigationObservation(
            source_frame_id=frame_id, timestamp_ns=timestamp_ns)
        card_locate_ms = 0.0
        arrow_ms = 0.0
        distance_ocr_ms = 0.0
        preprocess_ms = 0.0

        def finish(**changes):
            nonlocal card_locate_ms, arrow_ms, distance_ocr_ms, preprocess_ms
            values = base.to_dict()
            values.update(changes)
            total_ms = (time.perf_counter() - start) * 1000.0
            values["inference_ms"] = total_ms
            values["preprocess_ms"] = preprocess_ms
            values["engine_ms"] = distance_ocr_ms
            values["card_locate_ms"] = card_locate_ms
            values["arrow_ms"] = arrow_ms
            values["distance_ocr_ms"] = distance_ocr_ms
            values["ocr_timestamp_ns"] = time.monotonic_ns()
            self._last = NavigationObservation(**values)
            return self._last

        if (not isinstance(nav_bgr, np.ndarray) or nav_bgr.dtype != np.uint8
                or nav_bgr.shape != (540, 960, 3)):
            return finish(reason="Q2_IMAGE_REQUIRED_960X540_BGR")

        x, y, w, h = self._locate_card(nav_bgr)
        card_locate_ms = (time.perf_counter() - start) * 1000.0
        card = nav_bgr[y:y + h, x:x + w]
        if card.size == 0:
            return finish(reason="Q2_CARD_EMPTY")
        arrow_started = time.perf_counter()
        arrow = self._arrow_in_card(card)
        arrow_ms = (time.perf_counter() - arrow_started) * 1000.0
        preprocess_ms = card_locate_ms + arrow_ms
        if arrow is None:
            return finish(reason="Q2_ARROW_NOT_FOUND")
        _, arrow_conf, direction, box = arrow
        distance_started = time.perf_counter()
        left = min(card.shape[1], int(box[0] + box[2] + 5))
        top = max(0, int(box[1] - 12))
        right = min(card.shape[1], left + 170)
        bottom = min(card.shape[0], int(box[1] + box[3] + 12))
        number_line = card[top:bottom, left:right]
        if number_line.size == 0:
            return finish(reason="Q2_DISTANCE_REGION_NOT_FOUND")
        enlarged = cv2.resize(
            number_line, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        try:
            engine = self._engine()
            result = engine(enlarged, use_det=False, use_cls=False)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            return finish(reason=f"OCR_UNAVAILABLE_{type(exc).__name__}")
        distance_ocr_ms = (time.perf_counter() - distance_started) * 1000.0
        texts = list(result.txts or ())
        scores = list(result.scores or ())
        if len(texts) != 1 or len(scores) != 1 or scores[0] < 0.55:
            return finish(reason="Q2_DISTANCE_OCR_UNCONFIRMED", raw_text=texts)
        distance_text = texts[0].strip()
        distance = parse_distance(distance_text)
        if distance is None:
            numeric = re.match(r"^\s*(\d{1,4}(?:\.\d+)?)", distance_text)
            if numeric is not None:
                distance = parse_distance(numeric.group(1))
        if distance is None:
            return finish(reason="Q2_DISTANCE_TEXT_NOT_VALID", raw_text=texts)
        return finish(
            direction=direction,
            distance_m=distance,
            road_name="",
            valid=True,
            confidence=min(float(arrow_conf), float(scores[0])),
            raw_text=texts,
            reason="Q2_LIGHTWEIGHT_CARD_ARROW_OCR",
        )
