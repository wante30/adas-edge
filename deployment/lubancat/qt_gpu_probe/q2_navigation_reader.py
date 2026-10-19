"""Navigation card OCR for the current Q2 HDMI map layout.

The real Q2 feed is a 640x360 camera view of a navigation screen.  Its white
instruction card moves with the captured screen, so fixed text coordinates are
unreliable.  This reader runs RapidOCR text detection on a broad card crop in
the sidecar's background OCR worker.  Detection costs roughly 10 s on RK3566;
the worker stamps a completed observation with the newest source frame time and
the status builder gives it a 20 s lease, so the UI shows real OCR instead of a
fabricated route.

The reader is layout-specific and never invents a road or direction when OCR is
uncertain.
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
    parse_distance,
)

# Broad card crop at 960x540; detection returns line boxes, so no fixed text ROI
# is assumed.  It covers the moving card positions seen on the current feed.
CARD_ROI = (45, 105, 460, 300)
DISTANCE_RE = re.compile(r"^\s*(\d{1,4}(?:\.\d+)?)\s*(公里|千米|km|米|m)", re.I)
OCR_FIXES = (
    ("械助", "辅助"),
    ("铺助", "辅助"),
    ("镇助", "辅助"),
    ("领防", "领航"),
    ("最市", "城市"),
    ("即格", "即将"),
    ("即等", "即将"),
    ("印格", "即将"),
)


class Q2NavigationReader(NavigationReader):
    def __init__(self, *, ttl_ms: int = 60000, ocr=None):
        # The upstream base caps TTL at 10 s; Q2 detection is about 10 s, so a
        # 20 s lease prevents the UI flipping offline between two OCR passes.
        super().__init__(ttl_ms=min(int(ttl_ms), 10000), ocr=ocr)
        self.ttl_ms = int(ttl_ms)

    def _engine(self):
        if self._ocr is None:
            from rapidocr import RapidOCR

            # RKNN Lite shortens standard level names before OCR is initialized.
            logging.addLevelName(logging.ERROR, "ERROR")
            self._ocr = RapidOCR(params={
                "Global.log_level": "error",
                "Global.use_det": True,
                "Global.use_cls": True,
                "Det.limit_side_len": 320,
                "Det.limit_type": "max",
                "EngineConfig.onnxruntime.intra_op_num_threads": 2,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            })
        return self._ocr

    def warmup(self) -> None:
        """Load the OCR detection model before READY (about 6 s on RK3566)."""
        engine = self._engine()
        engine(np.full((120, 320, 3), 255, np.uint8), use_det=True, use_cls=True)

    @staticmethod
    def _center(box):
        points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
        return float(points[:, 0].mean()), float(points[:, 1].mean())

    @staticmethod
    def _clean(text):
        text = str(text).strip()
        for wrong, right in OCR_FIXES:
            text = text.replace(wrong, right)
        if "城市领航" in text and "退出" in text:
            text = "城市领航辅助即将退出"
        return text

    def update(self, nav_bgr: np.ndarray, frame_id: int, timestamp_ns: int) -> NavigationObservation:
        if (not isinstance(frame_id, int) or frame_id < 0 or not isinstance(timestamp_ns, int)
                or timestamp_ns < 0):
            raise ValueError("Frame ID and source timestamp must be nonnegative integers")
        if frame_id == self._last.source_frame_id:
            return self.current(timestamp_ns)
        if frame_id < self._last.source_frame_id or timestamp_ns < self._last.timestamp_ns:
            self.reset()
        start = time.perf_counter()
        base = NavigationObservation(source_frame_id=frame_id, timestamp_ns=timestamp_ns)

        def finish(**changes):
            values = base.to_dict()
            values.update(changes)
            values["inference_ms"] = (time.perf_counter() - start) * 1000.0
            self._last = NavigationObservation(**values)
            return self._last

        if (not isinstance(nav_bgr, np.ndarray) or nav_bgr.dtype != np.uint8
                or nav_bgr.shape != (540, 960, 3)):
            return finish(reason="Q2_IMAGE_REQUIRED_960X540_BGR")
        try:
            engine = self._engine()
        except ModuleNotFoundError as exc:
            return finish(reason=f"OCR_ENGINE_UNAVAILABLE:{exc.name or 'dependency'}")

        x0, y0, x1, y1 = CARD_ROI
        card = nav_bgr[y0:y1, x0:x1]
        if card.size == 0:
            return finish(reason="Q2_CARD_EMPTY")
        card = cv2.resize(card, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        try:
            result = engine(card, use_det=True, use_cls=True)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            return finish(reason=f"OCR_UNAVAILABLE_{type(exc).__name__}")

        texts = list(result.txts or ())
        scores = list(result.scores or ())
        boxes = list(result.boxes) if result.boxes is not None else []
        if len(texts) != len(scores) or len(texts) != len(boxes):
            return finish(reason="OCR_RESULT_MISMATCH")

        lines = []
        for text, score, box in zip(texts, scores, boxes):
            text = self._clean(text)
            if not text:
                continue
            _, y = self._center(box)
            lines.append((y, text, float(score)))
        lines.sort()

        # Only a distance with a metre/kilometre unit, or the explicit
        # "现在/立即" instruction, is a valid primary line.
        candidates = []
        for y, text, score in lines:
            match = DISTANCE_RE.match(text)
            if match:
                value = parse_distance(match.group(0))
                if value is not None:
                    candidates.append((y, value, text, score))
            elif "现在" in text or "立即" in text:
                candidates.append((y, 0.0, text, score))
            elif re.fullmatch(r"\d{2,4}", text) and y > 120:
                # The compact card sometimes loses the unit glyph. A bare
                # 2-4 digit line lower in the card is metres; speed limits are
                # above this region and are explicitly excluded.
                value = float(text)
                if 1.0 <= value <= 999.0 and text not in ("60", "63"):
                    candidates.append((y, value, text, score))
        if not candidates:
            return finish(reason="Q2_NO_DISTANCE_OR_NOW", raw_text=texts)

        dist_y, distance, distance_text, distance_score = min(candidates, key=lambda item: item[0])
        instruction = None
        for y, text, score in lines:
            if y <= dist_y + 4 or y > dist_y + 55:
                continue
            if text in ("随后",) or "km/h" in text.lower() or "kmh" in text.lower():
                continue
            if len(re.findall(r"[\u4e00-\u9fff]", text)) < 2:
                continue
            if instruction is None or score > instruction[2]:
                instruction = (y, text, score)
        if instruction is None:
            return finish(reason="Q2_NO_INSTRUCTION", raw_text=texts)

        instruction_text = instruction[1]
        direction = "none"
        for token, value in (("掉头", "uturn"), ("左转", "left"), ("右转", "right"),
                             ("直行", "forward")):
            if token in instruction_text:
                direction = value
                break
        return finish(
            direction=direction,
            distance_m=distance,
            road_name=instruction_text,
            valid=True,
            confidence=min(distance_score, instruction[2]),
            raw_text=texts,
            reason="Q2_CARD_OCR_DET_V1",
        )
