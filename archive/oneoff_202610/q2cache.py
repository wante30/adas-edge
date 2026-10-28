from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\q2_navigation_reader.py')
s=p.read_text(encoding='utf-8')
a=s.index('    def update(self, nav_bgr: np.ndarray, frame_id: int, timestamp_ns: int)')
new=r'''    def update(self, nav_bgr: np.ndarray, frame_id: int, timestamp_ns: int) -> NavigationObservation:
        if (not isinstance(frame_id, int) or frame_id < 0 or not isinstance(timestamp_ns, int)
                or timestamp_ns < 0):
            raise ValueError("Frame ID and source timestamp must be nonnegative integers")
        if frame_id == self._last.source_frame_id:
            return self.current(timestamp_ns)
        if frame_id < self._last.source_frame_id or timestamp_ns < self._last.timestamp_ns:
            self.reset()
        start = time.perf_counter()
        preprocess_ms = 0.0
        base = NavigationObservation(source_frame_id=frame_id, timestamp_ns=timestamp_ns)

        def finish(**changes):
            values = base.to_dict()
            values.update(changes)
            total_ms = (time.perf_counter() - start) * 1000.0
            values["inference_ms"] = total_ms
            values["preprocess_ms"] = preprocess_ms
            values["engine_ms"] = max(0.0, total_ms - preprocess_ms)
            values["ocr_timestamp_ns"] = time.monotonic_ns()
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
        preprocess_ms = (time.perf_counter() - start) * 1000.0
        now_ns = time.monotonic_ns()
        # Detection is the expensive part (~6 s on RK3566). Detect once,
        # cache the line boxes, then run recognition-only on those lines at
        # 1 Hz. A periodic detection (15 s) recovers from card movement.
        use_detection = (not self._cached_line_boxes
                         or now_ns - self._last_detection_ns > 15_000_000_000
                         or self._recognition_failures >= 2)
        lines = []
        raw_text = []
        try:
            if use_detection:
                result = engine(card, use_det=True, use_cls=True)
                texts = list(result.txts or ())
                scores = list(result.scores or ())
                boxes = list(result.boxes) if result.boxes is not None else []
                if len(texts) != len(scores) or len(texts) != len(boxes):
                    return finish(reason="OCR_RESULT_MISMATCH")
                cached = []
                for text, score, box in zip(texts, scores, boxes):
                    text = self._clean(text)
                    if not text:
                        continue
                    points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
                    bx1, by1 = points[:, 0].min(), points[:, 1].min()
                    bx2, by2 = points[:, 0].max(), points[:, 1].max()
                    cached.append((float(bx1), float(by1), float(bx2), float(by2)))
                    lines.append(((by1 + by2) * 0.5, text, float(score)))
                    raw_text.append(text)
                self._cached_line_boxes = tuple(cached)
                self._last_detection_ns = now_ns
                self._recognition_failures = 0
            else:
                for bx1, by1, bx2, by2 in self._cached_line_boxes:
                    pad = 3
                    left = max(0, int(round(bx1)) - pad)
                    top = max(0, int(round(by1)) - pad)
                    right = min(card.shape[1], int(round(bx2)) + pad)
                    bottom = min(card.shape[0], int(round(by2)) + pad)
                    if right <= left or bottom <= top:
                        continue
                    result = engine(card[top:bottom, left:right], use_det=False, use_cls=False)
                    texts = list(result.txts or ())
                    scores = list(result.scores or ())
                    if not texts:
                        continue
                    text = self._clean(texts[0])
                    if not text:
                        continue
                    lines.append(((by1 + by2) * 0.5, text, float(scores[0])))
                    raw_text.append(text)
                if len(lines) < 2:
                    self._recognition_failures += 1
                if not lines:
                    return finish(reason="Q2_RECOGNITION_EMPTY")
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            return finish(reason=f"OCR_UNAVAILABLE_{type(exc).__name__}")

        lines.sort()
        candidates = []
        for y, text, score in lines:
            match = DISTANCE_RE.match(text)
            if match:
                value = parse_distance(match.group(0))
                if value is not None:
                    candidates.append((y, value, text, score))
            elif "现在" in text or "立即" in text:
                candidates.append((y, 0.0, text, score))
            elif re.fullmatch(r"\d{2,4}", text) and y > card.shape[0] * 0.42:
                # The compact card sometimes loses the unit glyph. A bare
                # 2-4 digit line lower in the card is metres; speed limits are
                # above this region and are explicitly excluded.
                value = float(text)
                if 1.0 <= value <= 999.0 and text not in ("60", "63"):
                    candidates.append((y, value, text, score))
        if not candidates:
            self._recognition_failures += 1
            return finish(reason="Q2_NO_DISTANCE_OR_NOW", raw_text=raw_text)

        dist_y, distance, distance_text, distance_score = min(candidates, key=lambda item: item[0])
        instruction = None
        window = max(8, round(card.shape[0] * 0.14))
        for y, text, score in lines:
            if y <= dist_y + max(2, round(card.shape[0] * 0.01)) or y > dist_y + window:
                continue
            if text in ("随后",) or "km/h" in text.lower() or "kmh" in text.lower():
                continue
            if len(re.findall(r"[\u4e00-\u9fff]", text)) < 2:
                continue
            if instruction is None or score > instruction[2]:
                instruction = (y, text, score)
        if instruction is None:
            self._recognition_failures += 1
            return finish(reason="Q2_NO_INSTRUCTION", raw_text=raw_text)

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
            raw_text=raw_text,
            reason="Q2_CARD_OCR_RECOGNITION_CACHE",
        )
'''
s=s[:a]+new
# initialize cache before update method: insert after super class init in __init__
marker='''        super().__init__(ttl_ms=min(int(ttl_ms), 10000), ocr=ocr)
        self.ttl_ms = int(ttl_ms)'''
repl='''        super().__init__(ttl_ms=min(int(ttl_ms), 10000), ocr=ocr)
        self.ttl_ms = int(ttl_ms)
        self._cached_line_boxes = ()
        self._last_detection_ns = 0
        self._recognition_failures = 0'''
if marker not in s: raise SystemExit('init marker')
s=s.replace(marker,repl)
p.write_text(s,encoding='utf-8')
print('q2 cache patch done')
