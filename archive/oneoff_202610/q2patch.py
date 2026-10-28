from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\tools\competition_console\navigation_reader.py')
s=p.read_text(encoding='utf-8')
old='''    raw_text: list[str] = field(default_factory=list)
    inference_ms: float = 0.'''
new='''    raw_text: list[str] = field(default_factory=list)
    inference_ms: float = 0.
    preprocess_ms: float = 0.
    engine_ms: float = 0.
    ocr_timestamp_ns: int = 0'''
if old not in s: raise SystemExit('nav fields not found')
p.write_text(s.replace(old,new),encoding='utf-8')

p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\q2_navigation_reader.py')
s=p.read_text(encoding='utf-8')
old='''        start = time.perf_counter()
        base = NavigationObservation(source_frame_id=frame_id, timestamp_ns=timestamp_ns)

        def finish(**changes):
            values = base.to_dict()
            values.update(changes)
            values["inference_ms"] = (time.perf_counter() - start) * 1000.0
            self._last = NavigationObservation(**values)
            return self._last'''
new='''        start = time.perf_counter()
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
            return self._last'''
if old not in s: raise SystemExit('q2 finish not found')
s=s.replace(old,new)
old='''        card = nav_bgr[y0:y1, x0:x1]
        if card.size == 0:
            return finish(reason="Q2_CARD_EMPTY")
        card = cv2.resize(card, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        try:
            result = engine(card, use_det=True, use_cls=True)'''
new='''        card = nav_bgr[y0:y1, x0:x1]
        if card.size == 0:
            return finish(reason="Q2_CARD_EMPTY")
        # RapidOCR already limits the detector long side to 320; the old 2x
        # upscale only duplicated resize work and memory traffic.
        preprocess_ms = (time.perf_counter() - start) * 1000.0
        try:
            result = engine(card, use_det=True, use_cls=True)'''
if old not in s: raise SystemExit('q2 card block not found')
s=s.replace(old,new)
s=s.replace('elif re.fullmatch(r"\\d{2,4}", text) and y > 120:', 'elif re.fullmatch(r"\\d{2,4}", text) and y > card.shape[0] * 0.42:')
s=s.replace('if y <= dist_y + 4 or y > dist_y + 55:', 'if y <= dist_y + max(2, round(card.shape[0] * 0.01)) or y > dist_y + max(8, round(card.shape[0] * 0.14)):')
p.write_text(s,encoding='utf-8')
print('patched')
