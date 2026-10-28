from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\q2_navigation_reader.py')
s=p.read_text(encoding='utf-8')
s=s.replace('''        self._cached_line_boxes = ()
        self._last_detection_ns = 0
        self._recognition_failures = 0''','''        self._cached_line_boxes = ()
        self._cached_text_boxes = ()
        self._focus_boxes = ()
        self._last_detection_ns = 0
        self._recognition_failures = 0''')
s=s.replace('''        # Detection is ~3.5 s when the board is not saturated. Recognition-only
        # from stale line boxes is faster but loses the road-name line when the
        # card shifts; keep detection authoritative for every OCR result.
        use_detection = True''','''        use_detection = (not self._focus_boxes
                         or now_ns - self._last_detection_ns > 15_000_000_000
                         or self._recognition_failures >= 2)''')
s=s.replace('''                self._cached_line_boxes = tuple(cached)
                self._last_detection_ns = now_ns
                self._recognition_failures = 0''','''                self._cached_line_boxes = tuple(cached)
                self._cached_text_boxes = tuple(zip(raw_text, cached))
                self._last_detection_ns = now_ns
                self._recognition_failures = 0''')
s=s.replace('''                for bx1, by1, bx2, by2 in self._cached_line_boxes:
                    pad = 3''','''                for bx1, by1, bx2, by2 in self._focus_boxes:
                    pad = 3''')
old='''        instruction_text = instruction[1]
        direction = "none"
        for token, value in (("掉头", "uturn"), ("左转", "left"), ("右转", "right"),
                             ("直行", "forward")):
            if token in instruction_text:
                direction = value
                break
        return finish('''
new='''        instruction_text = instruction[1]
        focus = []
        for text, box in self._cached_text_boxes:
            if text in (distance_text, instruction_text):
                focus.append(box)
        if focus:
            self._focus_boxes = tuple(focus)
        direction = "none"
        for token, value in (("掉头", "uturn"), ("左转", "left"), ("右转", "right"),
                             ("直行", "forward")):
            if token in instruction_text:
                direction = value
                break
        return finish('''
if old not in s: raise SystemExit('instruction marker')
s=s.replace(old,new)
p.write_text(s,encoding='utf-8')
print('focus patch')
