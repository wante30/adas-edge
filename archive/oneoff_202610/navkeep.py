from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\ai_bridge.py')
s=p.read_text(encoding='utf-8')
old='''            previous = self.navigation.current(self.latest_source_timestamp_ns)
            if (observation.valid and previous.valid'''
new='''            previous = self.navigation.current(self.latest_source_timestamp_ns)
            if not observation.valid and previous.valid:
                # A single failed OCR recognition must not erase a still-fresh
                # validated card. Keep it until its own TTL expires; the next
                # focused recognition or periodic detection can refresh it.
                self.navigation_timings.append(float(observation.inference_ms))
                continue
            if (observation.valid and previous.valid'''
if old not in s: raise SystemExit('poll marker')
s=s.replace(old,new)
p.write_text(s,encoding='utf-8')
print('keep valid patched')
