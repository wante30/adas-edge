from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\ai_bridge.py')
s=p.read_text(encoding='utf-8')
old='''            self.navigation._last = observation
            self.navigation_timings.append(float(observation.inference_ms))'''
new='''            self.navigation._last = observation
            print(
                f"NAV_OCR_RESULT valid={int(observation.valid)} reason={observation.reason} "
                f"distance={observation.distance_m} road={observation.road_name!r} "
                f"infer_ms={observation.inference_ms:.1f} pre_ms={observation.preprocess_ms:.2f} "
                f"engine_ms={observation.engine_ms:.1f}",
                flush=True,
            )
            self.navigation_timings.append(float(observation.inference_ms))'''
if old not in s: raise SystemExit('ocr result marker')
s=s.replace(old,new)
p.write_text(s,encoding='utf-8')
print('patched')
