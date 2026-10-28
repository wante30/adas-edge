from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\ai_bridge.py')
s=p.read_text(encoding='utf-8')
s=s.replace('self.navigation = Q2NavigationReader(ttl_ms=20000)','self.navigation = Q2NavigationReader(ttl_ms=8000)')
s=s.replace('args=(self.nav_request, self.nav_result, 20000),','args=(self.nav_request, self.nav_result, 8000),')
s=s.replace('navigation_ttl_ms=20000,','navigation_ttl_ms=8000,')
old='''            if observation.valid and self.latest_source_timestamp_ns:
                observation = replace(observation, timestamp_ns=self.latest_source_timestamp_ns)
            self.navigation._last = observation
            self.navigation_timings.append(float(observation.inference_ms))'''
new='''            previous = self.navigation.current(self.latest_source_timestamp_ns)
            if (observation.valid and previous.valid
                    and previous.distance_m is not None and observation.distance_m is not None):
                if (abs(float(observation.distance_m) - float(previous.distance_m)) > 500.0
                        and observation.ocr_timestamp_ns - previous.ocr_timestamp_ns < 3_000_000_000):
                    self.navigation_jump_rejections += 1
                    continue
            if observation.valid and self.latest_source_timestamp_ns:
                observation = replace(observation, timestamp_ns=self.latest_source_timestamp_ns)
            self.navigation._last = observation
            self.navigation_timings.append(float(observation.inference_ms))
            self.navigation_preprocess_timings.append(float(observation.preprocess_ms))
            self.navigation_engine_timings.append(float(observation.engine_ms))'''
if old not in s: raise SystemExit('poll block not found')
s=s.replace(old,new)
s=s.replace('''        self.navigation_timings = []
        self.latest_source_timestamp_ns = 0''','''        self.navigation_timings = []
        self.navigation_preprocess_timings = []
        self.navigation_engine_timings = []
        self.navigation_jump_rejections = 0
        self.latest_source_timestamp_ns = 0''')
old3='''                        "result_publish_timestamp_ns": int(publish_ns),
                        "boxes": int(struct.unpack_from("<I", packet, 20)[0]),
                    }, separators=(",", ":")) + "\\n")'''
new3='''                        "result_publish_timestamp_ns": int(publish_ns),
                        "boxes": int(struct.unpack_from("<I", packet, 20)[0]),
                        "traffic": self.latest_dynamic_status.get("traffic"),
                        "traffic_reason": self.latest_dynamic_status.get("traffic_reason"),
                        "specialist_age_ms": self.latest_dynamic_status.get("specialist_age_ms"),
                        "navigation_state": (self.latest_dynamic_status.get("navigation") or {}).get("state"),
                        "navigation_ocr_age_ms": (self.latest_dynamic_status.get("navigation") or {}).get("ocr_age_ms"),
                    }, separators=(",", ":")) + "\\n")'''
if old3 not in s: raise SystemExit('timing block not found')
s=s.replace(old3,new3)
p.write_text(s,encoding='utf-8')
print('patched')
