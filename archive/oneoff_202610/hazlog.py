from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\ai_bridge.py')
s=p.read_text(encoding='utf-8')
old='''                        "navigation_state": (self.latest_dynamic_status.get("navigation") or {}).get("state"),
                        "navigation_ocr_age_ms": (self.latest_dynamic_status.get("navigation") or {}).get("ocr_age_ms"),'''
new='''                        "navigation_state": (self.latest_dynamic_status.get("navigation") or {}).get("state"),
                        "navigation_ocr_age_ms": (self.latest_dynamic_status.get("navigation") or {}).get("ocr_age_ms"),
                        "hazards": self.latest_dynamic_status.get("hazards"),
                        "hazard_details": self.latest_dynamic_status.get("hazard_details"),'''
if old not in s: raise SystemExit('timing marker')
s=s.replace(old,new)
p.write_text(s,encoding='utf-8')
print('hazard log added')
