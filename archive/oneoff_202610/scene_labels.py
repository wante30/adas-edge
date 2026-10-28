from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tools\tesla_scene_renderer.py')
s=p.read_text(encoding='utf-8')
old='''    cv2.putText(image, "SPATIAL SCHEMATIC - LIVE TRACKS", (22, 31),
                cv2.FONT_HERSHEY_SIMPLEX, .52, (165, 190, 200), 1, cv2.LINE_AA)
    cv2.putText(image, f"EGO + {len(tracked)} TRACKS - UNCALIBRATED", (22, 54),
                cv2.FONT_HERSHEY_SIMPLEX, .42, (110, 135, 145), 1, cv2.LINE_AA)
    cv2.putText(image, "EGO", (24, H - 14), cv2.FONT_HERSHEY_SIMPLEX, .40,
                (130, 185, 200), 1, cv2.LINE_AA)
    cv2.putText(image, "NOT A METRIC WORLD MODEL", (W - 250, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, .36, (110, 125, 135), 1, cv2.LINE_AA)
'''
new='''    # The Qt overlay owns titles and the "not calibrated" label; keep this
    # image free of duplicate chrome so the 3D scene can fill the viewport.
'''
if old not in s: raise SystemExit('labels')
s=s.replace(old,new)
p.write_text(s,encoding='utf-8')
print('labels removed')
