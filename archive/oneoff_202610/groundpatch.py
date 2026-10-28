from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tools\tesla_scene_renderer.py')
s=p.read_text(encoding='utf-8')
s=s.replace('w *= 1.15; h *= 1.12; d *= 1.10','w *= 1.24; h *= 1.16; d *= 1.12')
old='''    # Soft vignette keeps the eye on the ego/path without changing geometry.
'''
new='''    # Ground texture: only paint pixels that are still empty background, so
    # lane paint, vehicles and ego are never crossed by debug-looking lines.
    background = np.all(np.abs(image.astype(np.int16) - np.asarray(up.BEV_BG, np.int16)) < 7, axis=2)
    detail = np.zeros_like(image)
    for yy in np.linspace(5.0, 60.0, 12):
        a, b = _proj(up, -22.0, float(yy), 0.005), _proj(up, 22.0, float(yy), 0.005)
        if a and b:
            cv2.line(detail, a, b, (27, 33, 39), 1, cv2.LINE_AA)
    for xx in (-16.0, -11.0, -7.0, 7.0, 11.0, 16.0):
        pts = []
        for yy in np.linspace(4.0, 60.0, 16):
            p = _proj(up, float(xx), float(yy), 0.005)
            if p:
                pts.append(p)
        if len(pts) > 1:
            cv2.polylines(detail, [np.asarray(pts, np.int32)], False, (25, 31, 37), 1, cv2.LINE_AA)
    image[background] = ((image[background].astype(np.int16) * 3 +
                          detail[background].astype(np.int16)) // 4).astype(np.uint8)
    # Soft vignette keeps the eye on the ego/path without changing geometry.
'''
if old not in s: raise SystemExit('vignette marker')
s=s.replace(old,new,1)
p.write_text(s,encoding='utf-8')
print('ground/ego patched')
