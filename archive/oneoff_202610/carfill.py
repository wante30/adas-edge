from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tools\tesla_scene_renderer.py')
s=p.read_text(encoding='utf-8')
s=s.replace('''    if is_ego:
        body = (46, 50, 56); roof = (62, 68, 76); accent = (145, 190, 205)''','''    if is_ego:
        body = (92, 104, 116); roof = (126, 140, 152); accent = (150, 200, 214)''')
s=s.replace('''        left = [_proj(up, x - w * .5, y + fy * d, h * fz) for fy, fz in profile]''','''        # Solid lower body first; the silhouette extrusion then adds the
        # sloped glass/roof. This also guarantees a readable ego at the chase
        # camera distance even when the side polygon is viewed edge-on.
        _box(up, img, x, y, h * .12, h * .52, w * .98, d * .98, body, accent)
        left = [_proj(up, x - w * .5, y + fy * d, h * fz) for fy, fz in profile]''')
p.write_text(s,encoding='utf-8')
print('bright body patched')
