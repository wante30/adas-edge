from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tools\tesla_scene_renderer.py')
s=p.read_text(encoding='utf-8')
s=s.replace('up.FOCAL_BEV = H / (2.0 * math.tan(up.FOV_BEV / 2.0))','up.FOV_BEV = math.radians(65.0)\n    up.FOCAL_BEV = H / (2.0 * math.tan(up.FOV_BEV / 2.0))')
s=s.replace('''    w, h, d = dims
    x, y, z = center''','''    w, h, d = dims
    if is_ego:
        # The chase camera should make the ego unmistakable without pretending
        # the image is a metric reconstruction.
        w *= 1.35; h *= 1.22; d *= 1.18
    x, y, z = center''')
p.write_text(s,encoding='utf-8')
print('ego/fov patched')
