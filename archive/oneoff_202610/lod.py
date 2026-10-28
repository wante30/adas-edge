from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tools\tesla_scene_renderer.py')
s=p.read_text(encoding='utf-8')
old='''    if is_ego:
        # The chase camera should make the ego unmistakable without pretending
        # the image is a metric reconstruction.
        w *= 1.24; h *= 1.16; d *= 1.12
    x, y, z = center'''
new='''    if is_ego:
        # The chase camera should make the ego unmistakable without pretending
        # the image is a metric reconstruction.
        w *= 1.24; h *= 1.16; d *= 1.12
    x, y, z = center
    if not is_ego and y > 40.0:
        # Far LOD: silhouette only, no glass/wheels/lamps.
        if kind == "person":
            _box(up, img, x, y, .02, .9, .48, .35, (62, 78, 86), (110, 125, 132))
        else:
            _box(up, img, x, y, .12, h * .55, w, d, (118, 126, 132), (150, 160, 166))
            _box(up, img, x, y - d * .04, h * .55, h * .9, w * .76, d * .5, (145, 153, 160), (170, 178, 184))
        return
    if not is_ego and y > 27.0:
        # Mid LOD: body + cabin, no per-part glass/wheels.
        if kind == "person":
            _box(up, img, x, y, .02, .62, .48, .35, (62, 78, 86), (120, 140, 148))
            _box(up, img, x, y, .62, 1.05, .34, .3, (92, 104, 112), (130, 145, 152))
        elif kind == "truck":
            _box(up, img, x, y - d * .28, .18, h * .78, w * .94, d * .26, (118, 126, 132), (150, 160, 166))
            _box(up, img, x, y + d * .16, .22, h * .98, w, d * .62, (104, 112, 120), (145, 155, 162))
        elif kind == "bus":
            _box(up, img, x, y, .18, h * .93, w, d, (88, 108, 124), (120, 145, 160))
        else:
            _box(up, img, x, y, .12, h * .52, w, d, (118, 126, 132), (150, 160, 166))
            _box(up, img, x, y - d * .04, h * .52, h * .9, w * .82, d * .58, (145, 153, 160), (170, 178, 184))
        return'''
if old not in s: raise SystemExit('ego marker')
s=s.replace(old,new,1)
p.write_text(s,encoding='utf-8')
print('LOD patched')
