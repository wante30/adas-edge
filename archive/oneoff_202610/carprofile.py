from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tools\tesla_scene_renderer.py')
s=p.read_text(encoding='utf-8')
marker='''    else:
        # Lower body, tapered hood and a narrower cabin.
        _box(up, img, x, y, .15, h * .48, w, d, body, accent)
'''
new='''    if kind == "car":
        # Side-profile extrusion gives a car silhouette (sloped rear glass,
        # trunk, roof, bonnet) instead of stacked cuboids.
        profile = [(-.50, .12), (-.50, .38), (-.38, .50), (-.27, .56),
                   (-.16, .84), (-.02, .94), (.14, .94), (.28, .62),
                   (.44, .50), (.50, .38), (.50, .12)]
        left = [_proj(up, x - w * .5, y + fy * d, h * fz) for fy, fz in profile]
        right = [_proj(up, x + w * .5, y + fy * d, h * fz) for fy, fz in profile]
        if all(left) and all(right):
            _quad(img, left, tuple(int(v * .78) for v in body), accent)
            _quad(img, right, body, accent)
            cv2.polylines(img, [np.asarray(left, np.int32)], True, accent, 1, cv2.LINE_AA)
            cv2.polylines(img, [np.asarray(right, np.int32)], True, accent, 1, cv2.LINE_AA)
            roof = [_proj(up, x - w * .34, y + .14 * d, h * .94),
                    _proj(up, x + w * .34, y + .14 * d, h * .94),
                    _proj(up, x + w * .34, y - .02 * d, h * .94),
                    _proj(up, x - w * .34, y - .02 * d, h * .94)]
            _quad(img, roof, roof_color, accent)
            # Side glass on the visible right/left flank.
            for side, base in ((-1, left), (1, right)):
                glass = [base[4], base[5], base[6], base[7]]
                _quad(img, glass, (29, 38, 45), (88, 104, 112))
            rear_face = [_proj(up, x - w * .44, y - .5 * d, h * .12),
                         _proj(up, x + w * .44, y - .5 * d, h * .12),
                         _proj(up, x + w * .44, y - .5 * d, h * .44),
                         _proj(up, x - w * .44, y - .5 * d, h * .44)]
            _quad(img, rear_face, tuple(int(v * .62) for v in body), accent)
            rear_glass = [_proj(up, x - w * .34, y - .178 * d, h * .60),
                          _proj(up, x + w * .34, y - .178 * d, h * .60),
                          _proj(up, x + w * .26, y - .04 * d, h * .91),
                          _proj(up, x - w * .26, y - .04 * d, h * .91)]
            _quad(img, rear_glass, (26, 35, 42), (80, 96, 105))
            for side in (-1, 1):
                for sy in (-.30, .30):
                    _box(up, img, x + side * (w * .54), y + sy * d, .02, h * .28,
                         .16, .34, (23, 26, 30))
            for sx in (-1, 1):
                lamp = _proj(up, x + sx * w * .34, y - .5 * d - .01, h * .46)
                if lamp:
                    cv2.line(img, (lamp[0] - 7, lamp[1]), (lamp[0] + 7, lamp[1]),
                             (150, 200, 215), 2, cv2.LINE_AA)
        if is_ego:
            frame = _proj(up, 0, 0, .01)
            if frame:
                cv2.line(img, (frame[0] - 36, frame[1] + 2), (frame[0] + 36, frame[1] + 2),
                         (95, 180, 200), 2, cv2.LINE_AA)
        return
    else:
        # Lower body, tapered hood and a narrower cabin.
        _box(up, img, x, y, .15, h * .48, w, d, body, accent)
'''
if marker not in s: raise SystemExit('car branch marker')
# define roof_color in car branch
new=new.replace('''        profile = [''','''        roof_color = tuple(min(255, int(v * 1.16)) for v in body)
        profile = [''')
s=s.replace(marker,new,1)
p.write_text(s,encoding='utf-8')
print('car profile patch')
