from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tools\tesla_scene_renderer.py')
s=p.read_text(encoding='utf-8')
old='''        _box(up, img, x, y - d * .04, h * .78, h * .94, w * .66, d * .44, roof, accent)
'''
new='''        _box(up, img, x, y - d * .04, h * .78, h * .94, w * .66, d * .44, roof, accent)
        # Rear glass (the chase camera sees the back of each car).
        rear_glass = [(_proj(up, x - w * .28, y - d * .32, h * .55),
                       _proj(up, x + w * .28, y - d * .32, h * .55),
                       _proj(up, x + w * .22, y - d * .33, h * .78),
                       _proj(up, x - w * .22, y - d * .33, h * .78))]
        _quad(img, rear_glass, (26, 34, 41))
        # Rear light bar, amber/white rather than hazard red.
        rear_lamp = _proj(up, x, y - d * .5 - .01, h * .46)
        if rear_lamp:
            cv2.line(img, (rear_lamp[0] - 10, rear_lamp[1]), (rear_lamp[0] + 10, rear_lamp[1]),
                     (145, 195, 210), 2, cv2.LINE_AA)
'''
if old not in s: raise SystemExit('car marker')
s=s.replace(old,new)
s=s.replace('image[:70, :] = up.BEV_BG','image[:96, :] = up.BEV_BG')
# ego silhouette
old2='''    if is_ego:
        # Clear ego outline; the car is deliberately drawn at the bottom centre.
        frame = _proj(up, 0, 0, .01)
        if frame:
            cv2.line(img, (frame[0] - 34, frame[1] + 2), (frame[0] + 34, frame[1] + 2),
                     (90, 180, 200), 1, cv2.LINE_AA)'''
new2='''    if is_ego:
        # Clear ego silhouette; the car is deliberately drawn at the bottom centre.
        outline = [(_proj(up, -w / 2, -d / 2, .15), _proj(up, w / 2, -d / 2, .15),
                    _proj(up, w * .44, d / 2, .15), _proj(up, -w * .44, d / 2, .15))]
        _quad(img, outline, (0, 0, 0), (95, 175, 195))
        frame = _proj(up, 0, 0, .01)
        if frame:
            cv2.line(img, (frame[0] - 34, frame[1] + 2), (frame[0] + 34, frame[1] + 2),
                     (95, 180, 200), 2, cv2.LINE_AA)'''
if old2 not in s: raise SystemExit('ego marker')
s=s.replace(old2,new2)
p.write_text(s,encoding='utf-8')
print('car detail patched')
