from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tools\tesla_scene_renderer.py')
s=p.read_text(encoding='utf-8')
old='''def _proj(up, x, y, z):
    return up.project_bev(np.array([float(x), float(y), float(z)], dtype=float))
'''
new='''_CAM = None


def _cache_camera(up):
    global _CAM
    _CAM = (up._bev_pos, up._bev_fwd, up._bev_right, up._bev_up,
            float(up.FOCAL_BEV), int(up.BEV_W), int(up.BEV_H))


def _proj(up, x, y, z):
    # Scalar projection: the upstream numpy project_bev is called thousands of
    # times per frame and dominates CPU on RK3566.
    if _CAM is None:
        _cache_camera(up)
    pos, fwd, right, upv, focal, w, h = _CAM
    dx, dy, dz = float(x) - pos[0], float(y) - pos[1], float(z) - pos[2]
    depth = dx * fwd[0] + dy * fwd[1] + dz * fwd[2]
    if depth <= 0.1:
        return None
    sx = dx * right[0] + dy * right[1] + dz * right[2]
    sy = dx * upv[0] + dy * upv[1] + dz * upv[2]
    return int(w * .5 + sx / depth * focal), int(h * .5 - sy / depth * focal)
'''
if old not in s: raise SystemExit('proj marker')
s=s.replace(old,new)
# cache after update in configure
old2='''    up._rebuild_bev_basis(28.0)
    up.update_bev_camera(0.0)'''
new2='''    up._rebuild_bev_basis(28.0)
    up.update_bev_camera(0.0)
    _cache_camera(up)'''
if old2 not in s: raise SystemExit('configure marker')
s=s.replace(old2,new2,1)
p.write_text(s,encoding='utf-8')
print('scalar projection patched')
