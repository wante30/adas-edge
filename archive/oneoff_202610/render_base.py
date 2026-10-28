from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tools\tesla_scene_renderer.py')
s=p.read_text(encoding='utf-8')
start=s.index('def render(up, scene):')
end=s.index('\n\ndef main():', start)
new=r'''_BASE = None


def build_base(up):
    """Render the static road/ground once; only vehicles change per frame."""
    image = np.full((H, W, 3), up.BEV_BG, np.uint8)
    up.draw_clean_lanes_animated(image, 0.0, 0.0)
    up.draw_fsd_path_gradient(image, 0.0, 0.0)
    for side in (-1, 1):
        pts = []
        for yy in np.linspace(3.0, 64.0, 18):
            p = _proj(up, side * (up.LANE_HALF + 0.38), float(yy), 0.01)
            if p:
                pts.append(p)
        if len(pts) > 1:
            cv2.polylines(image, [np.asarray(pts, np.int32)], False, (42, 53, 61), 2, cv2.LINE_AA)
    detail = image.copy()
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
    # Blend only where the road/lane did not paint, so the grid stays subtle.
    bg = np.all(np.abs(image.astype(np.int16) - np.asarray(up.BEV_BG, np.int16)) < 7, axis=2)
    image[bg] = ((image[bg].astype(np.int16) * 3 + detail[bg].astype(np.int16)) // 4).astype(np.uint8)
    yy, xx = np.mgrid[0:H, 0:W]
    radius = np.sqrt(((xx - W * .5) / (W * .62)) ** 2 + ((yy - H * .62) / (H * .72)) ** 2)
    shade = np.clip(1.0 - 0.18 * np.clip(radius - .72, 0.0, 1.0), .78, 1.0)
    return np.clip(image.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)


def render(up, scene):
    global _BASE
    if _BASE is None:
        _BASE = build_base(up)
    tracked = sorted(scene, key=lambda item: item["ego_y"], reverse=True)
    image = _BASE.copy()
    # LOD: distant targets need silhouette only; close targets get glass,
    # wheels and lamps. This is the main CPU saving for crowded scenes.
    for obj in tracked:
        dims = up.DIMS.get(obj["cls"], up.DIMS["car"])
        draw_vehicle(up, image, (obj["ego_x"], obj["ego_y"], 0), dims, False, obj["cls"])
    draw_vehicle(up, image, (0, 0, 0), up.DIMS["car"], True, "car")
    return image
'''
s=s[:start]+new+s[end:]
p.write_text(s,encoding='utf-8')
print('base cache render installed')
