#!/usr/bin/env python3
"""Live adapter around the checked-in Tesla-BevCam-Clone renderer.

The upstream renderer is imported from .deps, with its detector-only imports
stubbed because the RK3566 sidecar already owns YOLO. The upstream projection,
road and camera math stay authoritative. Only the vehicle primitives, palette,
camera framing and status chrome are upgraded here, and the result is emitted
at a low fixed rate into /dev/shm for the Qt spatial view.
"""
from __future__ import annotations
import argparse, importlib.util, json, math, os, sys, time
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(os.environ.get("ADAS_EDGE_ROOT", "/home/cat/competition_cockpit_port_20261018"))
UPSTREAM = ROOT / ".deps" / "Tesla-BevCam-Clone" / "main.py"
TRACKS = Path("/dev/shm/tesla_scene_tracks.json")
ACTIVE = Path("/dev/shm/tesla_scene.active")
DEBUG = Path("/dev/shm/tesla_scene_debug.json")
OUT_A = Path("/dev/shm/tesla_scene_a.png")
OUT_B = Path("/dev/shm/tesla_scene_b.png")
W, H = 720, 405


def load_upstream():
    import types
    ute = types.ModuleType("ultralytics"); ute.YOLO = object
    sys.modules["ultralytics"] = ute
    scipy = types.ModuleType("scipy")
    optimize = types.ModuleType("scipy.optimize")
    optimize.linear_sum_assignment = lambda c: (np.arange(min(c.shape)), np.arange(min(c.shape)))
    scipy.optimize = optimize
    sys.modules["scipy"] = scipy; sys.modules["scipy.optimize"] = optimize
    spec = importlib.util.spec_from_file_location("tesla_upstream", str(UPSTREAM))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {UPSTREAM}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure(up):
    up.BEV_W, up.BEV_H = W, H
    up.BEV_BG = (14, 18, 23)           # BGR: near-black blue-grey ground
    up.LANE_LINE = (118, 132, 140)
    up.FSD_BLUE_CORE = (190, 105, 35)  # muted Tesla blue
    up.FSD_FLOW_COLOR = (225, 170, 105)
    up.EGO_COLOR = (48, 64, 78)
    up.OBJ_COLOR = (118, 126, 132)
    up.SHADOW_COLOR = (34, 38, 44)
    up.TEXT_COLOR = (205, 218, 224)
    up.UI_ACCENT = (145, 170, 180)
    up.MAX_DIST = 65.0
    up.BEV_CAM_Y = -13.0
    up.FOV_BEV = math.radians(75.0)
    up.FOCAL_BEV = H / (2.0 * math.tan(up.FOV_BEV / 2.0))
    up._rebuild_bev_basis(28.0)
    up.update_bev_camera(0.0)
    _cache_camera(up)


_CAM = None


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


def _quad(img, pts, color, outline=None, width=1):
    pts = [p for p in pts if p]
    if len(pts) < 3:
        return
    arr = np.asarray(pts, np.int32)
    cv2.fillPoly(img, [arr], color, cv2.LINE_AA)
    if outline is not None:
        cv2.polylines(img, [arr], True, outline, width, cv2.LINE_AA)


def _box(up, img, cx, cy, z0, z1, width, depth, color, outline=None):
    x0, x1 = cx - width / 2, cx + width / 2
    y0, y1 = cy - depth / 2, cy + depth / 2
    b = [_proj(up, x0, y0, z0), _proj(up, x1, y0, z0), _proj(up, x1, y1, z0), _proj(up, x0, y1, z0)]
    t = [_proj(up, x0, y0, z1), _proj(up, x1, y0, z1), _proj(up, x1, y1, z1), _proj(up, x0, y1, z1)]
    # side/back/front faces; top last
    _quad(img, [b[0], b[1], t[1], t[0]], tuple(int(v * .82) for v in color), outline)
    _quad(img, [b[1], b[2], t[2], t[1]], tuple(int(v * .92) for v in color), outline)
    _quad(img, [b[2], b[3], t[3], t[2]], tuple(int(v * .70) for v in color), outline)
    _quad(img, [b[3], b[0], t[0], t[3]], tuple(int(v * .88) for v in color), outline)
    _quad(img, t, tuple(min(255, int(v * 1.18)) for v in color), outline)
    return b, t


def draw_vehicle(up, img, center, dims, is_ego=False, kind="car"):
    w, h, d = dims
    if is_ego:
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
    if not is_ego and y > 22.0:
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
        return
    if kind == "person":
        # Head, torso and legs are still low-poly, but not a rectangular block.
        body = (62, 78, 86) if not is_ego else (65, 90, 105)
        _box(up, img, x, y, 0.02, 0.62, 0.48, 0.35, body, (120, 140, 148))
        head = _proj(up, x, y, 0.95)
        if head:
            cv2.circle(img, head, max(3, int(5 * max(0.45, 6.0 / max(6.0, y)))), (108, 120, 126), -1, cv2.LINE_AA)
        return
    shadow = [(_proj(up, x - w * .55, y - d * .55, .015), _proj(up, x + w * .55, y - d * .55, .015),
               _proj(up, x + w * .55, y + d * .55, .015), _proj(up, x - w * .55, y + d * .55, .015))]
    _quad(img, shadow, (30, 34, 39))
    if is_ego:
        body = (96, 88, 78); roof = (122, 114, 104); accent = (150, 200, 214)
    elif kind == "truck":
        body = (118, 126, 132); roof = (142, 150, 154); accent = (160, 172, 178)
    elif kind == "bus":
        body = (88, 108, 124); roof = (112, 132, 146); accent = (165, 182, 192)
    else:
        body = (145, 153, 160); roof = (173, 179, 184); accent = (190, 198, 202)

    if kind == "truck":
        # Cab plus cargo box, with a visible gap.
        _box(up, img, x, y - d * .28, .18, h * .78, w * .94, d * .26, body, accent)
        _box(up, img, x, y + d * .16, .22, h * .98, w, d * .62, tuple(int(v * .88) for v in body), accent)
        cab_top = _proj(up, x, y - d * .28, h * .78)
        if cab_top:
            cv2.rectangle(img, (cab_top[0] - 7, cab_top[1] - 4), (cab_top[0] + 7, cab_top[1] + 2), (28, 36, 42), -1)
    elif kind == "bus":
        _box(up, img, x, y, .18, h * .93, w, d, body, accent)
        # Continuous side window band.
        for yy in (y - d * .32, y, y + d * .32):
            q = [(_proj(up, x - w / 2 - .01, yy - d * .12, h * .55),
                  _proj(up, x - w / 2 - .01, yy + d * .12, h * .55),
                  _proj(up, x - w / 2 - .01, yy + d * .12, h * .78),
                  _proj(up, x - w / 2 - .01, yy - d * .12, h * .78))]
            _quad(img, q, (26, 35, 41))
    if kind == "car":
        # Side-profile extrusion gives a car silhouette (sloped rear glass,
        # trunk, roof, bonnet) instead of stacked cuboids.
        roof_color = tuple(min(255, int(v * 1.16)) for v in body)
        profile = [(-.50, .12), (-.50, .38), (-.38, .50), (-.27, .56),
                   (-.16, .84), (-.02, .94), (.14, .94), (.28, .62),
                   (.44, .50), (.50, .38), (.50, .12)]
        # Solid lower body first; the silhouette extrusion then adds the
        # sloped glass/roof. This also guarantees a readable ego at the chase
        # camera distance even when the side polygon is viewed edge-on.
        _box(up, img, x, y, h * .12, h * .52, w * .98, d * .98, body, accent)
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
        _box(up, img, x, y - d * .04, h * .44, h * .82, w * .86, d * .62, tuple(int(v * .96) for v in body), accent)
        # Windshield and side glass.
        windshield = [(_proj(up, x - w * .36, y + d * .28, h * .50),
                       _proj(up, x + w * .36, y + d * .28, h * .50),
                       _proj(up, x + w * .30, y + d * .29, h * .78),
                       _proj(up, x - w * .30, y + d * .29, h * .78))]
        _quad(img, windshield, (30, 40, 47))
        _box(up, img, x, y - d * .04, h * .78, h * .94, w * .66, d * .44, roof, accent)
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
    # Wheels: four dark low boxes; projection naturally shrinks with distance.
    wheel_z = .30
    for sx in (-1, 1):
        for sy in (-1, 1):
            wx = x + sx * (w / 2 - .06)
            wy = y + sy * (d * .31)
            _box(up, img, wx, wy, .02, wheel_z, .20, .38, (24, 27, 31))
    # Lamps: neutral white/amber only; never use hazard red here.
    for sx in (-1, 1):
        lamp = _proj(up, x + sx * w * .32, y + d * .5 + .01, h * .46)
        if lamp:
            cv2.circle(img, lamp, 2, (210, 225, 235) if not is_ego else (180, 230, 240), -1, cv2.LINE_AA)
    if is_ego:
        # Clear ego silhouette; the car is deliberately drawn at the bottom centre.
        outline = [(_proj(up, -w / 2, -d / 2, .15), _proj(up, w / 2, -d / 2, .15),
                    _proj(up, w * .44, d / 2, .15), _proj(up, -w * .44, d / 2, .15))]
        _quad(img, outline, (0, 0, 0), (95, 175, 195))
        frame = _proj(up, 0, 0, .01)
        if frame:
            cv2.line(img, (frame[0] - 34, frame[1] + 2), (frame[0] + 34, frame[1] + 2),
                     (95, 180, 200), 2, cv2.LINE_AA)


def install(up):
    up.draw_solid_3d_vehicle = lambda img, center, dims, is_ego=False, kind="car": draw_vehicle(
        up, img, center, dims, is_ego, kind)


class SceneState:
    def __init__(self):
        self.source = None
        self.targets = {}
        self.last = 0.0

    def update(self, payload):
        now = time.monotonic()
        source = str(payload.get("source") or "")
        if source != self.source:
            self.targets.clear(); self.source = source
        seen = set()
        for item in payload.get("tracks") or ():
            try:
                tid = int(item.get("id", -1)); cls = str(item.get("class", ""))
                x = float(item["x"]); y = float(item["y"])
                w = float(item["width"]); h = float(item["height"])
            except (KeyError, TypeError, ValueError):
                continue
            if tid < 0 or cls not in {"car", "truck", "bus", "person", "motorcycle"} or w < 3 or h < 3:
                continue
            cx = (x + w * .5) / 1280.0
            bottom = (y + h) / 720.0
            tx = max(-5.5, min(5.5, (cx - .5) * 7.0))
            ty = max(3.5, min(58.0, 3.5 + (1.0 - bottom) * 52.0))
            seen.add(tid)
            target = self.targets.get(tid)
            if target is None:
                target = {"x": tx, "y": ty, "cls": cls, "seen": now}
                self.targets[tid] = target
            else:
                target["cls"] = cls
                dt = max(0.0, now - target["seen"])
                alpha = 1.0 - math.exp(-dt / 0.28)
                dx = max(-5.0 * dt, min(5.0 * dt, tx - target["x"]))
                dy = max(-14.0 * dt, min(14.0 * dt, ty - target["y"]))
                target["x"] += dx * alpha
                target["y"] += dy * alpha
                target["seen"] = now
        for tid in list(self.targets):
            if tid not in seen and now - self.targets[tid]["seen"] > 1.6:
                del self.targets[tid]
        self.last = now
        return [{"id": tid, "ego_x": t["x"], "ego_y": t["y"], "cls": t["cls"]}
                for tid, t in self.targets.items()]

    def clear(self):
        self.targets.clear()


_BASE = None


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=5.0)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    up = load_upstream(); configure(up); install(up)
    if args.preview:
        scene = [{"id": 1, "ego_x": 0, "ego_y": 13, "cls": "car"},
                 {"id": 2, "ego_x": 3.2, "ego_y": 27, "cls": "truck"}]
        img = render(up, scene)
        cv2.imwrite(args.out or "/tmp/tesla_scene_preview.bmp", img)
        print("preview", args.out or "/tmp/tesla_scene_preview.bmp", img.shape)
        return
    scene = SceneState()
    period = 1.0 / max(1.0, args.rate)
    tick = 0
    next_tick = time.monotonic()
    last_payload = 0.0
    while True:
        now = time.monotonic()
        if now < next_tick:
            time.sleep(min(.05, next_tick - now)); continue
        next_tick += period
        if next_tick < now:
            next_tick = now + period
        if not ACTIVE.exists() or ACTIVE.read_text(errors="ignore").strip() not in ("1", "true"):
            time.sleep(.1); continue
        payload = None
        try:
            if TRACKS.exists():
                payload = json.loads(TRACKS.read_text(encoding="utf-8"))
                last_payload = TRACKS.stat().st_mtime
        except (OSError, ValueError):
            payload = None
        if payload is None or time.time() - last_payload > 1.5:
            scene.clear()
            tracked = []
        else:
            tracked = scene.update(payload)
            try:
                DEBUG.write_text(json.dumps({
                    "source": scene.source, "tracks": tracked,
                    "source_frame_id": payload.get("source_frame_id"),
                    "timestamp_ns": payload.get("timestamp_ns"),
                }, ensure_ascii=False), encoding="utf-8")
            except OSError:
                pass
        # Always emit, including the ego-only cleanup frame after AI stops.
        img = render(up, tracked)
        ok, encoded = cv2.imencode(".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
        if ok:
            for target in (OUT_A, OUT_B):
                tmp = target.with_name(target.name + ".tmp")
                tmp.write_bytes(encoded.tobytes())
                os.replace(tmp, target)
            tick += 1


if __name__ == "__main__":
    main()
