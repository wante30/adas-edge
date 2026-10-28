from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tools\tesla_scene_renderer.py')
code=r'''#!/usr/bin/env python3
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
OUT_A = Path("/dev/shm/tesla_scene_a.bmp")
OUT_B = Path("/dev/shm/tesla_scene_b.bmp")
W, H = 960, 540


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
    up.FOCAL_BEV = H / (2.0 * math.tan(up.FOV_BEV / 2.0))
    up._rebuild_bev_basis(28.0)
    up.update_bev_camera(0.0)


def _proj(up, x, y, z):
    return up.project_bev(np.array([float(x), float(y), float(z)], dtype=float))


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
    b = [(_proj(up, x0, y0, z0), _proj(up, x1, y0, z0), _proj(up, x1, y1, z0), _proj(up, x0, y1, z0))]
    t = [(_proj(up, x0, y0, z1), _proj(up, x1, y0, z1), _proj(up, x1, y1, z1), _proj(up, x0, y1, z1))]
    # side/back/front faces; top last
    _quad(img, [b[0], b[1], t[1], t[0]], tuple(int(v * .82) for v in color), outline)
    _quad(img, [b[1], b[2], t[2], t[1]], tuple(int(v * .92) for v in color), outline)
    _quad(img, [b[2], b[3], t[3], t[2]], tuple(int(v * .70) for v in color), outline)
    _quad(img, [b[3], b[0], t[0], t[3]], tuple(int(v * .88) for v in color), outline)
    _quad(img, t, tuple(min(255, int(v * 1.18)) for v in color), outline)
    return b, t


def draw_vehicle(up, img, center, dims, is_ego=False, kind="car"):
    w, h, d = dims
    x, y, z = center
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
        body = (52, 68, 82); roof = (66, 84, 100); accent = (150, 205, 220)
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
        # Clear ego outline; the car is deliberately drawn at the bottom centre.
        frame = _proj(up, 0, 0, .01)
        if frame:
            cv2.line(img, (frame[0] - 34, frame[1] + 2), (frame[0] + 34, frame[1] + 2),
                     (90, 180, 200), 1, cv2.LINE_AA)


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


def render(up, scene):
    state = {"speed_mps": 0.0, "accel_x": 0.0, "heading": 0.0, "lat": 0.0, "lon": 0.0,
             "lane_source": "SCHEMATIC", "motion_mode": "NO_POSE"}
    tracked = sorted(scene, key=lambda item: item["ego_y"], reverse=True)
    image = up.render_tesla_ui(tracked, state, 0.0, None)
    # Remove upstream's numerical/telemetry chrome; it is not calibrated here.
    image[:70, :] = up.BEV_BG
    image[-38:, :300] = up.BEV_BG
    image[-38:, -300:] = up.BEV_BG
    cv2.putText(image, "SPATIAL SCHEMATIC  ·  LIVE TRACKS", (22, 32),
                cv2.FONT_HERSHEY_SIMPLEX, .52, (165, 190, 200), 1, cv2.LINE_AA)
    cv2.putText(image, f"EGO + {len(tracked)} TRACKS  ·  UNCALIBRATED", (22, 55),
                cv2.FONT_HERSHEY_SIMPLEX, .42, (110, 135, 145), 1, cv2.LINE_AA)
    cv2.putText(image, "EGO", (24, H - 14), cv2.FONT_HERSHEY_SIMPLEX, .40,
                (130, 185, 200), 1, cv2.LINE_AA)
    cv2.putText(image, "NOT A METRIC WORLD MODEL", (W - 250, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, .36, (110, 125, 135), 1, cv2.LINE_AA)
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
        else:
            tracked = scene.update(payload)
            img = render(up, tracked)
            target = OUT_A if tick % 2 == 0 else OUT_B
            tmp = target.with_suffix(".tmp")
            cv2.imwrite(str(tmp), img)
            os.replace(tmp, target)
            tick += 1


if __name__ == "__main__":
    main()
'''
p.write_text(code,encoding='utf-8')
print('written',len(code))
