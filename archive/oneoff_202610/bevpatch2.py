from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tools\tesla_scene_renderer.py')
s=p.read_text(encoding='utf-8')
s=s.replace('''    if is_ego:
        body = (52, 68, 82); roof = (66, 84, 100); accent = (150, 205, 220)''','''    if is_ego:
        body = (46, 50, 56); roof = (62, 68, 76); accent = (145, 190, 205)''')
s=s.replace('''    image = up.render_tesla_ui(tracked, state, 0.0, None)
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
    return image''','''    image = up.render_tesla_ui(tracked, state, 0.0, None)
    # Keep the upstream projection and palette, but add road-edge curb detail
    # and remove its uncalibrated numerical telemetry chrome.
    for side in (-1, 1):
        pts = []
        for yy in np.linspace(3.0, 64.0, 18):
            p = _proj(up, side * (up.LANE_HALF + 0.38), float(yy), 0.01)
            if p:
                pts.append(p)
        if len(pts) > 1:
            cv2.polylines(image, [np.asarray(pts, np.int32)], False, (42, 53, 61), 2, cv2.LINE_AA)
    image[:70, :] = up.BEV_BG
    image[-50:, :330] = up.BEV_BG
    image[-50:, 540:] = up.BEV_BG
    cv2.putText(image, "SPATIAL SCHEMATIC - LIVE TRACKS", (22, 31),
                cv2.FONT_HERSHEY_SIMPLEX, .52, (165, 190, 200), 1, cv2.LINE_AA)
    cv2.putText(image, f"EGO + {len(tracked)} TRACKS - UNCALIBRATED", (22, 54),
                cv2.FONT_HERSHEY_SIMPLEX, .42, (110, 135, 145), 1, cv2.LINE_AA)
    cv2.putText(image, "EGO", (24, H - 14), cv2.FONT_HERSHEY_SIMPLEX, .40,
                (130, 185, 200), 1, cv2.LINE_AA)
    cv2.putText(image, "NOT A METRIC WORLD MODEL", (W - 250, H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, .36, (110, 125, 135), 1, cv2.LINE_AA)
    # Soft vignette keeps the eye on the ego/path without changing geometry.
    yy, xx = np.mgrid[0:H, 0:W]
    radius = np.sqrt(((xx - W * .5) / (W * .62)) ** 2 + ((yy - H * .62) / (H * .72)) ** 2)
    shade = np.clip(1.0 - 0.18 * np.clip(radius - .72, 0.0, 1.0), .78, 1.0)
    image = np.clip(image.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)
    return image''')
p.write_text(s,encoding='utf-8')
print('patched')
