"""Render BDD100K polygon labels into the official lane/drivable mask encodings.

Lane marking mask encoding (from bdd100k/label/to_mask.py):
    value = category_id + (lane_direction << 5) + (lane_style << 4)
    background = 255
    category 0..7, style 0=solid 1=dashed, direction 0=parallel 1=vertical
Drivable mask: 0=direct, 1=alternative, 2=background
"""

from __future__ import annotations

LANE_CAT = {
    "crosswalk": 0,
    "double other": 1,
    "double white": 2,
    "double yellow": 3,
    "road curb": 4,
    "single other": 5,
    "single white": 6,
    "single yellow": 7,
}
LANE_BG = 255
DRIVABLE_BG = 2

DET_CLASSES = [
    "person", "rider", "car", "truck", "bus", "train", "motor", "bike",
    "traffic light", "traffic sign",
]
DET_ALIASES = {
    "pedestrian": "person",
    "motorcycle": "motor",
    "bicycle": "bike",
}


def _vertices(poly):
    """Normalise the two BDD100K poly2d serialisations to [[x, y, type], ...]."""
    if isinstance(poly, dict):
        verts = poly.get("vertices", [])
        types = poly.get("types", "")
        return [[float(v[0]), float(v[1]), types[i] if i < len(types) else "L"]
                for i, v in enumerate(verts)]
    return [[float(p[0]), float(p[1]), (p[2] if len(p) > 2 else "L")] for p in poly]


def sample_polyline(vertices, steps: int = 12):
    """Sample cubic/quadratic bezier segments described by Scalabel C/L types."""
    pts = [(vertices[0][0], vertices[0][1])]
    cur = pts[0]
    i = 1
    n = len(vertices)
    while i < n:
        typ = vertices[i][2] if len(vertices[i]) > 2 else "L"
        if typ == "L":
            cur = (vertices[i][0], vertices[i][1])
            pts.append(cur)
            i += 1
            continue

        ctrl = []
        while i < n and (vertices[i][2] if len(vertices[i]) > 2 else "L") == "C":
            ctrl.append((vertices[i][0], vertices[i][1]))
            i += 1
        end = (vertices[i][0], vertices[i][1]) if i < n else ctrl[-1]
        if len(ctrl) >= 3:
            p0, p1, p2, p3 = cur, ctrl[-3], ctrl[-2], end
            for k in range(1, steps + 1):
                t = k / steps
                mt = 1 - t
                x = mt**3*p0[0] + 3*mt*mt*t*p1[0] + 3*mt*t*t*p2[0] + t**3*p3[0]
                y = mt**3*p0[1] + 3*mt*mt*t*p1[1] + 3*mt*t*t*p2[1] + t**3*p3[1]
                pts.append((x, y))
        elif len(ctrl) == 2:
            p0, p1, p2 = cur, ctrl[-2], end
            for k in range(1, steps + 1):
                t = k / steps
                mt = 1 - t
                pts.append((mt*mt*p0[0] + 2*mt*t*p1[0] + t*t*p2[0],
                            mt*mt*p0[1] + 2*mt*t*p1[1] + t*t*p2[1]))
        elif len(ctrl) == 1:
            pts.append(ctrl[0])
        pts.append(end)
        cur = end
        i += 1
    return pts


def _polyline_list(polys):
    """poly2d can be a flat vertex list, or a list of polylines."""
    if not polys:
        return []
    first = polys[0]
    if isinstance(first, (int, float)):
        return [polys]
    if isinstance(first, list) and first and isinstance(first[0], (int, float)):
        return [polys]
    return polys


def render_lane(draw, objects, sx: float, sy: float, width: int):
    """Draw all lane polylines with the official category/style/direction bits."""
    count = 0
    for obj in objects:
        cat_name = str(obj.get("category", ""))
        if not cat_name.startswith("lane/"):
            continue
        short = cat_name.split("/", 1)[1]
        if short not in LANE_CAT:
            continue
        attrs = obj.get("attributes") or {}
        style_raw = attrs.get("style", attrs.get("laneStyle", "solid"))
        dir_raw = attrs.get("direction", attrs.get("laneDirection", "parallel"))
        style = 1 if str(style_raw).lower() == "dashed" else 0
        direction = 1 if str(dir_raw).lower() == "vertical" else 0
        value = LANE_CAT[short] + (direction << 5) + (style << 4)
        for poly in _polyline_list(obj.get("poly2d") or []):
            pts = sample_polyline(_vertices(poly))
            if len(pts) < 2:
                continue
            xy = [(p[0] * sx, p[1] * sy) for p in pts]
            draw.line(xy, fill=value, width=width, joint="curve")
            count += 1
    return count


def render_drivable(draw, objects, sx: float, sy: float):
    count = 0
    for obj in objects:
        cat = str(obj.get("category", ""))
        if cat == "area/drivable":
            value = 0
        elif cat == "area/alternative":
            value = 1
        else:
            continue
        for poly in _polyline_list(obj.get("poly2d") or []):
            pts = sample_polyline(_vertices(poly))
            if len(pts) >= 3:
                draw.polygon([(p[0] * sx, p[1] * sy) for p in pts], fill=value)
                count += 1
    return count


def det_boxes(objects):
    """Convert BDD boxes to (cls, cx, cy, w, h) normalised to the image size."""
    out = []
    for obj in objects:
        cat = str(obj.get("category", ""))
        cat = DET_ALIASES.get(cat, cat)
        if cat not in DET_CLASSES:
            continue
        box = obj.get("box2d")
        if not box:
            continue
        x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((DET_CLASSES.index(cat), x1, y1, x2, y2))
    return out
