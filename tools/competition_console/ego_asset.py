"""Cached, static ego-vehicle illustration from the bundled Carviz OBJ.

The PNG is a presentation asset, never an estimated ego pose or perception result.
Normal UI use only decodes/resizes the PNG once. Rebuilding is an offline operation.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


ASSET_DIR = Path(__file__).resolve().parent / "assets"
ASSET_PATH = ASSET_DIR / "ego_car_rear.png"


@lru_cache(maxsize=8)
def load_ego_asset(size: tuple[int, int] = (340, 370)) -> Image.Image:
    """Return an RGBA illustration; callers should not modify the cached image."""
    with Image.open(ASSET_PATH) as image:
        return image.convert("RGBA").resize(size, Image.Resampling.LANCZOS)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _mesh(path: Path):
    vertices, normals, triangles, materials = [], [], [], []
    material = "body_color"
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(x) for x in fields[1:4]])
        elif fields[0] == "vn":
            normals.append([float(x) for x in fields[1:4]])
        elif fields[0] == "usemtl":
            material = fields[1]
        elif fields[0] == "f":
            face = [tuple(int(i) - 1 if i else -1 for i in s.split("/")) for s in fields[1:]]
            for index in range(1, len(face) - 1):
                triangles.append((face[0], face[index], face[index + 1]))
                materials.append(material)
    return np.asarray(vertices), np.asarray(normals), triangles, materials


def build_ego_asset(obj_path: Path, output: Path = ASSET_PATH,
                    size: tuple[int, int] = (680, 740)) -> Path:
    """Rasterize the existing mesh with a depth buffer and smooth lighting offline."""
    vertices, normals, triangles, materials = _mesh(obj_path)
    # Existing mesh coordinates: +X forward, +Y up; tail lamps are at -X.
    center = np.array([1.35, 0.63, 0.03])
    eye = np.array([-8.8, 9.0, 1.2])
    forward = _unit(center - eye)
    right = _unit(np.cross(forward, np.array([0., 1., 0.])))
    up = _unit(np.cross(right, forward))
    camera = np.column_stack((right, up, forward))
    view = (vertices - center) @ camera
    width, height = size
    extent = np.ptp(view[:, :2], axis=0)
    scale = min(width * .79 / extent[0], height * .87 / extent[1])
    xy = view[:, :2] * np.array([scale, -scale])
    xy += np.array([width * .5, height * .48]) - (xy.min(0) + xy.max(0)) / 2
    depth = view[:, 2]
    z_buffer = np.full((height, width), np.inf)
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    light = _unit(np.array([-3., 8., -5.]))
    fill = _unit(np.array([3., 3., 5.]))
    to_eye = _unit(eye - center)
    half = _unit(light + to_eye)
    palette = {
        "body_color": (.73, .80, .84),
        "glass3": (.065, .115, .16),
        "Black_Plastic": (.035, .048, .061),
        "tire": (.045, .052, .063),
        "tirehouse": (.045, .055, .068),
        "metal": (.40, .47, .53),
        "metal2": (.69, .76, .80),
        "Material__19": (.74, .79, .80),
        "plt1": (.76, .87, .92),
        "plt3": (.76, .055, .065),
        "plt4": (.72, .37, .07),
        "wire_115115115": (.30, .38, .45),
        "wire_086086086": (.23, .28, .33),
    }
    for face, material in zip(triangles, materials):
        ids = np.asarray([x[0] for x in face])
        p = xy[ids]
        lo = np.maximum(np.floor(p.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(p.max(0)).astype(int), [width - 1, height - 1])
        if np.any(hi < lo):
            continue
        a, b, c = p
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-8:
            continue
        yy, xx = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        w0 = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denominator
        w1 = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denominator
        w2 = 1.0 - w0 - w1
        z = w0 * depth[ids[0]] + w1 * depth[ids[1]] + w2 * depth[ids[2]]
        local_z = z_buffer[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        mask = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8) & (z < local_z)
        if not np.any(mask):
            continue
        ni = [f[2] if len(f) > 2 else -1 for f in face]
        if min(ni) >= 0:
            n = _unit(w0[..., None] * normals[ni[0]] + w1[..., None] * normals[ni[1]] + w2[..., None] * normals[ni[2]])
        else:
            normal = _unit(np.cross(vertices[ids[1]] - vertices[ids[0]], vertices[ids[2]] - vertices[ids[0]]))
            n = np.broadcast_to(normal, (*w0.shape, 3))
        lambert = .46 + .45 * np.maximum(n @ light, 0) + .13 * np.maximum(n @ fill, 0)
        specular = np.maximum(n @ half, 0) ** (35 if material == "body_color" else 65)
        reflectivity = .38 if material in ("body_color", "glass3", "metal2") else .1
        rgb = np.asarray(palette.get(material, (.4, .4, .4))) * lambert[..., None]
        rgb += specular[..., None] * reflectivity
        if material == "glass3":
            # Broad environment highlight gives windows depth without a texture download.
            fresnel = np.maximum(1 - np.abs(n @ to_eye), 0) ** 3
            rgb += fresnel[..., None] * np.array([.11, .17, .20])
        rgba = np.empty((*w0.shape, 4), np.uint8)
        rgba[..., :3] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        rgba[..., 3] = 255
        region = pixels[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        region[mask] = rgba[mask]
        local_z[mask] = z[mask]
    # Ground shadow is illustrative, independent of live sensor data.
    shadow = Image.new("RGBA", size)
    draw = ImageDraw.Draw(shadow)
    draw.ellipse((width * .18, height * .29, width * .82, height * .96), fill=(0, 4, 11, 90))
    shadow = shadow.filter(ImageFilter.GaussianBlur(width * .045))
    result = Image.alpha_composite(shadow, Image.fromarray(pixels, "RGBA"))
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", type=Path, default=ASSET_DIR / "source/car.obj")
    parser.add_argument("--output", type=Path, default=ASSET_PATH)
    args = parser.parse_args()
    print(build_ego_asset(args.obj, args.output))
