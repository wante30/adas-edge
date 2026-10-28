"""RK3566 runtime port of Open-3D-Surround-View's calibrated LUT renderer.

Projection/mask equations adapted from Nick Pai's MIT licensed upstream
stitching_bev.py and stitching_bowl.py, pinned to f47349a.
See ../licenses/Open-3D-Surround-View-MIT.txt for the full notice.
Calibration generation stays on the PC; the board only loads NPZ maps.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

ROLES = ("front", "left", "right", "rear")


def build_luts(calibration: Mapping, output_dir: Path, source_size: tuple[int, int],
               output_size: tuple[int, int] = (320, 320), projection: str = "bowl") -> dict:
    """PC preparation: world grid -> calibrated fisheye pixels, once only."""
    if projection not in ("bev", "bowl"):
        raise ValueError("projection must be bev or bowl")
    width, height = output_size
    source_w, source_h = source_size
    if min(width, height, source_w, source_h) < 32:
        raise ValueError("invalid LUT resolution")
    # Same ISO 8855 axes and 10 m extent as the upstream sample.
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x = 5.0 - v * 10.0 / height
    y = 5.0 - u * 10.0 / width
    radius = np.hypot(np.maximum(np.abs(x) - 2.5, 0), np.maximum(np.abs(y) - 2.5, 0))
    z = np.maximum(radius - 1.5, 0) ** 2 * 0.5 if projection == "bowl" else np.zeros_like(x)
    points = np.stack((x, y, z), axis=-1).reshape(-1, 1, 3).astype(np.float32)
    intrinsic = calibration["intrinsic"]
    k = np.asarray(intrinsic["K"], dtype=np.float64)
    d = np.asarray(intrinsic["D"], dtype=np.float64).reshape(4, 1)
    raw = {}
    total = np.zeros((height, width), np.float32)
    for role in ROLES:
        ext = calibration["extrinsic"]["back" if role == "rear" else role]
        rvec = np.asarray(ext["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(ext["tvec"], dtype=np.float64).reshape(3, 1)
        r, _ = cv2.Rodrigues(rvec)
        camera_points = r @ points.reshape(-1, 3).T + tvec
        projected, _ = cv2.fisheye.projectPoints(points, rvec, tvec, k, d)
        projected = projected.reshape(height, width, 2)
        mx, my = projected[..., 0], projected[..., 1]
        valid = ((camera_points[2].reshape(height, width) > 0) &
                 (mx >= 0) & (mx < source_w - 1) & (my >= 0) & (my < source_h - 1))
        radial = np.hypot(mx - source_w / 2, my - source_h / 2) / (min(source_w, source_h) / 2 * 1.05)
        weight = (np.clip(1 - radial ** 2, 0, 1) * valid).astype(np.float32)
        raw[role] = (mx, my, weight)
        total += weight
    output_dir.mkdir(parents=True, exist_ok=True)
    for role, (mx, my, weight) in raw.items():
        np.savez_compressed(output_dir / f"lut_{role}.npz", map_x=mx, map_y=my,
                            weight=weight / np.maximum(total, 1e-6))
    manifest = {
        "schema": "calibrated-surround-lut/v1", "calibration_mode": "UPSTREAM_SYNTHETIC_CALIBRATED",
        "projection": projection, "source_size": list(source_size), "output_size": list(output_size),
        "extent_m": 10.0, "car_length_m": 4.8, "car_width_m": 1.9,
        "bowl": {"flat_rect_x": 2.5, "flat_rect_y": 2.5, "flat_margin": 1.5, "steepness": 0.5, "max_radius": 4.8},
        "upstream": "https://github.com/nick8592/Open-3D-Surround-View", "upstream_commit": "f47349a",
        "coverage_fraction": float(np.mean(total > 0)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class CalibratedSurroundComposer:
    """Four remaps + pre-normalized fusion, no per-frame projection maths."""

    def __init__(self, lut_dir: str | Path):
        root = Path(lut_dir)
        self.manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.width, self.height = self.manifest["output_size"]
        source_w, source_h = self.manifest["source_size"]
        self.input_width = min(source_w, 640)
        self.input_height = round(source_h * self.input_width / source_w)
        self.luts = {}
        for role in ROLES:
            with np.load(root / f"lut_{role}.npz", allow_pickle=False) as lut:
                # Pre-scale the maps to the inexpensive runtime input canvas.
                mx = lut["map_x"] * self.input_width / source_w
                my = lut["map_y"] * self.input_height / source_h
                maps = cv2.convertMaps(mx.astype(np.float32), my.astype(np.float32), cv2.CV_16SC2)
                weight = np.repeat(lut["weight"][..., None], 3, axis=2).astype(np.float32)
                self.luts[role] = (maps, weight)

    def metadata(self) -> dict:
        return {**self.manifest, "renderer": "open3d_calibrated_cached_lut",
                "input_size": [self.input_width, self.input_height]}

    def compose(self, frames: Mapping[str, np.ndarray]) -> np.ndarray:
        if set(frames) != set(ROLES):
            raise ValueError("calibrated surround requires all four synchronized camera roles")
        output = np.zeros((self.height, self.width, 3), np.float32)
        for role in ROLES:
            frame = frames[role]
            if frame.shape[:2] != (self.input_height, self.input_width):
                frame = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
            maps, weight = self.luts[role]
            warped = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            output += warped.astype(np.float32) * weight
        return np.clip(output, 0, 255).astype(np.uint8)

    def close(self) -> None:
        pass
