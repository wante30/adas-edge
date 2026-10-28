"""Render a four-camera AVM video with calibrated inverse ground mapping.

The output map is built from A2D2 camera intrinsics, distortion, and vehicle
extrinsics.  No hand-authored destination quadrilaterals are used.  The
script also writes single-camera projections, camera ownership, and overlap
diagnostics before producing the video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping

import cv2
import numpy as np


OUT_W = 640
OUT_H = 640
SRC_W = 640
SRC_H = 320
NATIVE_W = 1920
NATIVE_H = 1208
ROLES = ("front", "left", "right", "rear")
CAMERA_NAMES = {
    "front": "front_center",
    "left": "side_left",
    "right": "side_right",
    "rear": "rear_center",
}
ROLE_DIRS = {"front": "front", "left": "left", "right": "right", "rear": "back"}


def _view_field(view: Mapping[str, object], name: str) -> np.ndarray:
    return np.asarray(view.get(name) or view.get(name.replace("_", "-")), dtype=np.float64)


def _project(q: np.ndarray, camera: Mapping[str, object], k_work: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = q[2].copy()
    z[np.abs(z) < 1e-12] = 1e-12
    x = q[0] / z
    y = q[1] / z
    r2 = x * x + y * y
    d = np.asarray(camera["Distortion"], dtype=np.float64).reshape(-1)
    if str(camera["Lens"]).lower() == "fisheye":
        r = np.sqrt(r2)
        theta = np.arctan(r)
        t2 = theta * theta
        td = theta * (1.0 + d[0] * t2 + d[1] * t2**2 + d[2] * t2**3 + d[3] * t2**4)
        scale = np.ones_like(r)
        nz = r > 1e-12
        scale[nz] = td[nz] / r[nz]
        xd, yd = x * scale, y * scale
    else:
        k3 = d[4] if d.size >= 5 else 0.0
        radial = 1.0 + d[0] * r2 + d[1] * r2**2 + k3 * r2**3
        xd = x * radial + 2 * d[2] * x * y + d[3] * (r2 + 2 * x * x)
        yd = y * radial + d[2] * (r2 + 2 * y * y) + 2 * d[3] * x * y
    return k_work[0, 0] * xd + k_work[0, 1] * yd + k_work[0, 2], k_work[1, 1] * yd + k_work[1, 2]


class CalibratedAVM:
    def __init__(self, calibration_path: Path, output_dir: Path):
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        cameras = calibration["cameras"]
        self.width, self.height = OUT_W, OUT_H
        self.source_width, self.source_height = SRC_W, SRC_H
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # A2D2 vehicle frame -> optical convention, matching avm_geometry.m.
        s = np.asarray([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        scale = np.asarray(
            [[SRC_W / NATIVE_W, 0.0, (SRC_W / NATIVE_W - 1.0) / 2.0],
             [0.0, SRC_H / NATIVE_H, (SRC_H / NATIVE_H - 1.0) / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        u, v = np.meshgrid(np.arange(OUT_W), np.arange(OUT_H))
        x = -7.5 + (u + 0.5) * 15.0 / OUT_W
        y = 10.0 - (v + 0.5) * 20.0 / OUT_H
        p = np.stack((x.reshape(-1), y.reshape(-1), np.zeros(OUT_W * OUT_H)), axis=0)
        self.body = (np.abs(x) <= 1.25) & (np.abs(y) <= 2.5)
        self.maps: Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        scores = []
        for role in ROLES:
            camera = cameras[CAMERA_NAMES[role]]
            view = camera["view"]
            ex = _view_field(view, "x_axis")
            ex = ex / np.linalg.norm(ex)
            ey = _view_field(view, "y_axis")
            ey = ey - ex * float(ex @ ey)
            ey = ey / np.linalg.norm(ey)
            ez = np.cross(ex, ey)
            ez = ez / np.linalg.norm(ez)
            r = np.vstack((-ey, -ez, ex))
            origin = s.T @ (_view_field(view, "origin") - np.zeros(3))
            rv = r @ s
            q = rv @ (p - origin.reshape(3, 1))
            k = np.asarray(camera["CamMatrixOriginal"], dtype=np.float64)
            k_work = scale @ k
            mx, my = _project(q, camera, k_work)
            mx = mx.reshape(OUT_H, OUT_W)
            my = my.reshape(OUT_H, OUT_W)
            qz = q[2].reshape(OUT_H, OUT_W)
            valid = (
                (qz > 0.05)
                & np.isfinite(mx)
                & np.isfinite(my)
                & (mx >= 0)
                & (my >= 0)
                & (mx < SRC_W - 1 - 1 / 16)
                & (my < SRC_H - 1 - 1 / 16)
                & ~self.body
            )
            mx[~valid] = 0.0
            my[~valid] = 0.0
            valid_u8 = valid.astype(np.uint8)
            feather = np.minimum(cv2.distanceTransform(valid_u8, cv2.DIST_L2, 3) / 30.0, 1.0)
            angular = np.maximum(qz / np.maximum(np.sqrt(np.sum(q * q, axis=0)).reshape(OUT_H, OUT_W), 1e-12), 0.0)
            score = feather * angular**2 * valid
            self.maps[role] = (mx.astype(np.float32), my.astype(np.float32), valid)
            scores.append(score)
        score_stack = np.stack(scores, axis=-1)
        order = np.argsort(score_stack, axis=-1)
        self.cam_a = order[..., -1].astype(np.uint8)
        self.cam_b = order[..., -2].astype(np.uint8)
        top = np.take_along_axis(score_stack, self.cam_a[..., None], axis=-1)[..., 0]
        second = np.take_along_axis(score_stack, self.cam_b[..., None], axis=-1)[..., 0]
        self.valid = top > 0
        self.blend = second > 0
        self.alpha = np.ones((OUT_H, OUT_W), np.float32)
        self.alpha[self.blend] = top[self.blend] / np.maximum(top[self.blend] + second[self.blend], 1e-9)
        self._write_diagnostics()

    def _write_diagnostics(self) -> None:
        palette = {
            0: (235, 80, 60),
            1: (70, 210, 90),
            2: (70, 150, 240),
            3: (210, 90, 210),
        }
        assignment = np.zeros((OUT_H, OUT_W, 3), np.uint8)
        for index, color in palette.items():
            assignment[self.valid & (self.cam_a == index)] = color
        assignment[self.body] = (48, 48, 48)
        cv2.imwrite(str(self.output_dir / "camera_assignment.png"), assignment)
        cv2.imwrite(str(self.output_dir / "blend_mask.png"), (self.blend.astype(np.uint8) * 255))
        coverage = {
            "valid_fraction": float(np.mean(self.valid)),
            "blend_fraction": float(np.mean(self.blend)),
            "body_fraction": float(np.mean(self.body)),
            "camera_primary_fraction": {
                role: float(np.mean(self.valid & (self.cam_a == index)))
                for index, role in enumerate(ROLES)
            },
        }
        (self.output_dir / "calibrated_mapping_report.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")

    @staticmethod
    def _prepare(frame: np.ndarray) -> np.ndarray:
        # Match the frozen FPGA working-resolution contract: direct 1920x1208
        # -> 640x320 resize, not a hand-authored crop or destination warp.
        return cv2.resize(frame, (SRC_W, SRC_H), interpolation=cv2.INTER_LINEAR)

    def _warp(self, role: str, frame: np.ndarray) -> np.ndarray:
        mx, my, _ = self.maps[role]
        return cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def render_single(self, role: str, frame: np.ndarray) -> np.ndarray:
        source = self._prepare(frame)
        warped = self._warp(role, source)
        _, _, valid = self.maps[role]
        canvas = np.zeros_like(warped)
        canvas[valid] = warped[valid]
        canvas[self.body] = (32, 38, 42)
        return canvas

    def render(self, frames: Mapping[str, np.ndarray]) -> np.ndarray:
        warped = np.stack([self._warp(role, self._prepare(frames[role])) for role in ROLES], axis=2)
        yy, xx = np.indices((OUT_H, OUT_W))
        a = warped[yy, xx, self.cam_a]
        b = warped[yy, xx, self.cam_b]
        alpha = self.alpha[..., None]
        output = (a.astype(np.float32) * alpha + b.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        # Neutral road fill only where no calibrated camera sees the ground.
        output[~self.valid] = (32, 38, 42)
        output[self.body] = (24, 29, 32)
        cv2.rectangle(output, (OUT_W // 2 - 48, OUT_H // 2 - 100), (OUT_W // 2 + 48, OUT_H // 2 + 100), (110, 145, 142), 2)
        return output

    def validate_seams(self, frames: Mapping[str, np.ndarray]) -> None:
        warped = np.stack([self._warp(role, self._prepare(frames[role])) for role in ROLES], axis=2)
        yy, xx = np.indices((OUT_H, OUT_W))
        a = warped[yy, xx, self.cam_a].astype(np.int16)
        b = warped[yy, xx, self.cam_b].astype(np.int16)
        difference = np.abs(a - b).mean(axis=2)
        values = difference[self.blend]
        result = {
            "overlap_pixels": int(np.count_nonzero(self.blend)),
            "mean_abs_bgr_difference": float(values.mean()) if values.size else 0.0,
            "p95_abs_bgr_difference": float(np.percentile(values, 95)) if values.size else 0.0,
            "note": "Difference is measured only where the calibrated top-two camera maps overlap on frame 0.",
        }
        (self.output_dir / "seam_validation_frame000.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--rear", "--back", dest="rear", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    renderer = CalibratedAVM(args.calibration, args.output_dir)
    paths = {"front": args.front, "left": args.left, "right": args.right, "rear": args.rear}
    captures = {role: cv2.VideoCapture(str(path)) for role, path in paths.items()}
    if not all(cap.isOpened() for cap in captures.values()):
        bad = [role for role, cap in captures.items() if not cap.isOpened()]
        raise SystemExit(f"cannot open camera videos: {', '.join(bad)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (OUT_W, OUT_H))
    if not writer.isOpened():
        raise SystemExit(f"cannot create {args.output}")

    first_frames: Dict[str, np.ndarray] | None = None
    count = 0
    try:
        for _ in range(max(1, args.frames)):
            frames: Dict[str, np.ndarray] = {}
            for role, cap in captures.items():
                ok, frame = cap.read()
                if not ok:
                    break
                frames[role] = frame
            if len(frames) != 4:
                break
            if first_frames is None:
                first_frames = dict(frames)
                renderer.validate_seams(first_frames)
            writer.write(renderer.render(frames))
            count += 1
    finally:
        writer.release()
        for cap in captures.values():
            cap.release()

    if first_frames is not None:
        for role in ROLES:
            cv2.imwrite(str(args.output_dir / f"single_{role}_projection.png"), renderer.render_single(role, first_frames[role]))
    report = {
        "output": str(args.output),
        "frames": count,
        "fps": args.fps,
        "resolution": [OUT_W, OUT_H],
        "mapping": "A2D2 calibrated inverse ground-plane projection",
        "camera_names": CAMERA_NAMES,
        "hardware_verified": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
