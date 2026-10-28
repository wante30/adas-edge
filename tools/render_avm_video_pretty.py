"""Make a visually polished 640x640 four-camera AVM preview.

This renderer deliberately uses a planar homography and feathered seams.  It
is meant for a readable PC demo from the supplied four videos; the FPGA
fixed-LUT renderer remains a separate, bit-exact engineering artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


W, H = 640, 640
IN_W, IN_H = 640, 320
ROLES = ("front", "left", "right", "rear")
QUADS = {
    "front": np.float32([[115, 18], [525, 18], [610, 320], [30, 320]]),
    "rear": np.float32([[30, 320], [610, 320], [525, 622], [115, 622]]),
    "left": np.float32([[18, 18], [305, 170], [305, 470], [18, 622]]),
    "right": np.float32([[335, 170], [622, 18], [622, 622], [335, 470]]),
}


def crop_resize(frame: np.ndarray) -> np.ndarray:
    """Crop source to the requested 2:1 camera aspect before resizing."""
    h, w = frame.shape[:2]
    target_h = min(h, int(round(w / 2.0)))
    y0 = max(0, (h - target_h) // 2)
    cropped = frame[y0 : y0 + target_h, :]
    return cv2.resize(cropped, (IN_W, IN_H), interpolation=cv2.INTER_AREA)


def load_undistort_maps(calibration: Path | None, width: int, height: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if calibration is None or not calibration.is_file():
        return {}
    raw = json.loads(calibration.read_text(encoding="utf-8"))["cameras"]
    camera_names = {"front": "front_center", "left": "side_left", "right": "side_right", "rear": "rear_center"}
    maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for role, name in camera_names.items():
        camera = raw[name]
        k = np.asarray(camera["CamMatrixOriginal"], dtype=np.float64)
        d = np.asarray(camera["Distortion"], dtype=np.float64).reshape(-1, 1)
        identity = np.eye(3, dtype=np.float64)
        if str(camera["Lens"]).lower() == "fisheye":
            map_x, map_y = cv2.fisheye.initUndistortRectifyMap(k, d, identity, k, (width, height), cv2.CV_16SC2)
        else:
            map_x, map_y = cv2.initUndistortRectifyMap(k, d, identity, k, (width, height), cv2.CV_16SC2)
        maps[role] = (map_x, map_y)
    return maps


def build_masks() -> dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:H, 0:W]
    angle = np.arctan2(yy - H / 2.0, xx - W / 2.0)
    anchor_angles = {"front": -np.pi / 2.0, "left": np.pi, "right": 0.0, "rear": np.pi / 2.0}
    masks: dict[str, np.ndarray] = {}
    for role, quad in QUADS.items():
        hard = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(hard, quad.astype(np.int32), 255)
        # A short feather keeps only the actual seam soft.  The angular term
        # gives each camera a natural 90-degree ownership sector, preventing
        # large front/side/rear overlaps from becoming double images.
        distance = cv2.distanceTransform(hard, cv2.DIST_L2, 3)
        feather = np.clip(distance / 24.0, 0.0, 1.0)
        delta = np.arctan2(np.sin(angle - anchor_angles[role]), np.cos(angle - anchor_angles[role]))
        angular = np.clip(np.cos(delta), 0.0, 1.0) ** 4
        masks[role] = (feather * angular).astype(np.float32)
    return masks


def make_base() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    radial = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    road = np.clip(1.0 - radial * 0.30, 0.68, 1.0)
    return np.stack((18.0 * road, 26.0 * road, 32.0 * road), axis=-1).astype(np.float32)


def compose(frames: dict[str, np.ndarray], masks: dict[str, np.ndarray], base: np.ndarray) -> np.ndarray:
    canvas = base.copy()
    weights = np.zeros((H, W), np.float32)
    source = np.float32([[0, 0], [IN_W - 1, 0], [IN_W - 1, IN_H - 1], [0, IN_H - 1]])
    for role in ROLES:
        matrix = cv2.getPerspectiveTransform(source, QUADS[role])
        warped = cv2.warpPerspective(frames[role], matrix, (W, H), borderMode=cv2.BORDER_CONSTANT)
        alpha = masks[role]
        canvas += warped.astype(np.float32) * alpha[..., None]
        weights += alpha
    valid = weights > 1e-4
    output = np.clip(base, 0, 255).astype(np.uint8)
    if np.any(valid):
        blended = canvas[valid] / weights[valid, None]
        output[valid] = np.clip(blended, 0, 255).astype(np.uint8)

    # Clean, stable vehicle mask and orientation cue.
    cx, cy = W // 2, H // 2
    car_w, car_h = 48, 92
    cv2.ellipse(output, (cx, cy + 4), (car_w + 8, car_h + 8), 0, 0, 360, (8, 12, 15), -1)
    cv2.rectangle(output, (cx - car_w, cy - car_h), (cx + car_w, cy + car_h), (20, 27, 30), -1)
    cv2.rectangle(output, (cx - car_w, cy - car_h), (cx + car_w, cy + car_h), (190, 220, 214), 2)
    cv2.arrowedLine(output, (cx, cy - car_h - 8), (cx, cy - car_h - 42), (66, 220, 227), 3, tipLength=0.35)
    cv2.putText(output, "AVM 640", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 240, 235), 2, cv2.LINE_AA)
    for role, pos in {"front": (W // 2 - 32, 70), "left": (75, H // 2), "right": (515, H // 2), "rear": (W // 2 - 25, 585)}.items():
        cv2.putText(output, role.upper(), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 238, 233), 1, cv2.LINE_AA)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--rear", "--back", dest="rear", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    paths = {"front": args.front, "left": args.left, "right": args.right, "rear": args.rear}
    captures = {role: cv2.VideoCapture(str(path)) for role, path in paths.items()}
    if not all(cap.isOpened() for cap in captures.values()):
        bad = [role for role, cap in captures.items() if not cap.isOpened()]
        raise SystemExit(f"cannot open camera videos: {', '.join(bad)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"cannot create {args.output}")
    masks = build_masks()
    base = make_base()
    undistort_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    first = None
    count = 0
    try:
        for _ in range(max(1, args.frames)):
            frames = {}
            for role, cap in captures.items():
                ok, frame = cap.read()
                if not ok:
                    break
                if not undistort_maps:
                    undistort_maps = load_undistort_maps(args.calibration, frame.shape[1], frame.shape[0])
                if role in undistort_maps:
                    frame = cv2.remap(frame, undistort_maps[role][0], undistort_maps[role][1], cv2.INTER_LINEAR)
                frames[role] = crop_resize(frame)
            if len(frames) != 4:
                break
            output = compose(frames, masks, base)
            writer.write(output)
            first = output if first is None else first
            count += 1
    finally:
        writer.release()
        for cap in captures.values():
            cap.release()
    if args.preview and first is not None:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview), first)
    print({"output": str(args.output), "frames": count, "fps": args.fps, "resolution": [W, H], "renderer": "cropped_planar_feathered_demo", "hardware_verified": False})


if __name__ == "__main__":
    main()
