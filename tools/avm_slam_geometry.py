"""Ground inverse maps for the supplied AVM-SLAM calibration (PC export).

Extrinsics are interpreted as row-major world-to-optical [R|t], metres.
That interpretation must be checked by camera centres and overlap diagnostics.
KB4 is an explicit, empirically checked hypothesis, not a schema field.
Formula: https://docs.opencv.org/4.13.0/db/d58/group__calib3d__fisheye.html
"""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np


def calibration(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    cameras = sorted(data["camParams"], key=lambda c: c["camId"])
    if [c["camId"] for c in cameras] != list(range(4)):
        raise ValueError("Need four unique camera IDs 0..3")
    report = []
    for c in cameras:
        e = np.asarray(c["Extrinsics"], float).reshape(3, 4)
        r, t = e[:, :3], e[:, 3]
        if not np.allclose(r @ r.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1):
            raise ValueError("Extrinsic rotation is not SO(3)")
        center = -r.T @ t
        if not 0.2 < center[2] < 2.5:
            raise ValueError("Camera height incompatible with metre world-to-camera interpretation")
        report.append(dict(cam_id=c["camId"], role=c["camInfo"], center_m=center.tolist(),
                           forward_axis=r[2].tolist(), rotation_determinant=float(np.linalg.det(r))))
    return data, cameras, report


def ground_maps(path: Path, size=(640, 640), source_size=(640, 320),
                extent=(7.13, 7.13), model="kb4"):
    data, cameras, report = calibration(path)
    width, height = size
    sw, sh = source_size
    iw, ih = data["inputCfg"][0]["rgbImgW"], data["inputCfg"][0]["rgbImgH"]
    x, y = np.meshgrid((np.arange(width) + .5) * 2 * extent[0] / width - extent[0],
                       extent[1] - (np.arange(height) + .5) * 2 * extent[1] / height)
    world = np.stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    # Match supplied VisualStart rectangle, not an invented vehicle image.
    half_w = float(cameras[0]["VisualStart"])
    half_l = float(cameras[1]["VisualStart"])
    body = (abs(x) < half_w) & (abs(y) < half_l)
    maps, masks, scores = [], [], []
    for c in cameras:
        e = np.asarray(c["Extrinsics"], float).reshape(3, 4)
        q = e[:, :3] @ world + e[:, 3:4]
        z = np.maximum(q[2], 1e-9)
        a, b = q[0] / z, q[1] / z
        d = np.asarray(c["Distortion"])
        if model == "kb4":
            radius = np.hypot(a, b)
            theta = np.arctan(radius)
            t2 = theta * theta
            td = theta * (1 + d[0]*t2 + d[1]*t2**2 + d[2]*t2**3 + d[3]*t2**4)
            scale = np.divide(td, radius, out=np.ones_like(td), where=radius > 1e-12)
            u, v = a * scale, b * scale
        elif model == "radtan4":
            r2 = a*a + b*b
            radial = 1 + d[0]*r2 + d[1]*r2*r2
            u = a*radial + 2*d[2]*a*b + d[3]*(r2+2*a*a)
            v = b*radial + d[2]*(r2+2*b*b) + 2*d[3]*a*b
        else:
            raise ValueError(model)
        k = np.asarray(c["oriIntrinsics"]).reshape(3, 3)
        u = ((k[0, 0]*u + k[0, 2] + .5) * sw/iw - .5).reshape(height, width)
        v = ((k[1, 1]*v + k[1, 2] + .5) * sh/ih - .5).reshape(height, width)
        valid = ((q[2].reshape(height, width) > .03) & np.isfinite(u) & np.isfinite(v)
                 & (u >= 0) & (v >= 0) & (u < sw-1-1/16) & (v < sh-1-1/16) & ~body)
        uv = np.stack((np.where(valid, u, 0), np.where(valid, v, 0)), -1).astype(np.float32)
        feather = np.minimum(cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 3)/20, 1)
        angle = (q[2] / np.maximum(np.linalg.norm(q, axis=0), 1e-9)).reshape(height, width)
        maps.append(uv); masks.append(valid); scores.append(feather * np.maximum(angle, 0)**2)
    return np.stack(maps), np.stack(masks), np.stack(scores), body, report


def warp_inputs(images, maps):
    return np.stack([cv2.remap(im, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR)
                     for im, uv in zip(images, maps)])


def blend(warped, scores):
    order = np.argsort(scores, axis=0, kind="stable")
    ca, cb = order[-1], order[-2]
    sa = np.take_along_axis(scores, ca[None], axis=0)[0]
    sb = np.take_along_axis(scores, cb[None], axis=0)[0]
    alpha = np.divide(sa, sa+sb, out=np.ones_like(sa), where=sa+sb > 0)
    ia = np.take_along_axis(warped, ca[None, ..., None], axis=0)[0]
    ib = np.take_along_axis(warped, cb[None, ..., None], axis=0)[0]
    image = np.clip(np.rint(ia*alpha[..., None] + ib*(1-alpha[..., None])), 0, 255).astype(np.uint8)
    image[sa <= 0] = 0
    return image, ca, cb, alpha, sa > 0, sb > 0
