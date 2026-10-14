"""Small-image background flow, used to reject camera turns and scene cuts.

This affine estimate is an image measurement, not ego speed or odometry.
"""
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageMotion:
    valid: bool = False
    dx: float = 0.0  # fractions of frame width per source second
    dy: float = 0.0
    rotation_dps: float = 0.0
    scale_rate: float = 0.0
    inlier_ratio: float = 0.0
    reason: str = "warming_up"

    @property
    def stable(self):
        return (self.valid and abs(self.dx) < .12 and abs(self.dy) < .12
                and abs(self.rotation_dps) < 8 and abs(self.scale_rate) < .12)


class BackgroundMotion:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous, self.timestamp = None, None

    def update(self, frame, detections, timestamp_ns):
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        mask = np.full_like(gray, 255)
        mask[:18] = 0
        mask[153:] = 0  # bonnet/dashboard are fixed to the camera
        for item in detections:
            x1, y1, x2, y2 = item['bbox']
            cv2.rectangle(mask, (int(x1/width*320)-5, int(y1/height*180)-5),
                          (int(x2/width*320)+5, int(y2/height*180)+5), 0, -1)
        previous, before = self.previous, self.timestamp
        self.previous, self.timestamp = gray, timestamp_ns
        if previous is None:
            return ImageMotion()
        dt = (timestamp_ns-before)/1e9
        if not .03 <= dt <= .3:
            return ImageMotion(reason='source_gap')
        points = cv2.goodFeaturesToTrack(previous, 140, .015, 9, mask=mask)
        if points is None or len(points) < 16:
            return ImageMotion(reason='insufficient_background')
        forward, ok, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None)
        if forward is None:
            return ImageMotion(reason='flow_failed')
        backward, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, previous, forward, None)
        if backward is None:
            return ImageMotion(reason='flow_failed')
        keep = (ok.ravel() == 1) & (back_ok.ravel() == 1)
        keep &= np.linalg.norm(points-backward, axis=2).ravel() < 1.5
        a, b = points[keep].reshape(-1, 2), forward[keep].reshape(-1, 2)
        if len(a) < 16:
            return ImageMotion(reason='flow_inconsistent')
        # Require distributed background support, not one moving patch.
        if np.ptp(a[:, 0]) < 130 or np.ptp(a[:, 1]) < 45:
            return ImageMotion(reason='background_support_too_local')
        affine, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                    ransacReprojThreshold=2.)
        if affine is None or float(inliers.mean()) < .6:
            return ImageMotion(reason='scene_cut_or_parallax')
        center = affine @ np.array([160., 90., 1.]) - [160., 90.]
        scale = math.hypot(affine[0, 0], affine[1, 0])
        rotation = math.degrees(math.atan2(affine[1, 0], affine[0, 0]))
        return ImageMotion(True, float(center[0]/320/dt), float(center[1]/180/dt),
                           rotation/dt, math.log(max(scale, 1e-6))/dt,
                           float(inliers.mean()), 'background_affine')
