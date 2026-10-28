"""Same-frame crops for a manually calibrated video containing a car display.

This is per-video geometry, not a general screen detector. Screen pixels outside
the captured frame are black and explicitly counted as unavailable.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SplitFrame:
    road: np.ndarray
    navigation: np.ndarray
    metadata: dict


class SplitVideo:
    def __init__(self, profile: dict):
        self.geometry = profile.get("split_geometry", profile)
        config = self.geometry
        if config.get("schema") != "competition.video_split.v1":
            raise ValueError("Unsupported split geometry schema")
        self.source_size = tuple(config["source_size"])
        if len(self.source_size) != 2 or any(not isinstance(v, int) or v <= 0 for v in self.source_size):
            raise ValueError("source_size must contain positive integer width/height")
        self.navigation_size = tuple(config.get("navigation_size", [960, 540]))
        if self.navigation_size != (960, 540):
            raise ValueError("Navigation output must be 960 x 540")
        self.mask_margin = np.asarray(config.get("road_screen_mask_margin", [0, 0, 0, 0]), float)
        if self.mask_margin.shape != (4,) or not np.isfinite(self.mask_margin).all() or (self.mask_margin < 0).any() or (self.mask_margin > 512).any():
            raise ValueError("Screen mask margins must be four finite nonnegative pixels, at most 512")
        self.source_frame_count = config["source_frame_count"]
        if not isinstance(self.source_frame_count, int) or isinstance(self.source_frame_count, bool) or self.source_frame_count < 2:
            raise ValueError("source_frame_count must be an integer of at least two")
        self.keys = config["keyframes"]
        self.ids = [key["frame_id"] for key in self.keys]
        if (len(self.ids) < 2 or any(not isinstance(i, int) for i in self.ids)
                or self.ids != sorted(set(self.ids)) or self.ids[0] != 0
                or self.ids[-1] != self.source_frame_count - 1):
            raise ValueError("Geometry keyframes must strictly increase and cover the entire source")
        for key in self.keys:
            self._validate(np.asarray(key["road_rect"], float), np.asarray(key["navigation_quad"], float))
        self.target_quad = np.asarray([[0, 0], [959, 0], [959, 539], [0, 539]], np.float32)
        width, height = self.source_size
        self.source_mask = np.full((height, width), 255, np.uint8)

    def _validate(self, rect, quad):
        width, height = self.source_size
        if (rect.shape != (4,) or not np.isfinite(rect).all() or not (0 <= rect[0] < rect[2] <= width)
                or not (0 <= rect[1] < rect[3] <= height)):
            raise ValueError("Road rectangle must be nonempty and inside the source frame")
        # A camera pan can clip the display: allow bounded extrapolation of its
        # geometric corners, while warpPerspective exposes missing source pixels.
        if (quad.shape != (4, 2) or not np.isfinite(quad).all()
                or (quad[:, 0] < -width * .3).any() or (quad[:, 0] > width * 1.3).any()
                or (quad[:, 1] < -height * .3).any() or (quad[:, 1] > height * 1.3).any()
                or not cv2.isContourConvex(quad.astype(np.float32))
                or cv2.contourArea(quad.astype(np.float32), oriented=True) < 100):
            raise ValueError("Navigation quad must be bounded, convex, and ordered TL/TR/BR/BL")

    def split(self, frame: np.ndarray, frame_id: int) -> SplitFrame:
        if isinstance(frame_id, bool) or not isinstance(frame_id, (int, np.integer)) or not 0 <= frame_id < self.source_frame_count:
            raise ValueError("Source frame ID is outside the calibrated video")
        width, height = self.source_size
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.shape != (height, width, 3):
            raise ValueError("Expected a uint8 BGR source frame at the calibrated resolution")
        right_index = min(bisect_right(self.ids, int(frame_id)), len(self.ids) - 1)
        left_index = max(0, right_index - 1)
        left, right = self.keys[left_index], self.keys[right_index]
        alpha = (frame_id - left["frame_id"]) / (right["frame_id"] - left["frame_id"])
        rect_start, rect_end = np.asarray(left["road_rect"], float), np.asarray(right["road_rect"], float)
        quad_start, quad_end = np.asarray(left["navigation_quad"], float), np.asarray(right["navigation_quad"], float)
        rect = rect_start + (rect_end - rect_start) * alpha
        quad = quad_start + (quad_end - quad_start) * alpha
        # Input keys were validated; clamp only numerical interpolation noise.
        rect = np.clip(rect, [0, 0, 0, 0], [width, height, width, height])
        self._validate(rect, quad)
        x1, y1 = np.floor(rect[:2]).astype(int)
        x2, y2 = np.floor(rect[2:]).astype(int)
        road = frame[y1:y2, x1:x2].copy()
        margin_left, margin_top, margin_right, margin_bottom = self.mask_margin
        screen_polygon = quad + np.asarray([[-margin_left, -margin_top], [margin_right, -margin_top],
                                            [margin_right, margin_bottom], [-margin_left, margin_bottom]])
        road_mask = np.zeros(road.shape[:2], np.uint8)
        polygon = np.rint(screen_polygon - [x1, y1]).astype(np.int32)
        cv2.fillConvexPoly(road_mask, polygon, 255)
        # The same integer rasterization directly fills all BGR channels; avoid
        # allocating a boolean image and advanced-indexing every road pixel.
        cv2.fillConvexPoly(road, polygon, (0, 0, 0))
        matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), self.target_quad)
        navigation = cv2.warpPerspective(frame, matrix, self.navigation_size, flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        valid = cv2.warpPerspective(self.source_mask, matrix, self.navigation_size, flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        coverage = float(np.count_nonzero(valid) / valid.size)
        return SplitFrame(road, navigation, {
            "source_frame_id": int(frame_id), "road_source_frame_id": int(frame_id),
            "navigation_source_frame_id": int(frame_id), "source_size": list(self.source_size),
            "road_rect": [int(x1), int(y1), int(x2), int(y2)], "navigation_quad": quad.tolist(),
            "road_screen_exclusion_polygon": screen_polygon.tolist(),
            "road_masked_fraction": float(np.count_nonzero(road_mask) / road_mask.size),
            "road_screen_mask_pixels": "BLACK_EXCLUDED_FROM_INFERENCE_NOT_RECONSTRUCTED",
            "navigation_size": list(self.navigation_size), "navigation_transform": matrix.tolist(),
            "navigation_visible_fraction": coverage, "navigation_partially_outside_source": coverage < .999,
            "unobserved_navigation_pixels": "BLACK_PADDING_NOT_RECONSTRUCTED",
            "geometry_source": "MANUAL_PER_VIDEO_KEYFRAMES_INTERPOLATED",
            "geometry_keyframes": [left["frame_id"], right["frame_id"]], "geometry_alpha": float(alpha),
            "screen_content_semantics": "NOT_CLASSIFIED_BY_GEOMETRY",
        })
