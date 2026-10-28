"""Adapter for the upstream Tesla-BevCam-Clone renderer.

The competition HMI keeps its SceneFrame protocol and detector ownership, but
exposes the checked-out upstream renderer as the primary BEV view. This file
does not copy or modify the upstream source; it only adapts SceneFrame data.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_MAIN = ROOT / ".deps" / "Tesla-BevCam-Clone" / "main.py"
UPSTREAM_COMMIT = "e99318bde342c665382950d46a3fc01b469b939a"


def _load_upstream() -> Any:
    if not UPSTREAM_MAIN.exists():
        raise FileNotFoundError(f"missing upstream renderer: {UPSTREAM_MAIN}")
    spec = importlib.util.spec_from_file_location("adas_tesla_bevcam_upstream", UPSTREAM_MAIN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load upstream renderer: {UPSTREAM_MAIN}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TeslaBevCamRenderer:
    """Render SceneFrame objects with the upstream OpenCV BEV UI."""

    name = "Tesla-BevCam-Clone"
    validation = "UPSTREAM_RENDERER_WITH_LOCAL_SCENEFRAME_OBJECTS"

    def __init__(self) -> None:
        self.upstream = _load_upstream()

    @staticmethod
    def _lane_polys(scene: Any) -> dict[int, np.ndarray]:
        result: dict[int, np.ndarray] = {}
        # SceneFrame has left/right boundaries; upstream calls the ego-lane
        # boundaries 1/2. Fit x(z) = a*z^2 + b*z + c.
        for index, side in ((1, "left"), (2, "right")):
            lane = scene.lanes.get(side)
            points = getattr(lane, "points", []) if lane is not None else []
            if len(points) < 3:
                continue
            z = np.asarray([float(point.z) for point in points], dtype=np.float64)
            x = np.asarray([float(point.x) for point in points], dtype=np.float64)
            if np.unique(z).size < 3:
                continue
            result[index] = np.asarray(np.polyfit(z, x, 2), dtype=np.float64)
        if 1 in result and 2 in result:
            # The current SceneFrame distances are explicitly pseudo-distance.
            # Reject a visibly impossible lane width instead of stretching the
            # upstream blue path across the whole BEV.
            left = float(np.polyval(result[1], 5.0))
            right = float(np.polyval(result[2], 5.0))
            width = right - left
            if not (2.0 <= width <= 6.5 and left < right):
                return {}
        return result

    def render(self, scene: Any) -> np.ndarray:
        tracked = [
            {"ego_x": float(item.world.x), "ego_y": float(item.world.z), "cls": item.class_name, "conf": float(item.confidence)}
            for item in scene.objects
            if item.class_name in self.upstream.DIMS
            and float(item.alpha) >= 0.35
            and float(item.confidence) >= 0.28
            and abs(float(item.world.x)) <= 8.2
            and 2.0 <= float(item.world.z) <= 58.0
        ]
        speed_mps = max(0.0, float(scene.ego.speed)) / 3.6
        self.upstream.update_bev_camera(speed_mps)
        ego_state = {"speed_mps": speed_mps, "accel_x": 0.0, "heading": 0.0, "lat": 0.0, "lon": 0.0}
        return self.upstream.render_tesla_ui(tracked, ego_state, 0.0, self._lane_polys(scene) or None)


__all__ = ["TeslaBevCamRenderer", "UPSTREAM_COMMIT"]
