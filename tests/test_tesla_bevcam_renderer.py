from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "adas_hmi_sim"))

from backend.scene_builder import SceneBuilder  # noqa: E402
from backend.tesla_bevcam_renderer import TeslaBevCamRenderer  # noqa: E402


def test_upstream_renderer_produces_bev_image_from_sceneframe() -> None:
    scene = SceneBuilder().build(
        0,
        0,
        [{"class_name": "car", "confidence": 0.9, "bbox": [400, 300, 500, 500]}],
        960,
        540,
        source="front",
        lane_metadata={
            "left": {"points": [{"x": -2.0, "z": z} for z in (4, 12, 30)]},
            "right": {"points": [{"x": 2.0, "z": z} for z in (4, 12, 30)]},
        },
    )
    image = TeslaBevCamRenderer().render(scene)
    assert image.shape == (720, 800, 3)
    assert image.dtype == np.uint8
    assert int(image.mean()) > 0
