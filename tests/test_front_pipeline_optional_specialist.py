import numpy as np

from competition_ai.scenario import Lane
from tools.competition_console.pipeline import FrontPipeline


class MainDetector:
    def detect(self, frame):
        return [], 1.0


def test_front_pipeline_supports_single_model_without_specialist():
    pipeline = FrontPipeline(
        MainDetector(),
        None,
        main_every_request=True,
    )

    pipeline.advance(np.zeros((720, 1280, 3), dtype=np.uint8), 1, 1_000_000_000)
    snapshot = pipeline.snapshot(next(iter(Lane)), 1_000_000_000)

    assert snapshot.main == []
    assert snapshot.specialist == []
    assert snapshot.main_age_ms == 0
    assert snapshot.specialist_age_ms is None
