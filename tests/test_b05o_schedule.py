"""Unit tests for the B-05O video-time scheduling helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.b05o_schedule_compare import MAIN_HZ, SPEC_HZ, bucket  # noqa: E402


def test_bucket_boundaries_are_stable() -> None:
    assert bucket(0.0, MAIN_HZ) == 0
    assert bucket(0.099, MAIN_HZ) == 0
    assert bucket(0.1, MAIN_HZ) == 1
    assert bucket(0.199, MAIN_HZ) == 1
    assert bucket(0.2, MAIN_HZ) == 2


def test_spec_bucket_is_half_rate() -> None:
    assert bucket(0.199, SPEC_HZ) == 0
    assert bucket(0.2, SPEC_HZ) == 1
    assert bucket(0.399, SPEC_HZ) == 1
    assert bucket(0.4, SPEC_HZ) == 2


def test_ten_hz_has_exactly_ten_inferences_per_second() -> None:
    fps = 30.0
    frames = int(fps * 10)
    buckets = [bucket(i / fps, MAIN_HZ) for i in range(frames)]
    changes = sum(1 for previous, current in zip(buckets, buckets[1:]) if current != previous)
    # bucket 0 starts at frame 0; each later second adds ten new buckets.
    assert changes == 99


def test_no_future_frame_can_be_displayed() -> None:
    # Held detections carry t_infer from the frame they were produced on.
    fps = 30.0
    last = -1
    displayed_ages = []
    for index in range(30):
        t = index / fps
        current = bucket(t, MAIN_HZ)
        if current != last:
            last = current
            t_infer = t
        displayed_ages.append(t - t_infer)
    assert min(displayed_ages) == 0.0
    assert max(displayed_ages) < 0.1 + 1e-9
