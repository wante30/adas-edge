"""Real native-pixel regressions; these fixtures are not an accuracy holdout."""
import hashlib
import json
from pathlib import Path

import cv2
import pytest

from competition_ai.multi_signal import Candidate, MultiSignalTracker
from competition_ai.signal_shape import ContourTemplateBackend

ROOT = Path(__file__).parent/'fixtures/real_signals'
ROWS = json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def backend():
    return ContourTemplateBackend()


@pytest.mark.parametrize('row', [r for r in ROWS if 'expected_color' in r], ids=lambda r: r['file'])
def test_real_native_crop_regression(row, backend):
    path = ROOT/row['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    attr = backend.classify(cv2.imread(str(path)))
    assert (attr.color, attr.shape) == (row['expected_color'], row['expected_shape']), attr


def test_same_color_faces_are_split_but_one_arrow_is_not(backend):
    image = cv2.imread(str(ROOT/'23_head.png'))
    assert len(backend.regions(image)) == 2
    h, w = image.shape[:2]
    tracker = MultiSignalTracker(backend=backend, confirm_updates=1)
    # Specialist's contained bulb must not discard the main model's full head.
    boxes = [Candidate((0, 0, w, h), .8, 1, 1_000_000_000),
             Candidate((85, 470, 188, 560), .95, 1, 1_000_000_000, 'specialist')]
    tracker.update(image, boxes, frame_id=1, timestamp_ns=1_000_000_000, context=('REAL', 1))
    rows = tracker.snapshot(1_000_000_000)['signals']
    assert len(rows) == 2
    assert all(r['color'] == 'GREEN' for r in rows)
    assert all(r['detectors'] == ['main', 'specialist'] for r in rows)
    assert any(r['shape'] == 'LEFT_ARROW' for r in rows)
    assert len(backend.regions(cv2.imread(str(ROOT/'02_0.png')))) == 1
