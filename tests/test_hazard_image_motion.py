import cv2
import numpy as np

from competition_ai.image_motion import BackgroundMotion, ImageMotion
from competition_ai.image_tracks import ImageTracker


def test_background_pan_is_measured_and_cannot_be_stable():
    rng = np.random.default_rng(18)
    frame = rng.integers(0, 256, (180, 320, 3), dtype=np.uint8)
    motion = BackgroundMotion()
    assert not motion.update(frame, [], 0).valid
    moved = cv2.warpAffine(frame, np.float32([[1, 0, 8], [0, 1, 0]]), (320, 180))
    result = motion.update(moved, [], 100_000_000)
    assert result.valid and .23 < result.dx < .27 and not result.stable
    assert not motion.update(moved, [], 2_000_000_000).valid


def test_ambiguous_occlusion_restarts_identity_and_source_gap_expires_tracks():
    tracker = ImageTracker()
    def item(box):
        return {'class_name': 'car', 'bbox': box}
    tracks = tracker.update([item([80, 50, 130, 90]), item([120, 50, 170, 90])], 0, (180, 320), ImageMotion())
    old = {t.track_id for t in tracks}
    merged = tracker.update([item([100, 50, 150, 90])], 100_000_000, (180, 320), ImageMotion())
    assert merged[0].track_id not in old and len(merged[0].history) == 1
    later = tracker.update([item([100, 50, 150, 90])], 1_000_000_000, (180, 320), ImageMotion())
    assert later[0].track_id != merged[0].track_id
