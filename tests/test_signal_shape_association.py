import cv2
import numpy as np
import pytest

from competition_ai.signal_shape import ContourTemplateBackend
from competition_ai.signal_association import associate


def arrow_image():
    image = np.zeros((80, 80, 3), np.uint8)
    # Independent polygon, not imported from the classifier's templates.
    points = np.array([(34, 66), (46, 66), (46, 34), (63, 34),
                       (40, 10), (17, 34), (34, 34)], np.int32)
    cv2.fillPoly(image, [points], (0, 255, 0))
    return image


@pytest.mark.parametrize('rot,shape', [(0, 'STRAIGHT_ARROW'), (1, 'LEFT_ARROW'), (3, 'RIGHT_ARROW')])
def test_clear_native_arrow(rot, shape):
    attributes = ContourTemplateBackend().classify(np.rot90(arrow_image(), rot).copy())
    assert attributes.color == 'GREEN'
    assert attributes.shape == shape, attributes


def test_round_small_blur_and_multiple_regions():
    backend = ContourTemplateBackend()
    image = np.zeros((80, 80, 3), np.uint8)
    cv2.circle(image, (40, 40), 20, (0, 0, 255), -1)
    assert backend.classify(image).shape == 'ROUND'
    assert backend.classify(cv2.resize(arrow_image(), (12, 12))).shape == 'UNKNOWN'
    assert backend.classify(cv2.GaussianBlur(arrow_image(), (21, 21), 7)).shape == 'UNKNOWN'
    cv2.circle(image, (12, 12), 8, (0, 0, 255), -1)
    assert backend.classify(image).shape == 'UNKNOWN'


def signal(tid=1, color='GREEN', shape='LEFT_ARROW'):
    return dict(track_id=tid, source_id='Q4', bbox=[10, 10, 40, 60],
                shape=shape, color=color, valid=True, confidence=.95,
                color_confidence=.96, shape_confidence=.90, reason='OK',
                timestamp_ns=1_000_000_000, expires_at_ns=2_200_000_000)


def binding(**changes):
    row = dict(source_id='Q4', context=['model', 1], coordinate_space='q4_native',
               head_id='head-A', group_id='group-A', calibration_id='cal-1',
               bbox=[10, 10, 40, 60], confidence=.95, timestamp_ns=1_000_000_000,
               expires_at_ns=1_500_000_000, ego_road_confirmed=True, viewpoint_confirmed=True,
               directions=['left'], evidence_type='CALIBRATED_VISUAL_HEAD')
    return dict(row, **changes)


def result(signals):
    return dict(signals=signals, context=['model', 1])


def test_visual_arrow_is_not_ego_lane_evidence_and_navigation_cannot_grant():
    out = associate(result([signal()]), now_ns=1_100_000_000, navigation_intent='left')
    assert out['signals'][0]['direction'] == 'left'
    assert out['signals'][0]['ego_relevant'] == 'UNKNOWN'
    assert out['directions']['left']['color'] == 'UNKNOWN'


def test_left_green_straight_red_with_explicit_head_evidence():
    straight = dict(signal(2, 'RED', 'STRAIGHT_ARROW'), bbox=[60, 10, 90, 60])
    bindings = [binding(), binding(head_id='head-B', bbox=straight['bbox'], directions=['straight'])]
    out = associate(result([signal(), straight]), now_ns=1_100_000_000, bindings=bindings)
    assert out['directions']['left']['color'] == 'GREEN'
    assert out['directions']['straight']['color'] == 'RED'
    assert out['directions']['right']['color'] == 'UNKNOWN'


def test_duplicate_groups_agree_conflict_and_round():
    first = signal()
    second = dict(signal(2), bbox=[60, 10, 90, 60])
    bindings = [binding(), binding(head_id='B', group_id='redundant', bbox=second['bbox'])]
    out = associate(result([first, second]), now_ns=1_100_000_000, bindings=bindings)
    assert out['directions']['left']['color'] == 'GREEN'
    second['color'] = 'RED'
    out = associate(result([first, second]), now_ns=1_100_000_000, bindings=bindings)
    assert out['directions']['left']['reason'] == 'CONFLICTING_ASSOCIATED_COLORS'
    assert out['directions']['left']['signal_ids'] == [1, 2]
    assert associate(result([signal(shape='ROUND')]), now_ns=1_100_000_000)['directions']['left']['color'] == 'UNKNOWN'


@pytest.mark.parametrize('changes', [dict(context=['other', 1]), dict(timestamp_ns=1_200_000_000),
    dict(expires_at_ns=1_050_000_000), dict(evidence_type='NAVIGATION'),
    dict(viewpoint_confirmed=False), dict(ego_road_confirmed=False), dict(confidence=float('nan')),
    dict(directions=['straight'])])
def test_stale_wrong_epoch_or_unsupported_evidence(changes):
    out = associate(result([signal()]), now_ns=1_100_000_000, bindings=[binding(**changes)])
    assert out['directions']['left']['color'] == 'UNKNOWN'


def test_ambiguous_head_bindings_fail_closed():
    out = associate(result([signal()]), now_ns=1_100_000_000,
                    bindings=[binding(), binding(head_id='other')])
    assert out['signals'][0]['association_reason'] == 'AMBIGUOUS_HEAD_ROAD_BINDING'
