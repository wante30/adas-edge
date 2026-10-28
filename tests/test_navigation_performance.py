"""Benchmark bookkeeping tests; no models/inference or performance claims."""
from copy import deepcopy

from tools.benchmark_navigation_reader import compare_observations, evaluate_reviewed


def observation(fid=0, **changes):
    row = {'source_frame_id': fid, 'timestamp_ns': 0, 'valid': True,
           'direction': 'right', 'distance_m': 126., 'road_name': '',
           'reason': 'PIXELS', 'raw_text': ['126'], 'confidence': .91,
           'inference_ms': 20., 'input_sha256': 'same_pixels'}
    row.update(changes)
    return row


def test_runtime_change_is_not_a_semantic_difference():
    result = compare_observations([observation()], [observation(inference_ms=2.)])
    assert result['regression_equal']
    assert result['accuracy_claim'] == 'NONE_OUTPUT_EQUALITY_IS_NOT_GROUND_TRUTH'


def test_same_wrong_prediction_does_not_become_ground_truth():
    wrong = observation(distance_m=26., raw_text=['26'])
    assert compare_observations([wrong], [deepcopy(wrong)])['regression_equal']
    result = evaluate_reviewed([wrong], {0: {'direction': 'right', 'distance_m': 126.}})
    assert result['positive_wrong_accepted'] == 1


def test_unknown_positive_is_a_rejection_not_correct_recognition():
    result = evaluate_reviewed([observation(valid=False, direction='none', distance_m=None)],
                               {0: {'direction': 'right', 'distance_m': 126.}})
    assert result['positive_rejected'] == 1 and result['positive_exact'] == 0


def test_side_view_false_instruction_is_a_failure():
    result = evaluate_reviewed([observation()], {0: {'direction': 'none', 'distance_m': None}})
    assert result['negative_false_accepted'] == 1 and result['wrong_acceptances']


def test_input_change_and_missing_samples_fail_regression():
    result = compare_observations([observation(), observation(10)],
                                  [observation(input_sha256='different_pixels')])
    assert not result['regression_equal']
    assert result['missing_frames'] == [10] and result['input_pixel_differences'] == [0]


def test_confidence_and_tensor_changes_are_reported():
    first = observation(tensor_digests=[{'input': 'a', 'output': 'b'}])
    second = observation(confidence=.90, tensor_digests=[{'input': 'a', 'output': 'c'}])
    result = compare_observations([first], [second])
    assert not result['regression_equal'] and result['confidence_exact_differences']
    assert result['tensor_digest_differences'] == [0]
