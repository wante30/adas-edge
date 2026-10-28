"""The regression auditor must detect lost outputs and unsafe equal failures."""
from copy import deepcopy

import pytest

from tools.compare_cockpit_inference import compare_rows, inventory


def row():
    return {'frame': 0, 'source_s': 0., 'controls': {'navigation': 'right'},
            'decision': {'direction': 'none', 'maneuver': 'stop'}, 'signal': {'light': 'UNKNOWN'},
            'binding_source': 'UNBOUND', 'hazards': {'events': []},
            'main_age_ms': 0., 'specialist_age_ms': 0.,
            'navigation': {'source_frame_id': 0, 'timestamp_ns': 0, 'direction': 'right',
                           'distance_m': 893., 'valid': True, 'inference_ms': 2.},
            'same_frame_split': {'road_rect': [1, 2, 30, 40]}, 'road_inference_frame_id': 0,
            'main': [{'class_name': 'car', 'bbox': [10., 20., 30., 40.], 'confidence': .8}],
            'specialist': [], 'hud_software_receipt': {'receiver_verified': True, 'hardware_writes': 0,
                                                    'request_word_hex': '0x00011909', 'projection': {'state': 265}}}


def test_only_wall_timings_are_excluded_and_source_age_still_matters():
    a, b = row(), row()
    b['navigation']['inference_ms'] = 30.
    assert compare_rows([a], [b])['pass']
    b['navigation']['timestamp_ns'] = 1
    assert not compare_rows([a], [b])['pass']


@pytest.mark.parametrize('field,change', [
    ('main', [{'class_name': 'car', 'bbox': [10., 20., 31., 40.], 'confidence': .8}]),
    ('main', [{'class_name': 'bus', 'bbox': [10., 20., 30., 40.], 'confidence': .8}]),
    ('main', [{'class_name': 'car', 'bbox': [10., 20., 30., 40.], 'confidence': .7}]),
    ('main', []), ('decision', {'direction': 'right', 'maneuver': 'turn_right'}),
    ('same_frame_split', {'road_rect': [1, 2, 31, 40]}),
])
def test_numerical_class_geometry_or_control_changes_fail(field, change):
    a, b = row(), row()
    b[field] = change
    assert not compare_rows([a], [b])['pass']


def test_missing_evidence_cannot_pass_by_matching_missing_fields():
    a = row()
    del a['main'][0]['bbox']
    assert not compare_rows([a], [deepcopy(a)])['pass']
    a = row()
    del a['navigation']
    assert not compare_rows([a], [deepcopy(a)])['pass']


def test_equal_failed_hud_receipts_or_missing_source_rows_fail():
    a = row()
    a['hud_software_receipt']['receiver_verified'] = False
    assert not compare_rows([a], [deepcopy(a)])['pass']
    assert not compare_rows([row()], [])['pass']


def test_explicit_numeric_tolerance_records_the_actual_delta():
    a, b = row(), row()
    b['main'][0]['bbox'][0] += 1e-6
    result = compare_rows([a], [b], bbox_atol=1e-5)
    assert result['pass'] and result['bbox_abs_difference'] > 0
    assert not compare_rows([a], [b])['pass']


def test_inventory_does_not_count_held_results_or_pause_as_new_inference():
    a, b = row(), row()
    b.update(frame=1, source_s=1/30, main_age_ms=1000/30, specialist_age_ms=1000/30)
    result = inventory([a, deepcopy(a), b])
    assert result['trace_rows'] == 3 and result['unique_source_frames'] == 2
    assert result['main_updates'] == result['specialist_updates'] == result['navigation_updates'] == 1
