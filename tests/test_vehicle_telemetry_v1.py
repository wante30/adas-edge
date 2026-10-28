"""Offline LVT1 checks; no FPGA, driver or hardware writes required."""
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / 'deployment/lubancat/qt_gpu_probe/vehicle_telemetry_v1.py'
spec = importlib.util.spec_from_file_location('vehicle_telemetry_v1', MODULE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# All values copied from the FPGA team's LVT1 handoff 2026-10-28.
EXAMPLE = {
    0x690: 0x3154564C, 0x694: 0x00010000, 0x698: 9, 0x69C: 7,
    0x6A0: 0x2A, 0x6A4: 0x00F90102, 0x6A8: 0x0000029D,
    0x6AC: 0x001AFFFF, 0x6B0: 0x01FF4802, 0x6B4: 7,
    0x6B8: 1001, 0x6BC: 500,
}


def test_official_example():
    v = m.read_snapshot(EXAMPLE.__getitem__)
    assert v['schema'] == 'pango.rk3566.vehicle_status.v1'
    assert v['lane_source'] == 'hsst'
    assert (v['left_line_count'], v['right_line_count']) == (2, 1)
    assert (v['left_nearest_type'], v['right_nearest_type']) == ('solid', 'dashed')
    assert v['current_action'] == 'brake' and v['last_sequence'] == 65535
    assert v['decision'] == 'warning' and v['reason_code'] == 8
    assert v['accept_count'] == 7 and v['brake_allow']


def test_stale_hsst_is_not_authoritative():
    w = EXAMPLE.copy()
    w[0x6A8] |= 1 << 1
    s = m.read_snapshot(w.__getitem__)
    assert s['lane_source'] == 'none'
    assert (s['left_line_count'], s['right_line_count']) == (None, None)
    assert (s['left_nearest_type'], s['right_nearest_type']) == ('unknown', 'unknown')


def test_zero_vs_null_and_no_side_fallback():
    w = EXAMPLE.copy()
    w[0x6A4] = ((1 << 20) | (1 << 21) | (1 << 22) | (1 << 23) |
                 (2 << 18) | (1 << 16) | (3 << 8))
    s = m.read_snapshot(w.__getitem__)
    assert s['left_line_count'] == 0 and s['left_nearest_type'] == 'unknown'
    assert s['right_line_count'] == 3 and s['right_nearest_type'] == 'dashed'
    w[0x6A4] &= ~(1 << 20)
    s = m.read_snapshot(w.__getitem__)
    assert s['left_line_count'] is None


def test_rk_source_does_not_invent_counts():
    w = EXAMPLE.copy()
    w[0x6A8] = (1 << 8) | (1 << 4) | (2 << 6) | (1 << 3)
    s = m.read_snapshot(w.__getitem__)
    assert s['lane_source'] == 'rk'
    assert (s['left_line_count'], s['right_line_count']) == (None, None)
    assert (s['left_nearest_type'], s['right_nearest_type']) == ('solid', 'dashed')


def test_inactive_action_does_not_use_historical_type():
    w = EXAMPLE.copy()
    w[0x6AC] &= ~(1 << 19)
    s = m.read_snapshot(w.__getitem__)
    assert s['current_action'] == 'none' and s['last_action'] == 'brake'
    assert s['decision'] == 'idle'


def test_generation_retry_then_coherence_failure():
    w = EXAMPLE.copy()
    generations = iter([9, 10, 10, 10])
    def reader(offset):
        return next(generations) if offset == 0x698 else w[offset]
    assert m.read_snapshot(reader)['generation'] == 10
    count = [0]
    def churn(offset):
        if offset == 0x698:
            count[0] += 1
            return count[0]
        return w[offset]
    with pytest.raises(m.CoherenceError):
        m.read_snapshot(churn)


@pytest.mark.parametrize('offset,value', [
    (0x690, 0xFFFFFFFF), (0x694, 0), (0x69C, 0),
])
def test_identity_mismatch_rejected(offset, value):
    w = EXAMPLE.copy()
    w[offset] = value
    with pytest.raises(m.TelemetryError):
        m.read_snapshot(w.__getitem__)


def test_zero_generation_is_valid():
    w = EXAMPLE.copy()
    w[0x698] = 0
    assert m.read_snapshot(w.__getitem__)['generation'] == 0


def test_atomic_status_file(tmp_path):
    path = tmp_path / 'status.json'
    m.write_status_file(path, m.read_snapshot(EXAMPLE.__getitem__))
    assert json.loads(path.read_text(encoding='utf-8'))['decision'] == 'warning'
    assert list(tmp_path.iterdir()) == [path]


def test_handoff_file_example_matches_decoder():
    handoff_path = MODULE.with_name('rk3566_pcie_vehicle_telemetry_v1.json')
    source = json.loads(handoff_path.read_text(encoding='utf-8'))
    assert source['schema'] == 'pango.rk3566.vehicle_telemetry_handoff.v1'
    fixture = EXAMPLE.copy()
    for offset, value in source['example']['registers_hex'].items():
        fixture[int(offset, 16)] = int(value, 16)
    actual = m.read_snapshot(fixture.__getitem__)
    for key, expected in source['example']['decoded'].items():
        assert actual[key] == expected, key
