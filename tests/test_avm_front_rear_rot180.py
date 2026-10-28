"""Exact orientation compensation, including fractional and mixed-camera pixels."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from avm_slam_lut import FIELDS, Replay, encode, decode
from avm_fpga_fixed_model import render
from export_avm_front_rear_rot180 import rotate_input_coordinates


def sample_fields():
    rng = np.random.default_rng(20261015)
    n = 2048
    f = {key: np.zeros(n, np.uint32) for key in FIELDS}
    f['valid'][:] = 1
    f['blend'][:] = 1
    f['alpha_q'] = rng.integers(0, 65, n, dtype=np.uint32)
    for side in ('a', 'b'):
        f[f'cam_{side}'] = (np.arange(n, dtype=np.uint32) // 256 + (side == 'b')) % 4
        f[f'addr_{side}'] = rng.integers(1, 319, n, dtype=np.uint32) * 640 + rng.integers(1, 639, n, dtype=np.uint32)
        f[f'fx_{side}'] = np.arange(n, dtype=np.uint32) % 16
        f[f'fy_{side}'] = (np.arange(n, dtype=np.uint32) // 16) % 16
    f['valid'][::19] = 0
    f['blend'][::19] = 0
    f['blend'][1] = 0
    f['alpha_q'][1] = 64
    return decode(encode(f))


def test_all_fraction_pairs_and_blends_match_independent_reference():
    f = sample_fields()
    changed = decode(encode(rotate_input_coordinates(f)))
    rng = np.random.default_rng(3566)
    images = [rng.integers(0, 65536, (320, 640), dtype=np.uint16) for _ in range(4)]
    rotated = [im[::-1, ::-1].copy() if i in (1, 3) else im for i, im in enumerate(images)]
    expected = render(f, images, 64, 32)
    assert np.array_equal(Replay(changed, (64, 32)).render(rotated), expected)
    assert np.array_equal(render(changed, rotated, 64, 32), expected)


def test_unselected_and_inactive_samples_and_controls_unchanged():
    f = sample_fields()
    changed = rotate_input_coordinates(f)
    for key in ('cam_a', 'cam_b', 'valid', 'blend', 'alpha_q'):
        assert np.array_equal(f[key], changed[key])
    for side in ('a', 'b'):
        selected = (f['valid'] != 0) & np.isin(f[f'cam_{side}'], (1, 3))
        if side == 'b':
            selected &= f['blend'] != 0
        for name in ('addr', 'fx', 'fy'):
            key = f'{name}_{side}'
            assert np.array_equal(f[key][~selected], changed[key][~selected])


def test_rotation_twice_restores_exact_packed_bytes():
    f = sample_fields()
    assert np.array_equal(encode(f), encode(rotate_input_coordinates(rotate_input_coordinates(f))))


@pytest.mark.parametrize('address', [640 * 10, 10])
def test_unrepresentable_last_row_or_column_is_rejected(address):
    f = sample_fields()
    f['cam_a'][1] = 1
    f['addr_a'][1] = address
    f['fx_a'][1] = f['fy_a'][1] = 0
    with pytest.raises(ValueError, match='border'):
        rotate_input_coordinates(f)
