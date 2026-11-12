"""Arithmetic tests independent of the actual network and generated vectors."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from experiments.training.fpga_road_seg.handoff_integer import (
    conv_accumulate, integer_add, nearest, requantize, rne_shift,
)


def test_signed_ties_even_and_saturation():
    x = np.arange(-101, 102, dtype=np.int64)
    for shift in range(6):
        np.testing.assert_array_equal(rne_shift(x, shift), np.rint(x / (1 << shift)))
    a = np.array([-1000, -1, 0, 1, 1000]).reshape(1, 1, 1, -1)
    np.testing.assert_array_equal(requantize(a, [1], [0], 128).ravel(), [0, 127, 128, 129, 255])
    np.testing.assert_array_equal(requantize(a, [1], [0], 128, True).ravel(), [128, 128, 128, 129, 255])


@pytest.mark.parametrize('groups,cin,cout,stride', [(1, 3, 5, 2), (4, 4, 4, 1), (2, 4, 6, 2)])
def test_integer_conv_matches_exact_small_double_oracle(groups, cin, cout, stride):
    rng = np.random.default_rng(154)
    x = rng.integers(0, 256, (1, cin, 7, 9), dtype=np.uint8)
    w = rng.integers(-128, 128, (cout, cin // groups, 3, 3), dtype=np.int8)
    bias = rng.integers(-10000, 10000, cout, dtype=np.int32)
    actual = conv_accumulate(x, w, bias, 119, [stride, stride], [1, 1, 1, 1], groups)
    # Integers are exactly representable in float64 at these tested bounds.
    expected = F.conv2d(torch.tensor(x.astype(np.int64) - 119, dtype=torch.float64),
                        torch.tensor(w, dtype=torch.float64), torch.tensor(bias, dtype=torch.float64),
                        stride=stride, padding=1, groups=groups).numpy()
    np.testing.assert_array_equal(actual, expected)


def test_add_round_once_and_nearest_floor():
    a = np.array([0, 1, 2, 3], np.uint8).reshape(1, 1, 2, 2)
    stage = dict(input_zero_points=[1, 2], multipliers=[1 << 30, 1 << 30], right_shift=31, zero_point=100)
    expected = np.clip(np.rint((a.astype(float)-1)*.5 + (a.astype(float)-2)*.5)+100, 0, 255)
    np.testing.assert_array_equal(integer_add(a, a, stage), expected)
    np.testing.assert_array_equal(nearest(a, [1, 1, 4, 4]), a.repeat(2, 2).repeat(2, 3))


def test_accumulator_overflow_is_error():
    with pytest.raises(OverflowError, match='INT32'):
        conv_accumulate(np.ones((1, 1, 1, 1), np.uint8), np.ones((1, 1, 1, 1), np.int8),
                        np.array([2147483647]), 0, [1, 1], [0, 0, 0, 0], 1)
