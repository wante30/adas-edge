from __future__ import annotations

import numpy as np

from tools.collect_rknn_zoo_onnx_predictions import prepare_model_zoo_onnx_input


def test_model_zoo_onnx_input_uses_black_padding_rgb_and_zero_to_one() -> None:
    image = np.empty((40, 80, 3), dtype=np.uint8)
    image[:, :, 0] = 10  # B
    image[:, :, 1] = 20  # G
    image[:, :, 2] = 30  # R

    tensor, scale, pad_x, pad_y = prepare_model_zoo_onnx_input(image, (96, 96))

    assert tensor.shape == (1, 3, 96, 96)
    assert tensor.dtype == np.float32
    assert scale == 1.2
    assert pad_x == 0
    assert pad_y == 24
    np.testing.assert_allclose(tensor[0, :, 24, 48], [30 / 255, 20 / 255, 10 / 255])
    np.testing.assert_array_equal(tensor[0, :, 0, :], 0.0)
