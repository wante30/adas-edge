"""Optional RK3566 RGA preprocessing for the live YOLO detector."""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

_LIBRARY = None
_FUNCTION = None
_LOAD_ATTEMPTED = False
_DISABLED = False


def _rga_function():
    global _LIBRARY, _FUNCTION, _LOAD_ATTEMPTED
    if _LOAD_ATTEMPTED:
        return _FUNCTION
    _LOAD_ATTEMPTED = True
    library_path = Path(__file__).with_name("librga_preprocess.so")
    if not library_path.is_file():
        return None
    try:
        library = ctypes.CDLL(str(library_path))
        function = library.adas_rga_bgr_resize_rgb
        byte_pointer = ctypes.POINTER(ctypes.c_uint8)
        function.argtypes = [
            byte_pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            byte_pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        function.restype = ctypes.c_int
    except (AttributeError, OSError):
        return None
    _LIBRARY = library
    _FUNCTION = function
    return _FUNCTION


def preprocess_bgr_rga(image, size):
    """Return RGA RGB letterbox output, or None to request the CPU fallback."""
    global _DISABLED
    if _DISABLED or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        return None
    function = _rga_function()
    if function is None:
        return None

    if isinstance(size, int):
        model_width = model_height = size
    else:
        model_width, model_height = int(size[0]), int(size[1])
    height, width = image.shape[:2]
    if min(width, height, model_width, model_height) <= 0:
        return None

    scale = min(model_width / width, model_height / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    pad_x = (model_width - resized_width) // 2
    pad_y = (model_height - resized_height) // 2
    source = np.ascontiguousarray(image)
    destination = np.full((model_height, model_width, 3), 114, dtype=np.uint8)
    status = function(
        source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        width, height, source.strides[0],
        destination.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        model_width, model_height, destination.strides[0],
        pad_x, pad_y, resized_width, resized_height,
    )
    if status <= 0:
        _DISABLED = True
        return None
    return destination, scale, pad_x, pad_y
