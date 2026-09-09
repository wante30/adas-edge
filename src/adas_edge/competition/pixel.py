from __future__ import annotations

from enum import IntEnum

import cv2
import numpy as np


class PixelFormat(IntEnum):
    UNKNOWN = 0
    BGR888 = 1
    RGB888 = 2
    NV12 = 3
    RGB565 = 4


def _check_payload(payload: bytes, expected: int) -> None:
    if len(payload) != expected:
        raise ValueError(f"payload size {len(payload)} != expected {expected}")


def payload_to_bgr(payload: bytes, width: int, height: int, pixel_format: PixelFormat | int, stride: int | None = None) -> np.ndarray:
    fmt = PixelFormat(pixel_format)
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    stride = int(stride or (width * (2 if fmt is PixelFormat.RGB565 else 3)))
    if fmt is PixelFormat.BGR888:
        _check_payload(payload, stride * height)
        return np.frombuffer(payload, np.uint8).reshape(height, stride)[:, : width * 3].reshape(height, width, 3).copy()
    if fmt is PixelFormat.RGB888:
        raw = payload_to_bgr(payload, width, height, PixelFormat.BGR888, stride)
        return raw[:, :, ::-1].copy()
    if fmt is PixelFormat.NV12:
        _check_payload(payload, stride * height + stride * ((height + 1) // 2))
        return cv2.cvtColor(np.frombuffer(payload, np.uint8).reshape(height * 3 // 2, stride)[:, :width], cv2.COLOR_YUV2BGR_NV12)
    if fmt is PixelFormat.RGB565:
        _check_payload(payload, stride * height)
        words = np.frombuffer(payload, np.uint8).reshape(height, stride)[:, : width * 2].reshape(height, width, 2)
        value = words[:, :, 0].astype(np.uint16) | (words[:, :, 1].astype(np.uint16) << 8)
        r = ((value >> 11) & 0x1F).astype(np.uint16)
        g = ((value >> 5) & 0x3F).astype(np.uint16)
        b = (value & 0x1F).astype(np.uint16)
        return np.stack(((b * 255 + 15) // 31, (g * 255 + 31) // 63, (r * 255 + 15) // 31), axis=-1)
    raise ValueError(f"unsupported pixel format: {pixel_format}")


def bgr_to_payload(frame: np.ndarray, pixel_format: PixelFormat | int, stride: int | None = None) -> bytes:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("frame must be uint8 HxWx3")
    fmt = PixelFormat(pixel_format)
    height, width = frame.shape[:2]
    if fmt is PixelFormat.BGR888:
        return frame.tobytes()
    if fmt is PixelFormat.RGB888:
        return frame[:, :, ::-1].tobytes()
    if fmt is PixelFormat.NV12:
        height, width = frame.shape[:2]
        if width % 2 or height % 2:
            raise ValueError("NV12 requires even width and height")
        i420 = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
        y_plane = i420[:height, :width]
        chroma = i420[height:, :width].reshape(height // 2, width)
        uv = np.empty_like(chroma)
        uv[:, 0::2] = chroma[:, : width // 2]
        uv[:, 1::2] = chroma[:, width // 2 :]
        return np.vstack((y_plane, uv)).tobytes()
    if fmt is PixelFormat.RGB565:
        r = (frame[:, :, 2].astype(np.uint16) * 31 // 255) << 11
        g = (frame[:, :, 1].astype(np.uint16) * 63 // 255) << 5
        b = frame[:, :, 0].astype(np.uint16) * 31 // 255
        value = r | g | b
        return np.stack((value & 0xFF, value >> 8), axis=-1).astype(np.uint8).tobytes()
    raise ValueError(f"unsupported pixel format: {pixel_format}")
