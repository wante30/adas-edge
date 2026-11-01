"""Latest-only reader for the board-side PGV1 PCIe frame publisher."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import struct
import time
import zlib

import cv2
import numpy as np


WIDTH = 1280
HEIGHT = 720
FRAME_BYTES = WIDTH * HEIGHT * 2
HEADER = struct.Struct("<4s7IQ")
FILE_BYTES = HEADER.size + FRAME_BYTES


@dataclass(frozen=True)
class PgvFrame:
    image: np.ndarray
    frame_id: int
    generation: int
    timestamp_ns: int
    crc32: int
    age_ms: float
    navigation_image: np.ndarray | None = None


class PgvLatestCapture:
    is_live = True

    def __init__(self, path, *, clock_ns=time.monotonic_ns, quadrant=None, navigation_quadrant=None):
        self.path = Path(path)
        self.clock_ns = clock_ns
        self.quadrant = self._validate_quadrant(quadrant)
        self.navigation_quadrant = self._validate_quadrant(navigation_quadrant)
        self._file_signature = None
        self._generation = None
        self._frame_id = None
        self._timestamp_ns = None
        self.dropped_frames = 0
        self.closed = False

    @staticmethod
    def _validate_quadrant(quadrant):
        if quadrant is not None and (isinstance(quadrant, bool) or quadrant not in (0, 1, 2, 3)):
            raise ValueError("Quadrant must be 0, 1, 2, 3, or None")
        return quadrant

    @staticmethod
    def _crop_quadrant(image, quadrant):
        if quadrant is None:
            return image
        row = 0 if quadrant in (0, 1) else 1
        col = 0 if quadrant in (0, 2) else 1
        return image[row * HEIGHT // 2:(row + 1) * HEIGHT // 2,
                     col * WIDTH // 2:(col + 1) * WIDTH // 2]

    def read_latest(self):
        if self.closed:
            raise RuntimeError("PGV capture is closed")
        try:
            source = self.path.open("rb")
        except FileNotFoundError:
            return None
        with source:
            stat = os.fstat(source.fileno())
            signature = (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if signature == self._file_signature:
                return None
            if stat.st_size != FILE_BYTES:
                raise ValueError(f"Invalid PGV file length: {stat.st_size}")
            raw = source.read(FILE_BYTES + 1)
        if len(raw) != FILE_BYTES:
            raise ValueError(f"Truncated PGV file: {len(raw)}")
        magic, version, width, height, frame_id, generation, crc32, length, timestamp_ns = (
            HEADER.unpack_from(raw)
        )
        if (magic, version, width, height, length) != (
            b"PGVF", 1, WIDTH, HEIGHT, FRAME_BYTES
        ):
            raise ValueError("Invalid PGVF header")
        payload = memoryview(raw)[HEADER.size:]
        if zlib.crc32(payload) & 0xFFFFFFFF != crc32:
            raise ValueError("PGV frame CRC mismatch")
        if generation == self._generation:
            delta = (frame_id - self._frame_id) & 0xFFFFFFFF
            if delta == 0 or delta >= 0x80000000:
                self._file_signature = signature
                return None
            self.dropped_frames += delta - 1
        packed = np.frombuffer(payload, dtype=np.uint8).reshape(HEIGHT, WIDTH, 2)
        image = cv2.cvtColor(self._crop_quadrant(packed, self.quadrant), cv2.COLOR_BGR5652BGR)
        navigation = None
        if self.navigation_quadrant is not None:
            navigation = cv2.cvtColor(
                self._crop_quadrant(packed, self.navigation_quadrant), cv2.COLOR_BGR5652BGR
            )
        self._file_signature = signature
        self._generation = generation
        self._frame_id = frame_id
        self._timestamp_ns = timestamp_ns
        return PgvFrame(
            image=image,
            frame_id=frame_id,
            generation=generation,
            timestamp_ns=timestamp_ns,
            crc32=crc32,
            age_ms=(self.clock_ns() - timestamp_ns) / 1_000_000,
            navigation_image=navigation,
        )

    def latest_age_ms(self):
        if self._timestamp_ns is None:
            return None
        return (self.clock_ns() - self._timestamp_ns) / 1_000_000

    @property
    def latest_frame_id(self):
        return self._frame_id

    def release(self):
        self.closed = True


class PgvLiveCapture:
    """Blocks for each newest PGV frame so the GUI consumes live source IDs."""

    is_live = True

    def __init__(self, path, *, timeout_s=5.0, clock_ns=time.monotonic_ns,
                 quadrant=None, navigation_quadrant=None):
        if timeout_s <= 0:
            raise ValueError("Timeout must be positive")
        self.source = PgvLatestCapture(path, clock_ns=clock_ns, quadrant=quadrant,
                                       navigation_quadrant=navigation_quadrant)
        self.timeout_s = timeout_s
        self.frame_id = None
        self.timestamp_ns = None
        self.age_ms = None
        self.navigation_image = None
        self.error = None
        self.closed = False

    def read(self):
        if self.closed:
            raise RuntimeError("PGV live capture is closed")
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                frame = self.source.read_latest()
            except (OSError, ValueError, zlib.error) as exc:
                self.error = exc
                return False, None
            if frame is not None:
                self.frame_id = frame.frame_id
                self.timestamp_ns = frame.timestamp_ns
                self.age_ms = frame.age_ms
                self.navigation_image = getattr(frame, "navigation_image", None)
                return True, frame.image
            if time.monotonic() >= deadline:
                return False, None
            time.sleep(0.001)

    def set(self, prop, value):
        return True

    def release(self):
        if not self.closed:
            self.source.release()
            self.closed = True

    @property
    def dropped_frames(self):
        return self.source.dropped_frames
