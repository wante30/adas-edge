import importlib
import os
import struct
import zlib

import numpy as np
import pytest


WIDTH = 1280
HEIGHT = 720
FRAME_BYTES = WIDTH * HEIGHT * 2
HEADER = struct.Struct("<4s7IQ")


def pgv_capture_type():
    runtime = importlib.import_module("tools.competition_console.pgv_capture")
    capture_type = getattr(runtime, "PgvLatestCapture", None)
    assert capture_type is not None, "PGV latest-frame source is not implemented"
    return capture_type


def write_frame(path, frame_id, generation=3, timestamp_ns=1_000_000_000, corrupt_crc=False):
    pixels = np.zeros((HEIGHT, WIDTH), dtype="<u2")
    pixels[0, :3] = (0xF800, 0x07E0, 0x001F)
    payload = pixels.tobytes()
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    if corrupt_crc:
        crc ^= 1
    header = HEADER.pack(
        b"PGVF", 1, WIDTH, HEIGHT, frame_id, generation, crc, FRAME_BYTES, timestamp_ns
    )
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(header + payload)
    os.replace(temporary, path)


def write_quadrant_frame(path, frame_id=11):
    pixels = np.zeros((HEIGHT, WIDTH), dtype="<u2")
    pixels[: HEIGHT // 2, : WIDTH // 2] = 0xF800
    pixels[: HEIGHT // 2, WIDTH // 2 :] = 0x07E0
    pixels[HEIGHT // 2 :, : WIDTH // 2] = 0xFFFF
    pixels[HEIGHT // 2 :, WIDTH // 2 :] = 0x001F
    payload = pixels.tobytes()
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    header = HEADER.pack(
        b"PGVF", 1, WIDTH, HEIGHT, frame_id, 3, crc, FRAME_BYTES, 1_000_000_000
    )
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(header + payload)
    os.replace(temporary, path)


def test_reads_only_new_pgv1_frames_and_preserves_source_identity(tmp_path):
    path = tmp_path / "latest.pgv"
    write_frame(path, 100, timestamp_ns=1_000_000_000)
    capture = pgv_capture_type()(path, clock_ns=lambda: 1_025_000_000)

    frame = capture.read_latest()

    assert frame.frame_id == 100
    assert frame.generation == 3
    assert frame.timestamp_ns == 1_000_000_000
    assert frame.age_ms == pytest.approx(25.0)
    assert capture.read_latest() is None


def test_rgb565_frame_is_converted_to_bgr_for_the_competition_pipeline(tmp_path):
    path = tmp_path / "latest.pgv"
    write_frame(path, 7)
    capture = pgv_capture_type()(path, clock_ns=lambda: 1_000_000_000)

    image = capture.read_latest().image

    assert image.shape == (HEIGHT, WIDTH, 3)
    assert image.dtype == np.uint8
    assert image[0, 0, 2] > 240 and image[0, 0, 0] < 8
    assert image[0, 1, 1] > 240 and image[0, 1, 2] < 8
    assert image[0, 2, 0] > 240 and image[0, 2, 2] < 8


def test_latest_only_reader_counts_source_frames_skipped_by_the_consumer(tmp_path):
    path = tmp_path / "latest.pgv"
    capture = pgv_capture_type()(path, clock_ns=lambda: 1_000_000_000)
    write_frame(path, 100)
    assert capture.read_latest().frame_id == 100
    write_frame(path, 103)

    assert capture.read_latest().frame_id == 103
    assert capture.dropped_frames == 2


def test_accepts_new_generation_but_ignores_regressed_ids_within_generation(tmp_path):
    path = tmp_path / "latest.pgv"
    capture = pgv_capture_type()(path, clock_ns=lambda: 1_000_000_000)
    write_frame(path, 10, generation=3)
    assert capture.read_latest().frame_id == 10
    write_frame(path, 9, generation=3)
    assert capture.read_latest() is None
    write_frame(path, 1, generation=4)

    frame = capture.read_latest()

    assert (frame.generation, frame.frame_id) == (4, 1)


def test_rejects_pgv1_frames_with_invalid_crc(tmp_path):
    path = tmp_path / "latest.pgv"
    write_frame(path, 1, corrupt_crc=True)
    capture = pgv_capture_type()(path, clock_ns=lambda: 1_000_000_000)

    with pytest.raises(ValueError, match="CRC"):
        capture.read_latest()


def test_routes_bottom_right_to_detection_and_top_right_to_navigation(tmp_path):
    path = tmp_path / "latest.pgv"
    write_quadrant_frame(path)
    capture = pgv_capture_type()(
        path,
        quadrant=3,
        navigation_quadrant=1,
        clock_ns=lambda: 1_000_000_000,
    )

    frame = capture.read_latest()

    assert frame.frame_id == 11
    assert frame.image.shape == (HEIGHT // 2, WIDTH // 2, 3)
    assert frame.navigation_image.shape == (HEIGHT // 2, WIDTH // 2, 3)
    assert frame.image[:, :, 0].mean() > 240
    assert frame.navigation_image[:, :, 1].mean() > 240
