#!/usr/bin/env python3
"""Decode a road video off the EGL thread into packed RGB565 replay frames."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


WIDTH = 1280
HEIGHT = 720
FRAME_BYTES = WIDTH * HEIGHT * 2


def frame_timestamps(video: Path) -> list[int]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "csv=p=0",
        str(video),
    ]
    output = subprocess.check_output(command, text=True)
    timestamps: list[int] = []
    for line in output.splitlines():
        value = line.strip().split(",", 1)[0]
        if value:
            timestamps.append(round(float(value) * 1_000_000_000))
    if not timestamps:
        raise RuntimeError(f"ffprobe returned no video timestamps for {video}")
    return timestamps


def read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def prepare(video: Path, output_dir: Path, manifest: Path, max_frames: int | None) -> int:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty replay directory: {output_dir}")
    if manifest.exists():
        raise FileExistsError(f"Refusing to overwrite replay manifest: {manifest}")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamps = frame_timestamps(video)
    if max_frames is not None:
        timestamps = timestamps[:max_frames]

    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        f"scale={WIDTH}:{HEIGHT}:flags=bicubic",
        "-vsync",
        "0",
        "-pix_fmt",
        "rgb565le",
        "-f",
        "rawvideo",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    written = 0
    try:
        while written < len(timestamps):
            payload = read_exact(process.stdout, FRAME_BYTES)
            if not payload:
                break
            if len(payload) != FRAME_BYTES:
                raise RuntimeError(f"short RGB565 frame at index {written}: {len(payload)} bytes")
            name = f"frame_{written:08d}.rgb565"
            (output_dir / name).write_bytes(payload)
            written += 1
    finally:
        process.stdout.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with {return_code}: {stderr[-1000:]}")
    if written != len(timestamps):
        raise RuntimeError(f"decoded {written} frames but ffprobe reported {len(timestamps)}")

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8", newline="\n") as output:
        output.write(f"# source={video}\n")
        output.write(f"# format={WIDTH}x{HEIGHT} RGB565_LE stride={WIDTH * 2}\n")
        for frame_id, timestamp_ns in enumerate(timestamps):
            output.write(f"{frame_id} {timestamp_ns} {output_dir / f'frame_{frame_id:08d}.rgb565'}\n")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    count = prepare(args.video, args.output_dir, args.manifest, args.max_frames)
    print(f"prepared {count} RGB565 frames at {args.output_dir}")


if __name__ == "__main__":
    main()
