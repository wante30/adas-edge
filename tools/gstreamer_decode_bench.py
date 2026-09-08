"""Measure concurrent mppvideodec throughput without inference."""
from __future__ import annotations

import argparse
import statistics
import subprocess
import threading
import time


def run_mpp(video: str, streams: int, frames_per_stream: int) -> None:
    import psutil
    command = ["gst-launch-1.0", "-q", "filesrc", f"location={video}", "!", "parsebin", "!", "mppvideodec", "!", "fakesink"]
    processes = [subprocess.Popen(command) for _ in range(streams)]
    samples = []
    started = time.perf_counter()
    while any(proc.poll() is None for proc in processes):
        samples.append(psutil.cpu_percent(interval=0.5))
    elapsed = time.perf_counter() - started
    for proc in processes:
        proc.wait()
    print({"backend": "mppvideodec", "streams": streams, "elapsed_s": elapsed, "frames": streams * frames_per_stream, "total_fps": streams * frames_per_stream / elapsed if elapsed else 0.0, "cpu_mean_percent": statistics.mean(samples) if samples else 0.0, "cpu_max_percent": max(samples) if samples else 0.0})


def run_opencv(video: str, streams: int) -> None:
    import cv2
    import psutil
    counts = [0] * streams
    done = threading.Event()

    def decode(stream_id: int) -> None:
        cap = cv2.VideoCapture(video)
        try:
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                counts[stream_id] += 1
        finally:
            cap.release()

    threads = [threading.Thread(target=decode, args=(stream_id,)) for stream_id in range(streams)]
    samples = []
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        samples.append(psutil.cpu_percent(interval=0.5))
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - started
    print({"backend": "opencv_ffmpeg_bgr", "streams": streams, "elapsed_s": elapsed, "frames": sum(counts), "total_fps": sum(counts) / elapsed if elapsed else 0.0, "cpu_mean_percent": statistics.mean(samples) if samples else 0.0, "cpu_max_percent": max(samples) if samples else 0.0})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--streams", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--frames", type=int, default=1084)
    parser.add_argument("--backend", choices=("mpp", "opencv", "both"), default="mpp")
    args = parser.parse_args()
    for streams in args.streams:
        if args.backend in ("mpp", "both"):
            run_mpp(args.video, streams, args.frames)
        if args.backend in ("opencv", "both"):
            run_opencv(args.video, streams)


if __name__ == "__main__":
    main()
