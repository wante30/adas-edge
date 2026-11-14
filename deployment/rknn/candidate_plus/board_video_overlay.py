"""Process a real road video on the RK3566: NPU segmentation + color overlay.

Decode with OpenCV, run RKNN Lite on every frame, restore the four probability
maps to the native resolution, overlay green/red/cyan/magenta, and encode with
ffmpeg/libx264.  Per-stage timings are recorded without smoothing the masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from common import (
    CLASS_NAMES,
    draw_legend,
    letterbox_rgb,
    masks_from_logits,
    overlay_masks,
    to_nhwc_uint8,
)


def read_text(path: str):
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


def rss_bytes():
    try:
        with open("/proc/self/status", "r") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "min": round(float(arr.min()), 3),
        "max": round(float(arr.max()), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="ultrafast")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError("cannot open source video: %s" % args.source)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("bad video metadata: fps=%r size=%rx%r" % (fps, width, height))

    from rknnlite.api import RKNNLite

    rknn = RKNNLite(verbose=False)
    if rknn.load_rknn(str(args.model)) != 0:
        raise RuntimeError("load_rknn failed")
    if rknn.init_runtime() != 0:
        raise RuntimeError("init_runtime failed")

    ok, warm_frame = cap.read()
    if not ok:
        raise RuntimeError("cannot read the first frame")
    warm_rgb, _ = letterbox_rgb(warm_frame)
    warm_input = to_nhwc_uint8(warm_rgb)
    for _ in range(args.warmup):
        rknn.inference(inputs=[warm_input], data_format="nhwc")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        "%dx%d" % (width, height),
        "-r",
        "%.6f" % fps,
        "-i",
        "pipe:0",
    ]
    if not args.no_audio:
        command += ["-i", str(args.source), "-map", "0:v:0", "-map", "1:a?"]
    command += [
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-pix_fmt",
        "yuv420p",
        "-threads",
        str(args.threads),
    ]
    if not args.no_audio:
        # Do not use -shortest here: it can drop the last few video frames when
        # the audio stream is slightly longer than the video stream.
        command += ["-c:a", "copy"]
    command += [str(args.output)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)

    freq_before = read_text("/sys/class/devfreq/fde40000.npu/cur_freq")
    load_before = read_text("/sys/class/devfreq/fde40000.npu/load")
    rss_before = rss_bytes()
    stages = {"decode": [], "preprocess": [], "infer": [], "postprocess": [], "encode": [], "total": []}
    rows = []
    frame_index = args.start_frame
    wall_start = time.perf_counter()
    try:
        while True:
            if args.max_frames and len(rows) >= args.max_frames:
                break
            t_frame = time.perf_counter()
            t0 = time.perf_counter()
            ok, frame = cap.read()
            decode_ms = (time.perf_counter() - t0) * 1000.0
            if not ok:
                break
            t0 = time.perf_counter()
            rgb, geometry = letterbox_rgb(frame)
            input_nhwc = to_nhwc_uint8(rgb)
            preprocess_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            outputs = rknn.inference(inputs=[input_nhwc], data_format="nhwc")
            infer_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            masks, _ = masks_from_logits(outputs[0], geometry, 0.5)
            overlay = draw_legend(overlay_masks(frame, masks, 0.45))
            postprocess_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            encoder.stdin.write(overlay.tobytes())
            encoder.stdin.flush()
            encode_ms = (time.perf_counter() - t0) * 1000.0
            total_ms = (time.perf_counter() - t_frame) * 1000.0

            stages["decode"].append(decode_ms)
            stages["preprocess"].append(preprocess_ms)
            stages["infer"].append(infer_ms)
            stages["postprocess"].append(postprocess_ms)
            stages["encode"].append(encode_ms)
            stages["total"].append(total_ms)
            rows.append(
                {
                    "frame": frame_index,
                    "decode_ms": round(decode_ms, 3),
                    "preprocess_ms": round(preprocess_ms, 3),
                    "infer_ms": round(infer_ms, 3),
                    "postprocess_ms": round(postprocess_ms, 3),
                    "encode_ms": round(encode_ms, 3),
                    "total_ms": round(total_ms, 3),
                    "area_pixels": [int(mask.sum()) for mask in masks],
                }
            )
            frame_index += 1
    finally:
        cap.release()
        if encoder.stdin:
            encoder.stdin.close()
        encoder.wait()

    wall_seconds = time.perf_counter() - wall_start
    freq_after = read_text("/sys/class/devfreq/fde40000.npu/cur_freq")
    load_after = read_text("/sys/class/devfreq/fde40000.npu/load")
    rss_after = rss_bytes()
    rknn.release()

    summary = {
        "model": str(args.model),
        "source": str(args.source),
        "output": str(args.output),
        "resolution": [width, height],
        "fps": fps,
        "source_total_frames": total,
        "start_frame": args.start_frame,
        "processed_frames": len(rows),
        "encoder": {
            "codec": "libx264",
            "preset": args.preset,
            "crf": args.crf,
            "threads": args.threads,
            "command": command,
            "returncode": encoder.returncode,
        },
        "wall_seconds": round(wall_seconds, 3),
        "pipeline_fps_wall": round(len(rows) / wall_seconds, 3) if wall_seconds else None,
        "pipeline_fps_sum_of_mean_stages": round(
            1000.0 / float(np.mean(stages["total"])), 3
        ),
        "stages_ms": {key: summarize(values) for key, values in stages.items()},
        "npu": {
            "cur_freq_before": freq_before,
            "cur_freq_after": freq_after,
            "load_before": load_before,
            "load_after": load_after,
        },
        "memory": {"rss_before": rss_before, "rss_after": rss_after},
        "classes": list(CLASS_NAMES),
        "timings": rows,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["frame"])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
