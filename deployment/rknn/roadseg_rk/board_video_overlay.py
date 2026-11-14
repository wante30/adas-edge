"""RK3566 board video overlay for RoadSeg-RK (RKNN Lite2, numpy-only postprocess).

Example:
  python3 board_video_overlay.py --model roadseg_int8.rknn \
      --video input.mp4 --output overlay.mp4 --width 416 --height 256 --ego
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from postprocess_np import compose, preprocess


def open_writer(path, width, height, fps, ffmpeg=True):
    if ffmpeg:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", "%dx%d" % (width, height), "-r", "%.6f" % fps, "-i", "-", "-an",
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-pix_fmt", "yuv420p", str(path)]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=416)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--ego", action="store_true")
    p.add_argument("--raw", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--no-ffmpeg", action="store_true")
    p.add_argument("--json", type=Path)
    args = p.parse_args()

    from rknnlite.api import RKNNLite
    rknn = RKNNLite(verbose=False)
    if rknn.load_rknn(str(args.model)) != 0:
        raise RuntimeError("load_rknn failed")
    if rknn.init_runtime() != 0:
        raise RuntimeError("init_runtime failed")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError("cannot open video")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = open_writer(args.output, width, height, fps, ffmpeg=not args.no_ffmpeg)
    stats = []
    frames = 0
    started = time.perf_counter()
    while True:
        tr0 = time.perf_counter()
        ok, frame = cap.read()
        tr1 = time.perf_counter()
        if not ok:
            break
        if args.max_frames and frames >= args.max_frames:
            break
        t0 = tr1
        rgb, _ = preprocess(frame, (args.width, args.height))
        t1 = time.perf_counter()
        outputs = rknn.inference(inputs=[rgb[None]], data_format="nhwc")
        t2 = time.perf_counter()
        arr = np.asarray(outputs[0], np.float32)[0]
        if arr.shape[0] == 4:
            probs = dict(direct=arr[0], alt=np.zeros_like(arr[0]), solid=arr[1], dashed=arr[2], crosswalk=arr[3])
        else:
            probs = dict(direct=arr[0], alt=arr[1], solid=arr[2], dashed=arr[3], crosswalk=arr[4])
        vis, meta = compose(frame, probs, args.threshold, args.ego, args.raw)
        t3 = time.perf_counter()
        if args.no_ffmpeg:
            writer.write(vis)
        else:
            writer.stdin.write(vis.tobytes())
        t4 = time.perf_counter()
        stats.append(dict(decode_ms=(t0 - tr0) * 1000, preprocess_ms=(t1 - t0) * 1000,
                          infer_ms=(t2 - t1) * 1000, postprocess_ms=(t3 - t2) * 1000,
                          write_ms=(t4 - t3) * 1000, **meta))
        frames += 1
        if frames % 30 == 0:
            print("frame %d" % frames, flush=True)
    cap.release()
    if args.no_ffmpeg:
        writer.release()
    else:
        writer.stdin.close(); writer.wait()
    wall = time.perf_counter() - started
    rknn.release()
    summary = dict(frames=frames, width=width, height=height, fps=fps, wall_seconds=wall,
                   pipeline_fps=frames / wall if wall else None,
                   preprocess_ms=float(np.mean([s["preprocess_ms"] for s in stats])),
                   infer_ms=float(np.mean([s["infer_ms"] for s in stats])),
                   postprocess_ms=float(np.mean([s["postprocess_ms"] for s in stats])),
                   write_ms=float(np.mean([s["write_ms"] for s in stats])),
                   infer_p50_ms=float(np.percentile([s["infer_ms"] for s in stats], 50)),
                   infer_p95_ms=float(np.percentile([s["infer_ms"] for s in stats], 95)),
                   ego_available_rate=float(np.mean([s["ego_available"] for s in stats])))
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
