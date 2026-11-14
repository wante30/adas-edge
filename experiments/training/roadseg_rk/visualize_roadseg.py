"""RoadSeg-RK visualization: low-resolution mask compositing, ego-lane heuristic.

The overlay is built at network resolution and alpha-blended once at native
resolution, so the 1080p Python post-processing bottleneck of the old
Candidate-plus demo is avoided.  Raw per-class masks are always available.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .eval_roadseg import load_model
from .models_roadseg import count_macs
from .train_roadseg import merged_prob
from .postprocess_np import ALPHA, COLORS, build_lowres_overlay, compose, ego_lane, preprocess


@torch.no_grad()
def infer(model, rgb_u8, device, layout):
    x = torch.from_numpy(rgb_u8.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device) / 255
    logits = model(x)
    if layout == "five":
        probs = logits[0].sigmoid().float().cpu().numpy()
        return dict(direct=probs[0], alt=probs[1], solid=probs[2], dashed=probs[3], crosswalk=probs[4])
    probs = logits[0].sigmoid().float().cpu().numpy()
    return dict(direct=probs[0], alt=np.zeros_like(probs[0]), solid=probs[1], dashed=probs[2], crosswalk=probs[3])


def save_raw_masks(probs, geometry, native_shape, out_dir, stem, threshold=0.5):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ih, iw = native_shape
    for name, p in probs.items():
        m = (p >= threshold).astype(np.uint8) * 255
        m = cv2.resize(m, (geometry["nw"], geometry["nh"]), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((ih, iw), np.uint8)
        canvas[geometry["top"]:geometry["top"] + geometry["nh"], geometry["left"]:geometry["left"] + geometry["nw"]] = m
        cv2.imwrite(str(out / f"{stem}_{name}.png"), canvas)


def open_writer(path, width, height, fps, ffmpeg=True):
    if ffmpeg:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-", "-an",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(path)]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def writer_frame(writer, frame, ffmpeg):
    if ffmpeg:
        writer.stdin.write(frame.tobytes())
    else:
        writer.write(frame)


def writer_close(writer, ffmpeg):
    if ffmpeg:
        writer.stdin.close(); writer.wait()
    else:
        writer.release()


def add_label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (max(140, 13 * len(text)), 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def run_video(args, models):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("cannot open video")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = open_writer(out, width * len(models), height, fps, ffmpeg=not args.no_ffmpeg)
    stats = []
    frames = 0
    started = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frames >= args.max_frames:
            break
        t0 = time.perf_counter()
        parts, metas = [], []
        for idx, (model, layout, size) in enumerate(models):
            rgb, geom = preprocess(frame, size)
            t1 = time.perf_counter()
            probs = infer(model, rgb, args.device, layout)
            t2 = time.perf_counter()
            vis, meta = compose(frame, probs, args.threshold, args.ego, args.raw)
            t3 = time.perf_counter()
            parts.append(add_label(vis, args.label[idx] if idx < len(args.label) else f"model{idx}"))
            metas.append(meta)
            stats.append(dict(preprocess_ms=(t1 - t0) * 1000, infer_ms=(t2 - t1) * 1000,
                              postprocess_ms=(t3 - t2) * 1000, **meta))
        writer_frame(writer, np.concatenate(parts, 1), not args.no_ffmpeg)
        frames += 1
        if frames % 30 == 0:
            print(f"frame {frames}", flush=True)
    cap.release()
    writer_close(writer, not args.no_ffmpeg)
    summary = dict(frames=frames, width=width, height=height, fps=fps,
                   wall_seconds=time.perf_counter() - started,
                   pipeline_fps=frames / max(1e-9, time.perf_counter() - started),
                   preprocess_ms=float(np.mean([s["preprocess_ms"] for s in stats])),
                   infer_ms=float(np.mean([s["infer_ms"] for s in stats])),
                   postprocess_ms=float(np.mean([s["postprocess_ms"] for s in stats])),
                   ego_available_rate=float(np.mean([s["ego_available"] for s in stats])),
                   records=stats[:200])
    Path(str(out) + ".json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=2), flush=True)


def run_image(args, models):
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit("cannot open image")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = []
    for idx, (model, layout, size) in enumerate(models):
        rgb, geom = preprocess(frame, size)
        probs = infer(model, rgb, args.device, layout)
        vis, meta = compose(frame, probs, args.threshold, args.ego, args.raw)
        parts.append(add_label(vis, args.label[idx] if idx < len(args.label) else f"model{idx}"))
        if args.raw_masks:
            save_raw_masks(probs, geom, frame.shape[:2], Path(str(out) + "_masks"), Path(args.image).stem, args.threshold)
        print(meta, flush=True)
    cv2.imwrite(str(out), np.concatenate(parts, 1) if len(parts) > 1 else parts[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", required=True)
    p.add_argument("--label", action="append", default=[])
    p.add_argument("--image")
    p.add_argument("--video")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--ego", action="store_true", default=True)
    p.add_argument("--no-ego", dest="ego", action="store_false")
    p.add_argument("--raw", action="store_true", help="disable morphology filtering")
    p.add_argument("--raw-masks", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--no-ffmpeg", action="store_true")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    args = p.parse_args()
    models = []
    for checkpoint in args.checkpoint:
        model, ck, layout, size, names = load_model(checkpoint, args.device)
        if args.width and args.height:
            size = (args.width, args.height)
        models.append((model, layout, size))
        print(f"loaded {checkpoint} model={ck['config']['model']} layout={layout} size={size}", flush=True)
    if args.video:
        run_video(args, models)
    elif args.image:
        run_image(args, models)
    else:
        raise SystemExit("--image or --video required")


if __name__ == "__main__":
    main()
