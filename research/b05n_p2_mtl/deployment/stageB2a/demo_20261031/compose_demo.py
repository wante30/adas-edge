import argparse
import gzip
import json
import os
import subprocess

import cv2
import numpy as np

VIDEO = r"D:\BaiduNetdiskDownload\紫光同创杯赛题三参考素材\video_20260929_112507.mp4"
DIR = r"D:\Datasets\adas_edge_training\b05n_mtl_demo_20261031"
OUT = r"D:\BaiduNetdiskDownload\紫光同创杯赛题三参考素材\video_20260929_112507_B2a_MTL_6Hz_demo.mp4"
W, H = 2560, 720
PW = 1280
NAMES = ["person", "rider", "car", "truck", "bus", "train", "motor", "bike", "traffic light", "traffic sign"]
LANE_COLORS = [(255, 0, 255), (255, 140, 0), (0, 255, 255), (255, 255, 0),
               (0, 200, 0), (150, 100, 255), (255, 255, 255), (255, 200, 0)]
DET_COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 200, 255), (0, 180, 255),
              (255, 255, 0), (255, 120, 0), (255, 120, 0), (0, 255, 255), (255, 0, 255)]


def load_dets(path):
    out = []
    for line in open(path):
        out.append(json.loads(line)["detections"])
    return out


def load_masks(path, frames):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    return np.frombuffer(raw, dtype=np.uint8).reshape(frames, 3, 144, 256)


def text(img, s, x, y, color=(255, 255, 255), scale=0.55, thick=1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render(base, det, masks, sample, total_det, cache):
    left = cv2.resize(base, (PW, H), interpolation=cv2.INTER_AREA)
    right = left.copy()
    if cache.get("sample") != sample:
        lane_up = cv2.resize(masks[0], (PW, H), interpolation=cv2.INTER_NEAREST)
        style_up = cv2.resize(masks[1], (PW, H), interpolation=cv2.INTER_NEAREST)
        drv_up = cv2.resize(masks[2], (PW, H), interpolation=cv2.INTER_NEAREST)
        cache["sample"] = sample
        cache["lane"], cache["style"], cache["drv"] = lane_up, style_up, drv_up
    lane_up, style_up, drv_up = cache["lane"], cache["style"], cache["drv"]

    # drivable (20-35% alpha)
    m = drv_up == 0
    right[m] = (right[m] * 0.78 + np.array([0, 220, 80]) * 0.22).astype(np.uint8)
    m = drv_up == 1
    right[m] = (right[m] * 0.80 + np.array([0, 140, 255]) * 0.20).astype(np.uint8)

    # lane category: soft fill (crosswalk stronger)
    fg = lane_up != 8
    for c, col in enumerate(LANE_COLORS):
        m = (lane_up == c) & fg
        if m.any():
            a = 0.42 if c == 0 else 0.12
            right[m] = (right[m] * (1 - a) + np.array(col) * a).astype(np.uint8)

    # lane style: thin inner edge (solid red / dashed cyan)
    for c, col in ((0, (255, 40, 40)), (1, (0, 230, 255))):
        m = ((style_up == c) & fg).astype(np.uint8)
        if m.any():
            er = cv2.erode(m, np.ones((3, 3), np.uint8))
            edge = (m > 0) & (er == 0)
            right[edge] = col

    # detection boxes
    for cls, score, x1, y1, x2, y2 in det:
        cls = int(cls)
        x1 = int(x1 * 2.5); x2 = int(x2 * 2.5)
        y1 = int(y1 * 2.5); y2 = int(y2 * 2.5)
        col = DET_COLORS[cls] if 0 <= cls < 10 else (255, 255, 255)
        thick = 3 if cls == 8 else 2
        cv2.rectangle(right, (x1, y1), (x2, y2), col, thick)
        if score >= 0.50:
            label = f"{NAMES[cls]} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            yy = max(0, y1 - th - 3)
            cv2.rectangle(right, (x1, yy), (x1 + tw + 4, yy + th + 3), col, -1)
            text(right, label, x1 + 2, yy + th, (0, 0, 0), 0.42, 1)

    # header
    ov = right.copy()
    cv2.rectangle(ov, (0, 0), (PW, 32), (0, 0, 0), -1)
    right = cv2.addWeighted(ov, 0.45, right, 0.55, 0)
    text(right, f"B-2a MTL - Seg-256 INT8 | det: {len(det)} | AI Updates: ~6 Hz (simulated cadence)",
         8, 22, (255, 255, 255), 0.52, 1)

    # left label
    ov = left.copy(); cv2.rectangle(ov, (0, 0), (120, 30), (0, 0, 0), -1)
    left = cv2.addWeighted(ov, 0.45, left, 0.55, 0)
    text(left, "Original", 8, 21, (255, 255, 255), 0.55, 1)

    # legend
    items = [("direct", (0, 220, 80)), ("alternative", (0, 140, 255)),
             ("crosswalk", (255, 0, 255)), ("solid", (255, 40, 40)), ("dashed", (0, 230, 255))]
    x = PW - 620
    ov = right.copy(); cv2.rectangle(ov, (x - 6, H - 26), (PW, H), (0, 0, 0), -1)
    right = cv2.addWeighted(ov, 0.45, right, 0.55, 0)
    for name, col in items:
        cv2.rectangle(right, (x, H - 19), (x + 12, H - 7), col, -1)
        text(right, name, x + 16, H - 8, (255, 255, 255), 0.42, 1)
        x += 118

    out = np.hstack([left, right])
    cv2.line(out, (PW, 0), (PW, H), (60, 60, 60), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--png", default="")
    args = ap.parse_args()
    dets = load_dets(os.path.join(DIR, "detections.jsonl"))
    masks = load_masks(os.path.join(DIR, "masks.bin.gz"), len(dets))
    total_det = sum(len(d) for d in dets)
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("video", count, fps, "samples", len(dets), "dets", total_det, flush=True)
    if args.png:
        for idx in [int(x) for x in args.png.split(",") if x]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            sample = min(idx // 5, len(dets) - 1)
            out = render(frame, dets[sample], masks[sample], sample, total_det, {})
            cv2.imwrite(os.path.join(DIR, f"preview_{idx:04d}.png"), out)
        cap.release()
        return

    silent = os.path.join(DIR, "demo_silent.mp4")
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
           "-crf", "20", "-pix_fmt", "yuv420p", silent]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cache = {}
    n = 0
    for idx in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        sample = min(idx // 5, len(dets) - 1)
        out = render(frame, dets[sample], masks[sample], sample, total_det, cache)
        proc.stdin.write(out.tobytes())
        n += 1
        if n % 100 == 0:
            print("frame", n, flush=True)
    cap.release()
    proc.stdin.close()
    proc.wait()
    print("silent done", n, proc.returncode, flush=True)

    cmd2 = ["ffmpeg", "-y", "-i", silent, "-i", VIDEO, "-c:v", "copy", "-c:a", "aac",
            "-b:a", "128k", "-map", "0:v:0", "-map", "1:a:0?", "-shortest", OUT]
    subprocess.run(cmd2, check=True)
    print("output", OUT, os.path.getsize(OUT), flush=True)


if __name__ == "__main__":
    main()
