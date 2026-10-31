import json
import os
import re
import resource
import sys
import threading
import time
from collections import Counter

import numpy as np

ROOT = "/home/cat/competition_cockpit_port_20261018"
sys.path.insert(0, ROOT)
from tools.profile_rknn_single_stream import letterbox_bgr
from deployment.rknn.decode_raw_heads import decode_raw_heads_thresholded
from rknnlite.api import RKNNLite

RAW = "/home/cat/b05g_113421_3s.rgb"
N_FRAMES = 60
ROUNDS = 5
WARMUP = 10
B05N = "/home/cat/cityscapes_arch_compare_20261027/models/b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn"
B2A = "/home/cat/models/b05n_p2_mtl_b2a/b05n_p2_mtl_b2a_int8.rknn"
CONF = 0.25
IOU = 0.45
BDD_NAMES = ("person", "rider", "car", "truck", "bus", "train", "motorcycle",
             "bicycle", "traffic light", "traffic sign")


def decode565_rgb(a):
    v = a[..., 0].astype(np.uint16) | (a[..., 1].astype(np.uint16) << 8)
    r = ((v >> 11) & 0x1F).astype(np.uint8); r = (r << 3) | (r >> 2)
    g = ((v >> 5) & 0x3F).astype(np.uint8); g = (g << 2) | (g >> 4)
    b = (v & 0x1F).astype(np.uint8); b = (b << 3) | (b >> 2)
    return np.stack([b, g, r], -1)


def cpu_snapshot():
    with open("/proc/stat") as f:
        parts = f.readline().split()
    v = list(map(int, parts[1:]))
    return sum(v), v[3] + v[4]


def npu_load():
    try:
        with open("/sys/kernel/debug/rknpu/load") as f:
            m = re.search(r"(\d+)%", f.read())
            return int(m.group(1)) if m else None
    except Exception:
        return None


def stats(v):
    a = np.asarray(v, dtype=np.float64)
    return {"samples": int(a.size), "mean_ms": float(a.mean()),
            "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95))}


raw = np.fromfile(RAW, dtype="<u2", count=N_FRAMES * 1280 * 720)
frames = [decode565_rgb(raw[i * 1280 * 720:(i + 1) * 1280 * 720].view(np.uint8).reshape(720, 1280, 2))
          for i in range(N_FRAMES)]


def bench(model_path, is_mtl):
    r = RKNNLite(verbose=False)
    if r.load_rknn(model_path) != 0 or r.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
        raise RuntimeError("rknn init failed")

    def infer(frame):
        t0 = time.perf_counter_ns()
        img, scale, px, py = letterbox_bgr(frame, (512, 288))
        t1 = time.perf_counter_ns()
        out = r.inference(inputs=[img[None, ...]], data_format="nhwc")
        t2 = time.perf_counter_ns()
        det = decode_raw_heads_thresholded(out[:12], input_width=512, input_height=288,
                                           confidence=CONF, iou=IOU, scale=scale, pad_x=px, pad_y=py)
        t3 = time.perf_counter_ns()
        seg = None
        if is_mtl:
            seg = [np.asarray(o)[0].argmax(0).astype(np.uint8) for o in out[12:15]]
        t4 = time.perf_counter_ns()
        return (t1 - t0) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6, (t4 - t3) / 1e6, (t4 - t0) / 1e6, det, seg

    for i in range(WARMUP):
        infer(frames[i % N_FRAMES])

    times = {"pre": [], "rknn": [], "det": [], "seg": [], "total": []}
    round_total = []
    all_det, tl_scores, seg_frac = [], [], []
    npu_samples = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            v = npu_load()
            if v is not None:
                npu_samples.append(v)
            time.sleep(0.05)

    thread = threading.Thread(target=sampler, daemon=True)
    c0, i0 = cpu_snapshot()
    thread.start()
    wall0 = time.perf_counter_ns()
    for rnd in range(ROUNDS):
        r0 = time.perf_counter_ns()
        for frame in frames:
            pre, rknn, det, seg, total, dets, masks = infer(frame)
            times["pre"].append(pre); times["rknn"].append(rknn)
            times["det"].append(det); times["seg"].append(seg); times["total"].append(total)
            all_det.extend(dets)
            tl_scores.extend(d[4] for d in dets if int(d[5]) == 8)
            if masks is not None:
                lane, style, drv = masks
                seg_frac.append([float((lane != 8).mean()), float((lane == 0).mean()),
                                 float((style == 0).mean()), float((style == 1).mean()),
                                 float((drv == 0).mean()), float((drv == 1).mean())])
        round_total.append((time.perf_counter_ns() - r0) / 1e9)
    wall1 = time.perf_counter_ns()
    c1, i1 = cpu_snapshot()
    stop.set(); thread.join(timeout=1)
    r.release()

    n = ROUNDS * N_FRAMES
    out = {
        "model": model_path, "size_bytes": os.path.getsize(model_path),
        "frames_distinct": N_FRAMES, "rounds": ROUNDS, "inferences": n,
        "warmup": WARMUP, "conf": CONF, "iou": IOU,
        "timing": {k: stats(v) for k, v in times.items()},
        "round_seconds": round_total,
        "round_fps": [N_FRAMES / s for s in round_total],
        "wall_seconds": (wall1 - wall0) / 1e9,
        "continuous_fps": n / ((wall1 - wall0) / 1e9),
        "cpu_percent": 100.0 * (1 - (i1 - i0) / max(1, (c1 - c0))),
        "npu_load_samples": len(npu_samples),
        "npu_load_mean": float(np.mean(npu_samples)) if npu_samples else None,
        "npu_load_max": int(max(npu_samples)) if npu_samples else None,
        "rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "detections_total": len(all_det),
        "det_classes": {BDD_NAMES[k]: v for k, v in Counter(int(d[5]) for d in all_det).items()},
        "tl_count": len(tl_scores),
        "tl_max": float(max(tl_scores)) if tl_scores else None,
        "tl_mean": float(np.mean(tl_scores)) if tl_scores else None,
    }
    if seg_frac:
        a = np.asarray(seg_frac)
        out["seg_fraction_mean"] = {
            "lane_fg": float(a[:, 0].mean()), "crosswalk": float(a[:, 1].mean()),
            "solid": float(a[:, 2].mean()), "dashed": float(a[:, 3].mean()),
            "direct": float(a[:, 4].mean()), "alternative": float(a[:, 5].mean()),
        }
        out["seg_nonzero_frames"] = {"lane": int((a[:, 0] > 0).sum()),
                                     "drivable": int(((a[:, 4] + a[:, 5]) > 0).sum()),
                                     "frames": int(a.shape[0])}
    return out


res = {
    "raw": RAW,
    "raw_format": "RGB565 little-endian 1280x720, 90 frames total; first 60 used",
    "source_note": "b05g_113421_3s.rgb decoded as RGB565; 60 distinct frames repeated 5 rounds = 300 inferences",
    "nice": os.nice(0),
}
res["b05n"] = bench(B05N, False)
res["b2a"] = bench(B2A, True)
with open("/home/cat/b05n_mtl_b2a_bench_20261031/bench_60_result_v2.json", "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))
