import json
import sys
from collections import Counter

import numpy as np

ROOT = "/home/cat/competition_cockpit_port_20261018"
sys.path.insert(0, ROOT)
from tools.profile_rknn_single_stream import letterbox_bgr
from deployment.rknn.decode_raw_heads import decode_raw_heads_thresholded
from rknnlite.api import RKNNLite

RAW = "/home/cat/b05g_113421_3s.rgb"
B05N = "/home/cat/cityscapes_arch_compare_20261027/models/b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn"
B2A = "/home/cat/models/b05n_p2_mtl_b2a/b05n_p2_mtl_b2a_int8.rknn"


def decode565_rgb(a):
    v = a[..., 0].astype(np.uint16) | (a[..., 1].astype(np.uint16) << 8)
    r = ((v >> 11) & 0x1F).astype(np.uint8); r = (r << 3) | (r >> 2)
    g = ((v >> 5) & 0x3F).astype(np.uint8); g = (g << 2) | (g >> 4)
    b = (v & 0x1F).astype(np.uint8); b = (b << 3) | (b >> 2)
    return np.stack([b, g, r], -1)


raw = np.fromfile(RAW, dtype="<u2", count=60 * 1280 * 720)
frames = [decode565_rgb(raw[i * 1280 * 720:(i + 1) * 1280 * 720].view(np.uint8).reshape(720, 1280, 2))
          for i in range(60)]


def run(model, mtl):
    r = RKNNLite(verbose=False)
    r.load_rknn(model)
    r.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    tl_frames = 0
    tl_counts = []
    classes = Counter()
    seg_rows = []
    for frame in frames:
        img, s, px, py = letterbox_bgr(frame, (512, 288))
        out = r.inference(inputs=[img[None, ...]], data_format="nhwc")
        det = decode_raw_heads_thresholded(out[:12], input_width=512, input_height=288,
                                           confidence=0.25, iou=0.45, scale=s, pad_x=px, pad_y=py)
        tls = [d for d in det if int(d[5]) == 8]
        tl_counts.append(len(tls))
        tl_frames += int(bool(tls))
        for d in det:
            classes[int(d[5])] += 1
        if mtl:
            lane = np.asarray(out[12])[0].argmax(0)
            style = np.asarray(out[13])[0].argmax(0)
            drv = np.asarray(out[14])[0].argmax(0)
            fg = lane != 8
            n = max(int(fg.sum()), 1)
            seg_rows.append([float(fg.mean()), float((lane == 0).mean()),
                             float(((style == 0) & fg).sum() / n),
                             float(((style == 1) & fg).sum() / n),
                             float((drv == 0).mean()), float((drv == 1).mean())])
    r.release()
    out = {"tl_frames": tl_frames, "frames": len(frames), "tl_boxes": int(sum(tl_counts)),
           "tl_per_frame_mean": float(np.mean(tl_counts)),
           "classes": {str(k): v for k, v in classes.items()}}
    if seg_rows:
        a = np.asarray(seg_rows)
        out["seg"] = {"lane_fg": float(a[:, 0].mean()), "crosswalk": float(a[:, 1].mean()),
                      "solid_in_lane": float(a[:, 2].mean()), "dashed_in_lane": float(a[:, 3].mean()),
                      "direct": float(a[:, 4].mean()), "alternative": float(a[:, 5].mean()),
                      "lane_fg_min": float(a[:, 0].min()), "lane_fg_max": float(a[:, 0].max()),
                      "solid_min": float(a[:, 2].min()), "solid_max": float(a[:, 2].max()),
                      "dashed_min": float(a[:, 3].min()), "dashed_max": float(a[:, 3].max())}
    return out


res = {"b05n": run(B05N, False), "b2a": run(B2A, True)}
print(json.dumps(res, indent=2))
open("/home/cat/b05n_mtl_b2a_bench_20261031/validate_outputs.json", "w").write(json.dumps(res, indent=2))
