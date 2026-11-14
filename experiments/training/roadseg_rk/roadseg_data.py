"""RoadSeg-RK data pipeline: 5-channel labels, 416x256 cache and augmentation.

Channels: 0 direct drivable, 1 alternative drivable, 2 solid lane,
3 dashed lane, 4 crosswalk.  Direct/alternative are rasterized from the
original BDD poly2d annotations; the old merged drivable channel is ignored.
RLMD has no drivable ground truth, so channels 0/1 are invalid there and
never become implicit background supervision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from experiments.training.fpga_road_seg.dataset_adapter import (
    frame_labels, letterbox, polyline_list, raster, read_json, write_json,
)

CLASSES5 = ("direct_drivable", "alternative_drivable", "solid_lane", "dashed_lane", "crosswalk")
SHORT = ("direct", "alt", "solid", "dashed", "crosswalk")


def _load_existing_masks(path):
    with np.load(path, allow_pickle=False) as m:
        return m["target"].copy(), m["valid"].copy()


def bdd_direct_alt(shape, label_json):
    """Return direct, alternative and task-available flag from raw BDD JSON."""
    frame = read_json(label_json)
    if isinstance(frame, dict) and "frames" in frame:
        frames = frame["frames"]
        if len(frames) != 1:
            raise ValueError("expected exactly one BDD frame")
        frame = frames[0]
    labels = frame_labels(frame)
    direct = np.zeros(shape, np.uint8)
    alt = np.zeros(shape, np.uint8)
    available = False
    for label in labels:
        category = label.get("category", "")
        if category not in ("area/drivable", "area/alternative"):
            continue
        available = True
        geometries = polyline_list(label.get("poly2d") or [])
        if not geometries:
            raise ValueError("drivable/alternative polygon without geometry")
        for poly in geometries:
            mask = raster(poly, shape, force_closed=True)
            if category == "area/drivable":
                direct |= mask
            else:
                alt |= mask
    alt &= ~direct
    return direct, alt, available


def _letterbox_np(image, y, v, size):
    w, h = size
    ih, iw = image.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
    left, top = (w - nw) // 2, (h - nh) // 2
    canvas = np.full((h, w, 3), 114, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    yy = np.zeros((len(y), h, w), np.uint8)
    vv = np.zeros((len(v), h, w), np.uint8)
    for c in range(len(y)):
        yy[c, top:top + nh, left:left + nw] = cv2.resize(y[c], (nw, nh), interpolation=cv2.INTER_NEAREST)
        vv[c, top:top + nh, left:left + nw] = cv2.resize(v[c], (nw, nh), interpolation=cv2.INTER_NEAREST)
    return canvas, yy, vv, dict(top=top, left=left, resized_shape=[nh, nw], native_shape=[ih, iw])


def make_sample(row, size):
    """Build one cached sample: letterboxed RGB JPEG bytes + packed 5ch labels."""
    w, h = size
    # BDD/RLMD JPEGs are 1280x720 / 1920x1080; decoding at half size is
    # sufficient for a 416x256 cache and much faster than full-resolution PIL.
    bgr = cv2.imread(row["image"], cv2.IMREAD_REDUCED_COLOR_2)
    if bgr is None:
        bgr = cv2.imread(row["image"], cv2.IMREAD_COLOR)
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    target4, valid4 = _load_existing_masks(row["masks"])
    if target4.shape[1:] != image.shape[:2]:
        target4 = np.stack([cv2.resize(c, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) for c in target4])
        valid4 = np.stack([cv2.resize(c, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) for c in valid4])
    if row["source"] == "bdd100k":
        # Rasterize at native annotation resolution, then resize with the image.
        native_h, native_w = int(round(image.shape[0] * 2)), int(round(image.shape[1] * 2))
        direct, alt, available = bdd_direct_alt((native_h, native_w), row["label_json"])
        direct = cv2.resize(direct, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        alt = cv2.resize(alt, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        direct = alt = None
        available = False
    y = np.zeros((5, *image.shape[:2]), np.uint8)
    v = np.zeros_like(y)
    if row["source"] == "bdd100k":
        if available:
            y[0], y[1] = direct, alt
            v[0, :] = 1
            v[1, :] = 1
        # Lane style comes from the audited renderer.  BDD crosswalk curves are
        # positive-only weak supervision and are explicitly disabled, matching
        # the original four-channel training protocol.
        y[2], y[3] = target4[1], target4[2]
        v[2], v[3] = valid4[1], valid4[2]
        y[4] = 0
        v[4] = 0
    elif row["source"] == "rlmd":
        # No drivable GT in RLMD: channels 0/1 stay invalid.
        y[2], y[3], y[4] = target4[1], target4[2], target4[3]
        v[2], v[3], v[4] = valid4[1], valid4[2], valid4[3]
    else:
        raise ValueError("unsupported source: %s" % row["source"])

    rgb, ys, vs, geometry = _letterbox_np(image, y, v, size)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    bits = np.concatenate([
        np.packbits(ys, axis=-1, bitorder="little"),
        np.packbits(vs, axis=-1, bitorder="little"),
    ]).astype(np.uint8)
    return dict(        id=row["id"], source=row["source"], split=row["split"], group_id=row["group_id"],
        jpg=encoded.tobytes(), bits=bits.tobytes(), shape=[h, w],
        top=geometry["top"], left=geometry["left"],
        resized_shape=geometry["resized_shape"], native_shape=geometry["native_shape"],
        positive=[int((ys[c] * vs[c]).sum()) for c in range(5)],
        valid=[int(vs[c].sum()) for c in range(5)],
    )


def _worker(args):
    index, row, out_dir, size = args
    out_dir = Path(out_dir)
    shard = out_dir / f"{index // 1000:04d}"
    shard.mkdir(parents=True, exist_ok=True)
    jpg_path = shard / f"{index:06d}.jpg"
    npz_path = shard / f"{index:06d}.npz"
    if jpg_path.is_file() and npz_path.is_file():
        with np.load(npz_path, allow_pickle=False) as m:
            meta = json.loads(bytes(m["meta"]).decode("utf-8"))
        return meta
    sample = make_sample(row, size)
    jpg_path.write_bytes(sample.pop("jpg"))
    meta = {k: sample[k] for k in ("id", "source", "split", "group_id", "shape", "top", "left",
                                   "resized_shape", "native_shape", "positive", "valid")}
    meta["jpg"] = str(jpg_path)
    meta["masks"] = str(npz_path)
    temporary = npz_path.with_suffix(".tmp.npz")
    packed = np.frombuffer(sample["bits"], np.uint8).reshape(10, size[1], (size[0] + 7) // 8)
    np.savez_compressed(temporary, bits=packed,
                        meta=np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"), np.uint8))
    os.replace(temporary, npz_path)
    return meta


def build_cache(manifest, output, size=(416, 256), workers=8, limit=0):
    manifest = Path(manifest).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if limit:
        rows = rows[:limit]
    tasks = [(i, row, str(output), size) for i, row in enumerate(rows)]
    index_path = output / "cache_index.jsonl"
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, record in enumerate(pool.map(_worker, tasks, chunksize=8), 1):
            records.append(record)
            if n % 500 == 0:
                print(f"cached {n}/{len(tasks)}", flush=True)
    records.sort(key=lambda r: r["id"])
    index_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    summary = dict(source_manifest=str(manifest), source_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   size=list(size), samples=len(records),
                   by_split={}, by_source={})
    for r in records:
        summary["by_split"][r["split"]] = summary["by_split"].get(r["split"], 0) + 1
        summary["by_source"][r["source"]] = summary["by_source"].get(r["source"], 0) + 1
    write_json(output / "cache_summary.json", summary)
    return summary


def load_cache_index(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.rng.seed(info.dataset.seed * 1000 + worker_id)


class RoadSegDataset(Dataset):
    def __init__(self, index, split, augment=False, size=(416, 256), seed=15):
        self.rows = [r for r in load_cache_index(index) if r["split"] == split]
        if not self.rows:
            raise ValueError("empty split: %s" % split)
        self.augment = augment
        self.size = size
        self.seed = seed
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def _read(self, row):
        bgr = cv2.imread(row["jpg"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(row["jpg"])
        x = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        with np.load(row["masks"], allow_pickle=False) as m:
            packed = m["bits"]
        cache_h, cache_w = row.get("shape", [self.size[1], self.size[0]])
        if packed.ndim == 1:
            packed = packed.reshape(10, cache_h, (cache_w + 7) // 8)
        unpacked = np.unpackbits(packed, axis=-1, count=cache_w, bitorder="little")
        y = unpacked[:5].astype(np.float32)
        v = unpacked[5:10].astype(np.float32)
        if (cache_w, cache_h) != self.size:
            nw, nh = self.size
            x = cv2.resize(x, (nw, nh), interpolation=cv2.INTER_LINEAR)
            y = np.stack([cv2.resize(c, (nw, nh), interpolation=cv2.INTER_NEAREST) for c in y])
            v = np.stack([cv2.resize(c, (nw, nh), interpolation=cv2.INTER_NEAREST) for c in v])
        return x, y, v

    def _photometric(self, x):
        # Brightness / contrast / gamma: shadows and over-exposure.
        gain = self.rng.uniform(0.7, 1.35)
        bias = self.rng.uniform(-0.06, 0.06)
        gamma = self.rng.uniform(0.75, 1.35)
        x = np.clip(x * gain + bias, 0, 1)
        x = np.power(x, gamma)
        # Mild colour-temperature shift.
        if self.rng.random() < 0.5:
            x[..., 0] = np.clip(x[..., 0] * self.rng.uniform(0.9, 1.08), 0, 1)
            x[..., 2] = np.clip(x[..., 2] * self.rng.uniform(0.9, 1.08), 0, 1)
        # Random soft shadow: image-only, labels unchanged.
        if self.rng.random() < 0.45:
            h, w = x.shape[:2]
            cx, cy = self.rng.uniform(0, w), self.rng.uniform(0, h)
            axes = (self.rng.uniform(w * 0.15, w * 0.55), self.rng.uniform(h * 0.10, h * 0.45))
            angle = self.rng.uniform(0, 180)
            shadow = np.zeros((h, w), np.float32)
            cv2.ellipse(shadow, (int(cx), int(cy)), (int(axes[0]), int(axes[1])), angle, 0, 360, 1, -1)
            shadow = cv2.GaussianBlur(shadow, (0, 0), max(1.0, w * 0.01))
            factor = self.rng.uniform(0.55, 0.85)
            x *= (1 - shadow[..., None] * (1 - factor))
        # Haze / low contrast.
        if self.rng.random() < 0.20:
            x = x * self.rng.uniform(0.8, 0.95) + self.rng.uniform(0.03, 0.12)
        # Mild blur and sensor noise.
        if self.rng.random() < 0.20:
            x = cv2.GaussianBlur(x, (3, 3), 0)
        if self.rng.random() < 0.20:
            x = np.clip(x + np.random.default_rng(self.rng.randrange(1 << 30)).normal(0, 0.012, x.shape).astype(np.float32), 0, 1)
        return np.clip(x, 0, 1)

    def _geometric(self, x, y, v):
        if self.rng.random() < 0.5:
            x, y, v = x[:, ::-1], y[:, ::-1], v[:, ::-1]
        # Zoom-out plus small translation, preserving all labelled content.
        if self.rng.random() < 0.35:
            scale = self.rng.uniform(0.85, 1.0)
            h, w = x.shape[:2]
            nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            xr = cv2.resize(x, (nw, nh), interpolation=cv2.INTER_LINEAR)
            yr = np.stack([cv2.resize(c, (nw, nh), interpolation=cv2.INTER_NEAREST) for c in y])
            vr = np.stack([cv2.resize(c, (nw, nh), interpolation=cv2.INTER_NEAREST) for c in v])
            canvas_x = np.full((h, w, 3), 114 / 255, np.float32)
            canvas_y = np.zeros_like(y)
            canvas_v = np.zeros_like(v)
            left = self.rng.randint(0, w - nw)
            top = self.rng.randint(0, h - nh)
            canvas_x[top:top + nh, left:left + nw] = xr
            canvas_y[:, top:top + nh, left:left + nw] = yr
            canvas_v[:, top:top + nh, left:left + nw] = vr
            x, y, v = canvas_x, canvas_y, canvas_v
        return np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(v)

    def __getitem__(self, i):
        x, y, v = self._read(self.rows[i])
        if self.augment:
            x = self._photometric(x)
            x, y, v = self._geometric(x, y, v)
        return (torch.from_numpy(x.transpose(2, 0, 1).copy()),
                torch.from_numpy(y.copy()), torch.from_numpy(v.copy()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--width", type=int, default=416)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    summary = build_cache(args.manifest, args.output, (args.width, args.height), args.workers, args.limit)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
