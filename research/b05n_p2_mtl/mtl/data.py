"""BDD100K multi-task dataset: image + detection boxes + lane masks + drivable mask."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def decode_lane(mask: np.ndarray):
    """Decode the official lane mask bits into (category 0..8, style 0/1/255)."""
    valid = mask != 255
    cat = np.where(valid, mask & 7, 8).astype(np.int64)
    style = np.where(valid, (mask >> 4) & 1, 255).astype(np.int64)
    return cat, style


class BDDMTLDataset(Dataset):
    def __init__(self, root: str, split: str = "train", imgsz=(288, 512), ids=None):
        self.root = Path(root)
        self.split = split
        self.h, self.w = imgsz
        self.img_dir = self.root / "images" / "flat" / split
        self.lane_dir = self.root / "label_mtl" / "lane" / split
        self.drivable_dir = self.root / "label_mtl" / "drivable" / split
        self.det_dir = self.root / "label_mtl" / "det" / split

        if ids is None:
            index_file = self.root / "label_mtl" / f"index_{split}.json"
            with open(index_file) as f:
                ids = [r["id"] for r in json.load(f)]
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def _image_path(self, sid: str) -> Path:
        p = self.img_dir / f"{sid}.jpg"
        if not p.exists():
            raise FileNotFoundError(f"{sid}.jpg under {self.img_dir}")
        return p

    def __getitem__(self, i):
        sid = self.ids[i]
        img = Image.open(self._image_path(sid)).convert("RGB").resize((self.w, self.h), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(arr.transpose(2, 0, 1)).float()

        lane_p = self.lane_dir / f"{sid}.png"
        if lane_p.exists():
            lm = np.asarray(Image.open(lane_p), dtype=np.uint8)
            if lm.shape != (self.h, self.w):
                lm = np.asarray(Image.fromarray(lm).resize((self.w, self.h), Image.NEAREST))
            lane_cat, lane_style = decode_lane(lm)
            has_lane = True
        else:
            lane_cat = np.full((self.h, self.w), 8, dtype=np.int64)
            lane_style = np.full((self.h, self.w), 255, dtype=np.int64)
            has_lane = False

        drv_p = self.drivable_dir / f"{sid}.png"
        if drv_p.exists():
            dm = np.asarray(Image.open(drv_p), dtype=np.uint8)
            if dm.shape != (self.h, self.w):
                dm = np.asarray(Image.fromarray(dm).resize((self.w, self.h), Image.NEAREST))
            drv = dm.astype(np.int64)
            drv[(drv < 0) | (drv > 2)] = 2
            has_drv = True
        else:
            drv = np.full((self.h, self.w), 2, dtype=np.int64)
            has_drv = False

        det_p = self.det_dir / f"{sid}.txt"
        boxes = np.zeros((0, 5), dtype=np.float32)
        if det_p.exists():
            rows = []
            with open(det_p) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == 5:
                        rows.append([float(x) for x in parts])
            if rows:
                boxes = np.asarray(rows, dtype=np.float32)

        return {
            "image": img_t,
            "sid": sid,
            "lane_cat": torch.from_numpy(lane_cat).long(),
            "lane_style": torch.from_numpy(lane_style).long(),
            "drivable": torch.from_numpy(drv).long(),
            "has_lane": torch.tensor(has_lane),
            "has_drivable": torch.tensor(has_drv),
            "boxes": torch.from_numpy(boxes).float(),
        }


def collate(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in
           ("image", "lane_cat", "lane_style", "drivable", "has_lane", "has_drivable")}
    out["sids"] = [b["sid"] for b in batch]
    out["boxes"] = [b["boxes"] for b in batch]
    return out
