"""Evaluate a multi-task checkpoint with the stage-A metric definitions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtl.data import BDDMTLDataset, collate  # noqa: E402
from mtl.losses import batch_stats  # noqa: E402
from mtl.model import build_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/bdd100k")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ds = BDDMTLDataset(args.root, "val")
    ds = Subset(ds, list(range(min(args.num, len(ds)))))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate, pin_memory=True)

    model = build_model("weights/b05n_p2_multiclass.pt", device="cuda")
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    agg, n = {}, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.amp.autocast("cuda"):
                seg, _, _ = model(batch["image"], run_det=False)
                stats = batch_stats(seg, batch)
            bs = batch["image"].shape[0]
            n += bs
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + float(v) * bs

    res = {k: v / max(n, 1) for k, v in agg.items()}
    res["images"] = n
    res["checkpoint"] = args.ckpt
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
