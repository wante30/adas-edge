"""Minimal PyTorch definition of RoadCandidatePlus (training architecture).

Kept separate from experiments/.../models.py so deployment evaluation does not
need matplotlib/vendor imports.  The module structure and forward pass are
copied verbatim from the training model.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def conv(cin, cout, stride=1, groups=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
    )


def ds(cin, cout, stride=1):
    return nn.Sequential(
        conv(cin, cin, stride, cin),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(),
    )


class RoadCandidatePlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = conv(3, 32, 2)
        self.enc2 = ds(32, 56, 2)
        self.enc3 = nn.Sequential(ds(56, 96, 2), ds(96, 96))
        self.project = nn.Conv2d(96, 56, 1)
        self.dec2 = ds(56, 56)
        self.project2 = nn.Conv2d(56, 32, 1)
        self.dec1 = ds(32, 32)
        self.head = nn.Conv2d(32, 4, 1)

    def forward(self, x):
        a = self.stem(x)
        b = self.enc2(a)
        c = self.enc3(b)
        d = self.dec2(F.interpolate(self.project(c), size=b.shape[-2:], mode="nearest") + b)
        e = self.dec1(F.interpolate(self.project2(d), size=a.shape[-2:], mode="nearest") + a)
        return F.interpolate(self.head(e), size=x.shape[-2:], mode="nearest")


def load_model(checkpoint: Path):
    model = RoadCandidatePlus()
    payload = torch.load(str(checkpoint), map_location="cpu")
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
