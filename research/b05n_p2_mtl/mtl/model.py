"""B05N P2 multi-task model: frozen YOLOv8n-P2 detector + shared lightweight seg decoder.

Detection backbone/neck/head and the P2..P5 detection structure are kept intact.
The decoder only consumes neck features (P2/P3/P4), so there is no second backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LANE_CATEGORIES = [
    "crosswalk",
    "double other",
    "double white",
    "double yellow",
    "road curb",
    "single other",
    "single white",
    "single yellow",
    "background",
]
LANE_STYLES = ["solid", "dashed", "background"]
DRIVABLE = ["direct", "alternative", "background"]


class ConvBNAct(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SegDecoder(nn.Module):
    """P4->P3->P2 top-down fusion, output stride 1/2 with three task heads.

    Only Conv / BatchNorm / SiLU / nearest Upsample are used so the graph stays
    friendly for a later ONNX -> RKNN conversion.
    """

    def __init__(self, c2: int, c3: int, c4: int, mid: int = 64, head_ch: int = 64,
                 n_lane_cat: int = 9, n_lane_style: int = 3, n_drivable: int = 3):
        super().__init__()
        self.lat4 = ConvBNAct(c4, mid, 1)
        self.lat3 = ConvBNAct(c3, mid, 1)
        self.lat2 = ConvBNAct(c2, mid, 1)
        self.fuse3 = ConvBNAct(mid, mid, 3)
        self.fuse2 = ConvBNAct(mid, mid, 3)
        self.stage_half = ConvBNAct(mid, head_ch, 3)  # stride 1/2
        self.head = ConvBNAct(head_ch, head_ch, 3)
        self.head2 = ConvBNAct(head_ch, head_ch, 3)
        self.refine = ConvBNAct(head_ch, head_ch // 2, 3)
        self.lane_cat = nn.Conv2d(head_ch // 2, n_lane_cat, 1)
        self.lane_style = nn.Conv2d(head_ch // 2, n_lane_style, 1)
        self.drivable = nn.Conv2d(head_ch // 2, n_drivable, 1)

    def forward(self, p2, p3, p4):
        f4 = self.lat4(p4)
        f3 = self.fuse3(self.lat3(p3) + F.interpolate(f4, scale_factor=2, mode="nearest"))
        f2 = self.fuse2(self.lat2(p2) + F.interpolate(f3, scale_factor=2, mode="nearest"))
        x = self.stage_half(F.interpolate(f2, scale_factor=2, mode="nearest"))
        x = self.head2(self.head(x))
        x = self.refine(F.interpolate(x, scale_factor=2, mode="nearest"))
        return {
            "lane_cat": self.lane_cat(x),
            "lane_style": self.lane_style(x),
            "drivable": self.drivable(x),
        }


class B05NMTL(nn.Module):
    """Wraps an Ultralytics DetectionModel and taps its P2/P3/P4/P5 neck outputs."""

    def __init__(self, det_model: nn.Module, mid: int = 64, head_ch: int = 64):
        super().__init__()
        self.det = det_model
        self.det.eval()
        for p in self.det.parameters():
            p.requires_grad_(False)
        self._unfreeze_layers: list[int] = []
        self._train_head = False

        self.tap_idx = list(self.det.save)[-4:]  # P2, P3, P4, P5
        self._feats: dict[int, torch.Tensor] = {}
        self._handles = []
        for idx in self.tap_idx:
            self._handles.append(
                self.det.model[idx].register_forward_hook(self._make_hook(idx))
            )

        # one tiny dry run to read the real tap channel counts
        self.det.eval()
        with torch.no_grad():
            self.det(torch.zeros(1, 3, 128, 128))
        c2 = self._feats[self.tap_idx[0]].shape[1]
        c3 = self._feats[self.tap_idx[1]].shape[1]
        c4 = self._feats[self.tap_idx[2]].shape[1]
        self._feats.clear()
        self.decoder = SegDecoder(c2, c3, c4, mid=mid, head_ch=head_ch)
        self.meta = {"channels": [c2, c3, c4], "tap_idx": self.tap_idx}

    @staticmethod
    def _out_channels(layer: nn.Module) -> int:
        for m in reversed(list(layer.modules())):
            if isinstance(m, nn.Conv2d):
                return m.out_channels
        raise RuntimeError("cannot infer output channels")

    def _make_hook(self, idx: int):
        def hook(_module, _inp, out):
            self._feats[idx] = out
        return hook

    def set_trainable(self, layer_idxs, train_head: bool = True):
        """Unfreeze selected detector layers (P2/P3 path) and optionally the Detect head."""
        for p in self.det.parameters():
            p.requires_grad_(False)
        for idx in layer_idxs:
            for p in self.det.model[idx].parameters():
                p.requires_grad_(True)
        self._unfreeze_layers = list(layer_idxs)
        self._train_head = bool(train_head)
        if train_head:
            for p in self.det.model[-1].parameters():
                p.requires_grad_(True)
        self.train(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.det.eval()  # frozen layers keep eval BN
        if mode:
            for idx in self._unfreeze_layers:
                self.det.model[idx].train()
            if self._train_head:
                self.det.model[-1].train()
        return self

    def forward(self, x: torch.Tensor, run_det: bool = True):
        """Single pass: detection head + segmentation heads.

        Returns (seg_logits dict at stride 1/2, det_output or None, det_features).
        In stage A the detector runs under no_grad to keep memory low.
        """
        self._feats.clear()
        if run_det:
            if any(p.requires_grad for p in self.det.parameters()):
                det_out = self.det(x)
            else:
                with torch.no_grad():
                    det_out = self.det(x)
        else:
            det_out = None
            with torch.no_grad():
                self.det(x)

        p2 = self._feats[self.tap_idx[0]]
        p3 = self._feats[self.tap_idx[1]]
        p4 = self._feats[self.tap_idx[2]]
        if not any(p.requires_grad for p in self.det.parameters()):
            p2, p3, p4 = p2.detach(), p3.detach(), p4.detach()

        seg = self.decoder(p2, p3, p4)
        feats = [self._feats[i] for i in self.tap_idx]
        return seg, det_out, feats

    def seg_parameters(self):
        return self.decoder.parameters()


def build_model(ckpt_path: str, device: str = "cuda", mid: int = 64, head_ch: int = 64) -> B05NMTL:
    from ultralytics import YOLO

    yolo = YOLO(ckpt_path)
    model = B05NMTL(yolo.model, mid=mid, head_ch=head_ch).to(device)
    return model
