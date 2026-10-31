"""Segmentation losses with foreground/background imbalance handling."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, n_classes: int,
              ignore_index: int = 255, eps: float = 1.0) -> torch.Tensor:
    """Soft dice over the first n_classes channels, ignoring ignore_index pixels."""
    probs = logits.softmax(dim=1)
    valid = target != ignore_index
    tgt = target.clone()
    tgt[~valid] = 0
    onehot = F.one_hot(tgt, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    onehot = onehot[:, :n_classes]
    probs = probs[:, :n_classes]
    mask = valid.unsqueeze(1).float()
    probs = probs * mask
    onehot = onehot * mask
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    union = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def lane_fg_dice(logits, target):
    """Binary lane-vs-background dice; directly optimises thin-line overlap."""
    prob = logits.softmax(1)[:, :8].sum(1)
    tgt = (target != 8).float()
    inter = (prob * tgt).sum((1, 2))
    union = prob.sum((1, 2)) + tgt.sum((1, 2))
    return (1 - (2 * inter + 1) / (union + 1)).mean()


def lane_fg_tversky(logits, target, alpha: float = 0.3, beta: float = 0.7):
    """Tversky on lane foreground; beta>alpha penalises false positives (thick blobs)."""
    prob = logits.softmax(1)[:, :8].sum(1)
    tgt = (target != 8).float()
    tp = (prob * tgt).sum((1, 2))
    fp = (prob * (1 - tgt)).sum((1, 2))
    fn = ((1 - prob) * tgt).sum((1, 2))
    return (1 - (tp + 1) / (tp + alpha * fp + beta * fn + 1)).mean()


def lane_category_loss(logits, target, class_weights):
    ce = F.cross_entropy(logits, target, weight=class_weights)
    dice = dice_loss(logits, target, n_classes=8)  # exclude background
    fg = lane_fg_dice(logits, target)
    tv = lane_fg_tversky(logits, target)
    return ce, dice, fg, tv


def lane_style_loss(logits, target):
    w = torch.tensor([1.0, 3.0, 0.2], device=logits.device, dtype=logits.dtype)
    ce = F.cross_entropy(logits, target, weight=w, ignore_index=255)
    dice = dice_loss(logits, target, n_classes=2, ignore_index=255)
    return ce, dice


def drivable_loss(logits, target):
    w = torch.tensor([1.0, 2.0, 0.5], device=logits.device, dtype=logits.dtype)
    ce = F.cross_entropy(logits, target, weight=w)
    dice = dice_loss(logits, target, n_classes=2)
    return ce, dice


@torch.no_grad()
def batch_stats(seg, batch):
    """Cheap per-batch metrics for logging (full-res argmax)."""
    h, w = batch["lane_cat"].shape[-2:]
    out = {}
    lane_logits = F.interpolate(seg["lane_cat"], size=(h, w), mode="bilinear", align_corners=False)
    pred = lane_logits.argmax(1)
    tgt = batch["lane_cat"]
    fg_p, fg_t = pred != 8, tgt != 8
    inter = (fg_p & fg_t).sum().item()
    union = (fg_p | fg_t).sum().item()
    out["lane_fg_iou"] = inter / max(union, 1)
    cw_p, cw_t = pred == 0, tgt == 0
    out["crosswalk_iou"] = (cw_p & cw_t).sum().item() / max((cw_p | cw_t).sum().item(), 1)

    style_logits = F.interpolate(seg["lane_style"], size=(h, w), mode="bilinear", align_corners=False)
    sp = style_logits.argmax(1)
    st = batch["lane_style"]
    valid = st != 255
    out["style_acc"] = ((sp == st) & valid).sum().item() / max(valid.sum().item(), 1)
    solid = valid & (st == 0)
    dashed = valid & (st == 1)
    out["solid_acc"] = ((sp == 0) & solid).sum().item() / max(solid.sum().item(), 1)
    out["dashed_acc"] = ((sp == 1) & dashed).sum().item() / max(dashed.sum().item(), 1)

    drv_logits = F.interpolate(seg["drivable"], size=(h, w), mode="bilinear", align_corners=False)
    dp = drv_logits.argmax(1)
    dt = batch["drivable"]
    ious = []
    for c in range(3):
        a, b = dp == c, dt == c
        ious.append((a & b).sum().item() / max((a | b).sum().item(), 1))
    out["drivable_miou"] = sum(ious) / 3
    return out
