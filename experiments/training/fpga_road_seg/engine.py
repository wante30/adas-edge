"""Masked multi-label learning and dataset-global confusion metrics."""
import numpy as np
import torch
from torch.nn import functional as F


def masked_loss(logits, target, valid, pos_weight=None, dice_weight=1.0):
    # Official BCEWithLogits broadcasting: [C,1,1] positive weights.
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
    counts = valid.sum((0, 2, 3))
    active = counts > 0
    if not active.any():
        return logits.sum()*0
    weights = None if pos_weight is None else logits.new_tensor(pos_weight)[:, None, None]
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weights, reduction='none')
    bce = (bce*valid).sum((0, 2, 3))/counts.clamp_min(1)
    prob = logits.sigmoid()*valid
    truth = target*valid
    dice = 1-(2*(prob*truth).sum((0,2,3))+1)/(prob.sum((0,2,3))+truth.sum((0,2,3))+1)
    return (bce+dice_weight*dice)[active].mean()


class Metrics:
    def __init__(self, channels):
        self.tp = np.zeros(channels, np.int64)
        self.fp = self.tp.copy(); self.fn = self.tp.copy(); self.tn = self.tp.copy()
        self.valid = self.tp.copy()

    def update(self, probability, target, valid, threshold=.5):
        p, t, v = probability >= threshold, target > .5, valid > .5
        def count(a):
            return a.sum((0,2,3)).cpu().numpy().astype(np.int64)
        self.tp += count(p & t & v); self.fp += count(p & ~t & v)
        self.fn += count(~p & t & v); self.tn += count(~p & ~t & v)
        self.valid += count(v)

    def report(self, names):
        def ratio(a,b):
            return float(a/b) if b else None
        return {n: dict(iou=ratio(a,a+b+c), f1=ratio(2*a,2*a+b+c),
                        recall=ratio(a,a+c), precision=ratio(a,a+b),
                        binary_miou=None if not (a+b+c and d+b+c) else (a/(a+b+c)+d/(d+b+c))/2,
                        tp=int(a),fp=int(b),fn=int(c),tn=int(d),valid_pixels=int(v))
                for n,a,b,c,d,v in zip(names,self.tp,self.fp,self.fn,self.tn,self.valid)}


@torch.no_grad()
def evaluate(model, loader, device, names, threshold=.5):
    model.eval()
    metrics = Metrics(len(names))
    loss_sum = n = 0
    for x,y,v in loader:
        x,y,v = x.to(device),y.to(device),v.to(device)
        logits = model(x)
        loss_sum += float(masked_loss(logits,y,v))*len(x); n += len(x)
        metrics.update(logits.sigmoid(), y, v, threshold)
    return dict(loss=loss_sum/max(n,1), samples=n, classes=metrics.report(names), threshold=threshold)
