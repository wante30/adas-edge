"""RK3566-oriented segmentation models for RoadSeg-RK.

A: original Candidate-plus at higher input resolution (4 channels, merged drivable).
B: RoadSegNet-5, a depthwise encoder/decoder with direct/alternative separation.
C: MobileNetV3-Large + LR-ASPP style head with a stride-8 refinement branch.

Only Conv2d (3x3/1x1/depthwise), BatchNorm, ReLU/Hardswish, Add, Concat,
global average pooling and nearest/bilinear resize are used.  No custom
operators, attention blocks or transposed convolutions are required.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from experiments.training.fpga_road_seg.models import RoadCandidatePlus


def conv_bn(cin, cout, k=3, s=1, p=None, groups=1, act=True):
    p = k // 2 if p is None else p
    layers = [nn.Conv2d(cin, cout, k, s, p, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResDW(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 residual block (expansion optional)."""

    def __init__(self, channels, expand=1):
        super().__init__()
        mid = channels * expand
        self.pre = conv_bn(channels, mid, 1) if expand != 1 else nn.Identity()
        self.dw = conv_bn(mid, mid, 3, groups=mid)
        self.pw = conv_bn(mid, channels, 1, act=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.pw(self.dw(self.pre(x))))


class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.dw = conv_bn(cin, cin, 3, s=2, groups=cin)
        self.pw = conv_bn(cin, cout, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class Fuse(nn.Module):
    """1x1 channel projection followed by depthwise spatial smoothing."""

    def __init__(self, cin, cout):
        super().__init__()
        self.proj = conv_bn(cin, cout, 1)
        self.dw = conv_bn(cout, cout, 3, groups=cout)

    def forward(self, x):
        return self.dw(self.proj(x))


class RoadSegNet5(nn.Module):
    def __init__(self, classes=5, width=1.0):
        super().__init__()
        c = lambda n: max(8, int(round(n * width)))
        self.stem = nn.Sequential(
            conv_bn(3, c(24), 3, s=2),
            conv_bn(c(24), c(24), 3, s=2, groups=c(24)),
            conv_bn(c(24), c(32), 1),
        )
        self.stage2 = nn.Sequential(Down(c(32), c(48)), ResDW(c(48)), ResDW(c(48)))
        self.stage3 = nn.Sequential(Down(c(48), c(96)), ResDW(c(96), 2), ResDW(c(96), 2))
        self.stage4 = nn.Sequential(Down(c(96), c(160)), ResDW(c(160), 2), ResDW(c(160), 2))
        self.stage5 = nn.Sequential(
            conv_bn(c(160), c(256), 1), ResDW(c(256), 2), ResDW(c(256), 2)
        )
        self.ppm = nn.Sequential(conv_bn(c(256), c(128), 1))
        self.ppm_out = conv_bn(c(256) + c(128), c(256), 1)
        self.up4 = Fuse(c(256) + c(160), c(160))
        self.up3 = Fuse(c(160) + c(96), c(96))
        self.up2 = Fuse(c(96) + c(48), c(64))
        self.up1 = Fuse(c(64) + c(32), c(32))
        self.head = nn.Sequential(
            conv_bn(c(32), c(16), 1),
            conv_bn(c(16), c(16), 3, groups=c(16)),
            nn.Conv2d(c(16), classes, 1),
        )

    @staticmethod
    def _up(x, size):
        return F.interpolate(x, size=size, mode="nearest")

    def forward(self, x):
        s1 = self.stem(x)           # H/4
        s2 = self.stage2(s1)        # H/8
        s3 = self.stage3(s2)        # H/16
        s4 = self.stage4(s3)        # H/32
        s5 = self.stage5(s4)        # H/32, wider
        p = self._up(self.ppm(F.adaptive_avg_pool2d(s5, 1)), s5.shape[-2:])
        s5 = self.ppm_out(torch.cat([s5, p], 1))
        d4 = self.up4(torch.cat([self._up(s5, s4.shape[-2:]), s4], 1))
        d3 = self.up3(torch.cat([self._up(d4, s3.shape[-2:]), s3], 1))
        d2 = self.up2(torch.cat([self._up(d3, s2.shape[-2:]), s2], 1))
        d1 = self.up1(torch.cat([self._up(d2, s1.shape[-2:]), s1], 1))
        y = self.head(self._up(d1, x.shape[-2:]))
        return y


class MobileNetV3LRASPP(nn.Module):
    """torchvision MobileNetV3 backbone with LR-ASPP and stride-8 refinement."""

    def __init__(self, classes=5, pretrained=True, arch="small"):
        super().__init__()
        import torchvision.models as tvm
        if arch == "small":
            net = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self.c4_index, self.c5_index, self.c8_index = 8, 12, 2
            c4_ch, c5_ch, c8_ch = 48, 576, 24
        else:
            net = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
            self.c4_index, self.c5_index, self.c8_index = 12, 16, 6
            c4_ch, c5_ch, c8_ch = 112, 960, 40
        self.arch = arch
        self.features = net.features
        self.high_proj = conv_bn(c5_ch, 128, 1)
        self.high = conv_bn(128, 128, 3)
        self.pool = nn.Sequential(conv_bn(c5_ch, 128, 1))
        self.low = conv_bn(c4_ch, 128, 1)
        self.refine = nn.Sequential(
            conv_bn(128 + c8_ch, 64, 1),
            conv_bn(64, 64, 3, groups=64),
            nn.Conv2d(64, classes, 1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        c4 = c5 = c8 = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.c8_index:
                c8 = x
            elif i == self.c4_index:
                c4 = x
            elif i == self.c5_index:
                c5 = x
        p = self._up(self.pool(F.adaptive_avg_pool2d(c5, 1)), c5.shape[-2:])
        h = self.high(self.high_proj(c5) + p)
        h = self._up(h, c4.shape[-2:]) + self.low(c4)
        h = self._up(h, c8.shape[-2:])
        y = self.refine(torch.cat([h, c8], 1))
        return self._up(y, input_size)

    @staticmethod
    def _up(x, size):
        return F.interpolate(x, size=size, mode="nearest")


class MobileNetV3LRASPPRefine(MobileNetV3LRASPP):
    """LR-ASPP plus stride-4/2 refinement for thin lane markings."""

    def __init__(self, classes=5, pretrained=True, arch="small"):
        super().__init__(classes=classes, pretrained=pretrained, arch=arch)
        self.c1_index = 0
        self.c2_index = 1
        c1_ch, c2_ch = (16, 16) if arch == "small" else (16, 24)
        self.refine_head = nn.Conv2d(64, 64, 1)
        self.up4 = Fuse(64 + c2_ch, 32)
        self.up2 = Fuse(32 + c1_ch, 16)
        self.head = nn.Conv2d(16, classes, 1)

    def forward(self, x):
        input_size = x.shape[-2:]
        c1 = c2 = c8 = c4 = c5 = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.c1_index:
                c1 = x
            elif i == self.c2_index:
                c2 = x
            elif i == self.c8_index:
                c8 = x
            elif i == self.c4_index:
                c4 = x
            elif i == self.c5_index:
                c5 = x
        p = self._up(self.pool(F.adaptive_avg_pool2d(c5, 1)), c5.shape[-2:])
        h = self.high(self.high_proj(c5) + p)
        h = self._up(h, c4.shape[-2:]) + self.low(c4)
        h = self._up(h, c8.shape[-2:])
        y8 = self.refine[1](self.refine[0](torch.cat([h, c8], 1)))
        y8 = self.refine_head(y8)
        y4 = self.up4(torch.cat([self._up(y8, c2.shape[-2:]), c2], 1))
        y2 = self.up2(torch.cat([self._up(y4, c1.shape[-2:]), c1], 1))
        y = self.head(y2)
        return self._up(y, input_size)


class MobileNetV3LRASPPResidual(MobileNetV3LRASPP):
    """Learned stride-4/2 residual on top of the LR-ASPP stride-8 logits."""

    def __init__(self, classes=5, pretrained=True, arch="small"):
        super().__init__(classes=classes, pretrained=pretrained, arch=arch)
        self.c1_index = 0
        self.c2_index = 1
        c1_ch, c2_ch = (16, 16) if arch == "small" else (16, 24)
        self.refine_head = nn.Conv2d(64, 64, 1)
        self.up4 = Fuse(64 + c2_ch, 32)
        self.up2 = Fuse(32 + c1_ch, 16)
        self.head = nn.Conv2d(16, classes, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        input_size = x.shape[-2:]
        c1 = c2 = c8 = c4 = c5 = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.c1_index:
                c1 = x
            elif i == self.c2_index:
                c2 = x
            elif i == self.c8_index:
                c8 = x
            elif i == self.c4_index:
                c4 = x
            elif i == self.c5_index:
                c5 = x
        p = self._up(self.pool(F.adaptive_avg_pool2d(c5, 1)), c5.shape[-2:])
        h = self.high(self.high_proj(c5) + p)
        h = self._up(h, c4.shape[-2:]) + self.low(c4)
        h = self._up(h, c8.shape[-2:])
        y8 = self.refine[1](self.refine[0](torch.cat([h, c8], 1)))
        base = self._up(self.refine[2](y8), input_size)
        y4 = self.up4(torch.cat([self._up(self.refine_head(y8), c2.shape[-2:]), c2], 1))
        y2 = self.up2(torch.cat([self._up(y4, c1.shape[-2:]), c1], 1))
        return base + self._up(self.head(y2), input_size)


class CandidatePlus5(nn.Module):
    """Candidate-plus encoder/decoder with a 5-channel head.

    The first four head rows are transferred from the original 4-channel
    weights: merged drivable -> direct, zero -> alternative, then solid,
    dashed, crosswalk.  Fine-tuning learns the direct/alternative split while
    preserving the proven road/lane features.
    """

    def __init__(self):
        super().__init__()
        self.net = RoadCandidatePlus()
        old = self.net.head
        new = nn.Conv2d(32, 5, 1)
        with torch.no_grad():
            new.weight.zero_(); new.bias.zero_()
            for new_idx, old_idx in ((0, 0), (2, 1), (3, 2), (4, 3)):
                new.weight[new_idx].copy_(old.weight[old_idx])
                new.bias[new_idx].copy_(old.bias[old_idx])
            # Start both direct and alternative from the old merged drivable
            # head so their union begins at the proven baseline instead of an
            # all-negative alternative channel.
            new.weight[1].copy_(old.weight[0])
            new.bias[1].copy_(old.bias[0])
        self.net.head = new

    def forward(self, x):
        return self.net(x)


class CandidatePlusA(nn.Module):
    """Original architecture (4 channels, merged drivable) at 416x256."""

    def __init__(self):
        super().__init__()
        self.net = RoadCandidatePlus()

    def forward(self, x):
        return self.net(x)


def build_roadseg_model(name, classes=5, pretrained=False, arch="small", width=1.0):
    if name == "candidate-plus-a":
        return CandidatePlusA()
    if name == "candidate-plus-5":
        return CandidatePlus5()
    if name == "roadsegnet-5":
        return RoadSegNet5(classes=classes, width=width)
    if name == "mobilenetv3-lraspp":
        return MobileNetV3LRASPP(classes=classes, pretrained=pretrained, arch=arch)
    if name == "mobilenetv3-lraspp-refine":
        return MobileNetV3LRASPPRefine(classes=classes, pretrained=pretrained, arch=arch)
    if name == "mobilenetv3-lraspp-residual":
        return MobileNetV3LRASPPResidual(classes=classes, pretrained=pretrained, arch=arch)
    raise ValueError("unknown model: %s" % name)


def count_macs(model, size=(256, 416)):
    """Count Conv2d/Linear MACs for one forward pass (eval mode)."""
    total = [0]

    def hook(module, inp, out):
        if isinstance(module, nn.Conv2d):
            total[0] += out.numel() * module.kernel_size[0] * module.kernel_size[1] * module.in_channels // module.groups
        elif isinstance(module, nn.Linear):
            total[0] += out.numel() * module.in_features

    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(1, 3, *size, device=device))
    for h in handles:
        h.remove()
    return total[0]


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    for name in ("candidate-plus-a", "roadsegnet-5", "mobilenetv3-lraspp"):
        m = build_roadseg_model(name, pretrained=False)
        print(name, "params", parameter_count(m), "macs", count_macs(m))
