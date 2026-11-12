"""Reference architecture and a separately named, operator-limited candidate."""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

from .vendor.twinlite.model import TwinLiteNetPlus, Encoder, CAAM, ConvBatchnormRelu, UpConvBlock


class TwinBinary(nn.Module):
    """Original two independent 2-class heads, exposed as binary logits.

    sigmoid(foreground-background) == softmax(head)[:,1]. Architecture unchanged.
    """
    def __init__(self, variant='small'):
        super().__init__()
        self.network = TwinLiteNetPlus(SimpleNamespace(config=variant))

    def forward(self, x):
        da, lane = self.network(x)
        return torch.stack((da[:, 1]-da[:, 0], lane[:, 1]-lane[:, 0]), 1)


class TwinFour(nn.Module):
    """Software four-mask extension; attention/deconv still FPGA blockers."""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder('small')
        self.attention = CAAM(32, 32, (2, 4), nn.BatchNorm2d)
        self.neck = ConvBatchnormRelu(32, 16)
        self.up1 = UpConvBlock(16, 8)
        self.up2 = UpConvBlock(8, 8)
        self.head = nn.Conv2d(8, 4, 1)

    def forward(self, x):
        e, half, quarter = self.encoder(x)
        y = self.up2(self.up1(self.neck(self.attention(e)), quarter), half)
        return F.interpolate(self.head(y), size=x.shape[-2:], mode='nearest')


def conv(cin, cout, stride=1, groups=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, groups=groups, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


def ds(cin, cout, stride=1):
    return nn.Sequential(conv(cin, cin, stride, cin), nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


class RoadCandidate(nn.Module):
    """Hypothetical PG2L200H profile: standard/depthwise Conv + ReLU + Add + nearest.

    BN folds into Conv at export. No claim that the teammate core supports these.
    The graph exports logits; sigmoid/threshold can run on the software side.
    """
    def __init__(self):
        super().__init__()
        self.stem = conv(3, 16, 2)
        self.enc2 = ds(16, 24, 2)
        self.enc3 = nn.Sequential(ds(24, 48, 2), ds(48, 48))
        self.project = nn.Conv2d(48, 24, 1)
        self.dec2 = ds(24, 24)
        self.project2 = nn.Conv2d(24, 16, 1)
        self.dec1 = ds(16, 16)
        self.head = nn.Conv2d(16, 4, 1)

    def forward(self, x):
        a = self.stem(x)
        b = self.enc2(a)
        c = self.enc3(b)
        d = self.dec2(F.interpolate(self.project(c), size=b.shape[-2:], mode='nearest') + b)
        e = self.dec1(F.interpolate(self.project2(d), size=a.shape[-2:], mode='nearest') + a)
        return F.interpolate(self.head(e), size=x.shape[-2:], mode='nearest')


class RoadCandidatePlus(nn.Module):
    """Wider FPGA-limited variant for the requested 30K-80K comparison."""
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
        d = self.dec2(F.interpolate(self.project(c), size=b.shape[-2:], mode='nearest') + b)
        e = self.dec1(F.interpolate(self.project2(d), size=a.shape[-2:], mode='nearest') + a)
        return F.interpolate(self.head(e), size=x.shape[-2:], mode='nearest')


def build_model(name):
    if name in ('twin-small', 'twin-nano'):
        return TwinBinary(name.split('-')[1])
    if name == 'twin-four':
        return TwinFour()
    if name == 'candidate':
        return RoadCandidate()
    if name == 'candidate-plus':
        return RoadCandidatePlus()
    raise ValueError(f'unknown model {name}')
