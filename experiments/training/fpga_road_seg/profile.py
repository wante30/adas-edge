"""Executed module shapes and arithmetic; count FLOPs as 2*MACs.

MAC count includes Conv/ConvTranspose/Linear/MatMul. Excludes BN, activations,
pooling, Add, resize, memory traffic. Not latency or FPGA resource utilization.
"""
import argparse
from collections import Counter
import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from .models import build_model
from .dataset_adapter import write_json


class MatmulCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__(); self.macs=0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out=func(*args,**(kwargs or {}))
        if func in (torch.ops.aten.mm.default,torch.ops.aten.bmm.default):
            self.macs += out.numel()*args[0].shape[-1]
        # nn.Linear dispatches addmm; counted by its module hook instead.
        return out


def profile_model(model, shape):
    records,handles=[],[]
    macs=0
    def pair(value):
        return [value,value] if isinstance(value,int) else list(value)
    def hook(name):
        def record(m, inp, out):
            nonlocal macs
            op=type(m).__name__; nmac=0
            if isinstance(m,nn.Conv2d):
                nmac=out.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
            elif isinstance(m,nn.ConvTranspose2d):
                nmac=inp[0].numel()*(m.out_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
            elif isinstance(m,nn.Linear):
                # Linear without bias dispatches mm/bmm, counted below.
                if m.bias is not None:
                    nmac=out.numel()*m.in_features
            macs+=nmac
            records.append(dict(name=name,op=op,input_shape=list(inp[0].shape),output_shape=list(out.shape),macs=nmac,
                groups=getattr(m,'groups',None),dilation=list(m.dilation) if hasattr(m,'dilation') else None,
                kernel=pair(m.kernel_size) if hasattr(m,'kernel_size') else None))
        return record
    for name,m in model.named_modules():
        if not list(m.children()):
            handles.append(m.register_forward_hook(hook(name)))
    try:
        with torch.no_grad(),MatmulCounter() as mm:
            output=model.eval()(torch.zeros(shape))
    finally:
        for h in handles: h.remove()
    return dict(parameters=sum(p.numel() for p in model.parameters()),shape=list(shape),output_shape=list(output.shape),
                macs=macs+mm.macs,flops_2_per_mac=2*(macs+mm.macs),matmul_macs=mm.macs,
                arithmetic_scope=__doc__,modules=records,module_counts=dict(Counter(r['op'] for r in records)))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    args=p.parse_args(); torch.set_num_threads(4)
    reports=[]
    for name in ('twin-small','twin-nano','twin-four','candidate','candidate-plus'):
        for w,h in ((320,192),(384,216),(512,288),(640,384)):
            try:
                report=profile_model(build_model(name),(1,3,h,w))
                report.update(model=name,status='executed_cpu')
            except (RuntimeError,ValueError) as e:
                report=dict(model=name,shape=[1,3,h,w],status='unsupported_shape',error=str(e))
            reports.append(report)
    write_json(args.output,reports)


if __name__ == '__main__':
    main()
