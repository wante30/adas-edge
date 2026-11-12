"""Synthetic fixed-batch training throughput, not end-to-end road FPS."""
import argparse
import time
import torch
from .models import build_model
from .engine import masked_loss
from .dataset_adapter import write_json


def benchmark(name, size, batch=8, warmup=10, steps=50, device='cuda'):
    torch.manual_seed(15); torch.set_num_threads(4)
    model=build_model(name).to(device).train()
    channels=2 if name in ('twin-small','twin-nano') else 4
    x=torch.rand(batch,3,size[1],size[0],device=device)
    y=(torch.rand(batch,channels,size[1],size[0],device=device)>.8).float()
    v=torch.ones_like(y); opt=torch.optim.AdamW(model.parameters(),lr=.001)
    def step():
        opt.zero_grad(set_to_none=True); loss=masked_loss(model(x),y,v); loss.backward(); opt.step()
    for _ in range(warmup): step()
    if device=='cuda':
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    for _ in range(steps): step()
    if device=='cuda': torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    speed=steps*batch/seconds
    return dict(model=name,size=list(size),batch=batch,warmup_steps=warmup,timed_steps=steps,seconds=seconds,
                training_images_per_second=speed,peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                gpu=torch.cuda.get_device_name() if device=='cuda' else None,torch_version=torch.__version__,
                illustrative_70000_images_50_epochs_hours=70000*50/speed/3600,
                evidence='synthetic GPU training-step benchmark; excludes real image decode, IO, augmentation, validation and checkpoints; NOT real-data training or deployment FPS')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',required=True)
    args=p.parse_args()
    cases=(('candidate',(320,192)),('twin-small',(320,192)),('candidate',(512,288)),('twin-small',(512,288)))
    reports=[benchmark(name,size) for name,size in cases]
    write_json(args.output,reports)
    for r in reports: print(r['model'],r['size'],r['training_images_per_second'],r['illustrative_70000_images_50_epochs_hours'])


if __name__=='__main__': main()
