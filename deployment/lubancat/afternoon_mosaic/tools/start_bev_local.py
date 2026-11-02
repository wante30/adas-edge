"""Launch the installed BEV fix with local replay; original services stay intact."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WORK=Path('/dev/shm/adas_bev_runtime')

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',default='/home/cat/qt_gpu_probe_ai_20261019/local_quad_full.rgb');p.add_argument('--no-cache',action='store_true');p.add_argument('--baseline',action='store_true');args=p.parse_args()
    source=Path(args.raw);assert source.is_file(),str(source)
    WORK.mkdir(exist_ok=True);(WORK/'evidence').mkdir(exist_ok=True)
    for name,target in [('candidate',ROOT),('baseline_src',ROOT.parent/'baseline_src')]:
        link=WORK/name
        if not link.exists():link.symlink_to(target,target_is_directory=True)
        elif link.resolve()!=target.resolve():raise RuntimeError('Workspace belongs to another installation')
    raw=source
    if not args.no_cache:
        # Preserve bytes exactly. A 600-frame / 10-second excerpt loops in RAM.
        raw=WORK/'replay_600.rgb';size=min(source.stat().st_size,600*1280*720*2)
        if not raw.exists() or raw.stat().st_size!=size:
            tmp=WORK/'replay_600.rgb.tmp'
            if shutil.disk_usage(WORK).free<size+150000000:raise RuntimeError('Insufficient RAM space; use --no-cache')
            with source.open('rb') as src,tmp.open('wb') as dst:
                left=size
                while left:
                    data=src.read(min(left,8*1024*1024))
                    if not data:raise RuntimeError('Short replay source')
                    dst.write(data);left-=len(data)
            tmp.replace(raw)
    import run_bev_ab
    run_bev_ab.BASE=WORK
    sys.argv=[sys.argv[0],'baseline_src' if args.baseline else 'candidate','--label','local_'+str(time.monotonic_ns()),'--seconds','0','--raw',str(raw),'--no-bev-trace']
    run_bev_ab.main()
if __name__=='__main__':main()
