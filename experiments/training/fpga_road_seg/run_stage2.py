"""Detached, restartable stage-two controller; never repeats stage-one training."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT/'outputs/fpga_road_seg_stage2'))
    parser.add_argument('--manifest', default=r'C:\Users\JingyuWang\datasets\rlmd_issue15\prepared_v1\mixed_manifest.jsonl')
    parser.add_argument('--postprocess-only', action='store_true',help='use existing FP32 checkpoints without training')
    parser.add_argument('--force-qat', action='store_true',help='request bounded QAT after real FP32 optimization')
    parser.add_argument('--report-only', action='store_true',help='reuse existing evaluation and visuals when recovering delivery')
    args = parser.parse_args()
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    # OS-held lock is released automatically after a crash; file can remain.
    lock = (out/'controller.lock').open('a+b')
    lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    env = os.environ.copy()
    env['BDD_SEG_CACHE'] = r'C:\Users\JingyuWang\datasets\bdd100k_issue15\tensor_cache_320x192_v1'
    env['PYTHONUNBUFFERED'] = '1'; env['PYTHONIOENCODING'] = 'utf-8'
    checkpoint_dir = out/'checkpoints/candidate-plus'
    config = ROOT/'experiments/training/fpga_road_seg/configs/stage2_candidate_plus.json'
    init = ROOT/'outputs/fpga_road_seg_final/checkpoints/candidate-plus/best.pt'
    def state(phase, **values):
        record = dict(phase=phase, controller_pid=os.getpid(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **values)
        tmp = out/'status.tmp.json'; tmp.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        os.replace(tmp, out/'status.json')
    with (out/'run.log').open('a', encoding='utf-8', buffering=1) as log:
        def run(command):
            log.write('\nUTC '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+repr(command)+'\n'); log.flush()
            return subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        done = out/'training_complete.json'
        if args.postprocess_only and not (checkpoint_dir/'best.pt').exists():
            raise ValueError('postprocess-only requires an existing FP32 best checkpoint')
        if not done.exists() and not args.postprocess_only:
            for attempt in range(1, 6):
                command = [sys.executable, '-u', '-m', 'experiments.training.fpga_road_seg.train',
                           '--manifest', args.manifest, '--config', str(config), '--output', str(checkpoint_dir),
                           '--amp', '--workers', '4', '--device', 'cuda']
                last = checkpoint_dir/'last.pt'
                if not last.exists() and checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
                    # A crash during initial evaluation can leave provenance but
                    # no resumable epoch. Preserve that attempt, then retry init.
                    checkpoint_dir.rename(checkpoint_dir.with_name(checkpoint_dir.name+f'.precheckpoint-{time.time_ns()}'))
                command.extend(['--resume', str(last)] if last.exists() else ['--init', str(init)])
                state('training', attempt=attempt, command=command)
                code = run(command)
                if code == 0:
                    done.write_text(json.dumps(dict(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n', encoding='utf-8')
                    break
                state('recovering_training', attempt=attempt, exit_code=code)
                time.sleep(10)
            else:
                state('failed_training', reason='five attempts failed; inspect run.log; rerun controller to resume')
                raise SystemExit(1)
        state('evaluating_and_exporting')
        command = [sys.executable, '-u', '-m', 'experiments.training.fpga_road_seg.deliver_stage2',
                    '--output', str(out), '--manifest', args.manifest, '--baseline', str(init)]
        if args.force_qat:
            command.append('--force-qat')
        if args.report_only:
            command.append('--report-only')
        code = run(command)
        if code:
            state('failed_delivery', exit_code=code, reason='training preserved; rerun controller to retry delivery only')
            raise SystemExit(code)
        state('complete')
    lock.close()


if __name__ == '__main__':
    main()
