"""Restartable full-training launcher for RoadSeg-RK."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--init")
    p.add_argument("--init-partial", action="store_true")
    p.add_argument("--teacher")
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--epochs", type=int)
    p.add_argument("--attempts", type=int, default=3)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log = Path(args.log); log.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, args.attempts + 1):
        cmd = [sys.executable, "-u", "-m", "experiments.training.roadseg_rk.train_roadseg",
               "--index", args.index, "--config", args.config, "--output", str(out),
               "--batch-size", str(args.batch_size), "--workers", str(args.workers), "--amp"]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        last = out / "last.pt"
        if last.exists():
            cmd += ["--resume", str(last)]
        elif args.init:
            cmd += ["--init", args.init]
            if args.init_partial:
                cmd += ["--init-partial"]
        if args.teacher:
            cmd += ["--teacher", args.teacher]
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n==== attempt {attempt} {time.strftime('%Y-%m-%dT%H:%M:%S')} ====\n")
            handle.write(" ".join(cmd) + "\n")
            handle.flush()
            code = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT).returncode
        if code == 0:
            (out / "TRAINING_DONE.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n", encoding="utf-8")
            return
        time.sleep(10)
    raise SystemExit("training failed after attempts")


if __name__ == "__main__":
    main()
