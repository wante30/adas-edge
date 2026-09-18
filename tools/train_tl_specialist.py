"""Train the FRONT-only traffic-light specialist (single class, small canvas).

Reuses the rectangular training recipe proven in this project
(road-ft-512x288: imgsz=512, rect=True, mosaic=0). The dataset is already
letterboxed to the deployment canvas, so rect training converges on that exact
shape. Resume-safe: if interrupted, rerun with --resume pointing at last.pt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None, help="last.pt to resume from")
    args = parser.parse_args()

    if args.resume:
        model = YOLO(str(args.resume))
        model.train(resume=True)
        return

    model = YOLO("yolo11n.pt")  # COCO-80 pretrained backbone, new single-class head
    results = model.train(
        data=str(args.data),
        project=str(args.project),
        name=args.name,
        imgsz=args.imgsz,
        rect=True,       # rectangular training at the deployment aspect
        mosaic=0.0,      # mosaic would break the uniform canvas aspect
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        device="0",
        plots=True,
    )

    metrics = {key: float(value) for key, value in results.results_dict.items()}
    best = Path(results.save_dir) / "weights" / "best.pt"
    args.project.mkdir(parents=True, exist_ok=True)
    (args.project / f"{args.name}-summary.json").write_text(
        json.dumps(
            {
                "task": "tl-specialist",
                "data": str(args.data),
                "imgsz": args.imgsz,
                "rect": True,
                "mosaic": 0.0,
                "epochs": args.epochs,
                "batch": args.batch,
                "seed": args.seed,
                "save_dir": str(results.save_dir),
                "best": str(best),
                "final_metrics": metrics,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print("DONE", json.dumps(metrics))


if __name__ == "__main__":
    main()
