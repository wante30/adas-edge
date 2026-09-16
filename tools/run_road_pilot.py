"""Train the Road-FT road-domain pilots from the pretrained COCO checkpoint.

Recipe source: ``configs/road_ft_pilot.yaml``. Both pilots start from the same
``yolo11n.pt`` and the same road-domain dataset, so the only variable left is
input geometry and pixel budget:

``320``       ``imgsz=320, rect=False`` -> a 320x320 square canvas, the same input
              geometry as BASE-320. Comparing it against BASE-320 isolates the
              data-domain variable.
``512x288``   ``imgsz=512, rect=True``  -> Ultralytics reshapes each batch to the
              batch aspect ratio, which for this all-16:9 road dataset is exactly
              ``ceil(0.5625 * 512 / 32) * 32 = 288`` by ``512``. Comparing it
              against Road-FT-320 isolates the 16:9 geometry variable.

Ultralytics cannot train a two-element ``imgsz`` (``trainer.py`` calls
``check_imgsz(..., max_dim=1)``, which collapses ``(512, 288)`` to ``512``), so
``rect=True`` is the supported route to a rectangular training input.

Mosaic/mixup/cutmix are disabled in both pilots because ``rect=True`` zeroes them
inside ``YOLODataset``; leaving them on for the square pilot would confound the
geometry comparison with an augmentation change.

The pretrained 80-class head is preserved: the dataset declares the 80 COCO names
and encodes the original COCO ids, so Ultralytics finds ``nc == 80`` and loads
every head weight instead of re-initialising it.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_CONFIG = Path("configs/road_ft_pilot.yaml")


def load_recipe(path: Path) -> dict:
    import yaml

    recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
    pilots = recipe.get("pilots") or {}
    if not pilots:
        raise SystemExit(f"{path} defines no pilots")
    for name, entry in pilots.items():
        for field in ("slug", "imgsz", "rect", "expected_input"):
            if field not in entry:
                raise SystemExit(f"{path}: pilot {name} is missing {field!r}")
    return recipe


def train_pilot(
    name: str,
    entry: dict,
    weights: Path,
    data: Path,
    project: Path,
    settings: dict,
) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model.train(
        data=str(data),
        imgsz=entry["imgsz"],
        rect=entry["rect"],
        epochs=settings["epochs"],
        batch=settings["batch"],
        workers=settings["workers"],
        device=settings["device"],
        seed=settings["seed"],
        deterministic=True,
        patience=settings["patience"],
        mosaic=settings["mosaic"],
        mixup=settings["mixup"],
        cutmix=settings["cutmix"],
        val=True,
        plots=False,
        project=str(project),
        name=f"road-ft-{entry['slug']}",
        exist_ok=True,
        verbose=True,
    )
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    canonical = project / f"road-ft-{entry['slug']}.pt"
    if best.exists():
        shutil.copy2(best, canonical)
    manifest = {
        "pilot": name,
        "slug": entry["slug"],
        "weights": str(weights),
        "data": str(data),
        "imgsz": entry["imgsz"],
        "rect": entry["rect"],
        "expected_train_input": entry["expected_input"],
        "epochs": settings["epochs"],
        "batch": settings["batch"],
        "mosaic": settings["mosaic"],
        "seed": settings["seed"],
        "device": settings["device"],
        "save_dir": str(save_dir),
        "best": str(canonical) if canonical.exists() else None,
        "last": str(last) if last.exists() else None,
        "final_metrics": dict(getattr(results, "results_dict", {}) or {}),
    }
    (project / f"road-ft-{entry['slug']}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, help="override the dataset.yaml from the config")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--project", type=Path, default=Path("D:/Datasets/adas_edge_training/road_pilot"))
    parser.add_argument("--pilots", help="override the config, e.g. 320,512x288")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--patience", type=int)
    args = parser.parse_args()

    recipe = load_recipe(args.config)
    settings = {
        "epochs": args.epochs if args.epochs is not None else recipe.get("epochs", 25),
        "batch": args.batch if args.batch is not None else recipe.get("batch", 16),
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed if args.seed is not None else recipe.get("seed", 0),
        "patience": args.patience if args.patience is not None else recipe.get("patience", 50),
        "mosaic": recipe.get("mosaic", 0.0),
        "mixup": recipe.get("mixup", 0.0),
        "cutmix": recipe.get("cutmix", 0.0),
    }
    data = args.data or Path(recipe["dataset"])
    weights = args.weights or Path(recipe.get("weights", "yolo11n.pt"))
    if not data.exists():
        raise SystemExit(f"dataset config not found: {data}")

    pilots = recipe["pilots"]
    if args.pilots:
        wanted = {part.strip() for part in args.pilots.split(",") if part.strip()}
        selected = {name: entry for name, entry in pilots.items() if entry["slug"] in wanted or name in wanted}
        missing = wanted - {entry["slug"] for entry in selected.values()} - set(selected)
        if missing:
            raise SystemExit(f"unknown pilot(s): {sorted(missing)}; available: {[e['slug'] for e in pilots.values()]}")
    else:
        selected = pilots

    args.project.mkdir(parents=True, exist_ok=True)
    for name, entry in selected.items():
        print(f"\n=== {name} :: imgsz={entry['imgsz']} rect={entry['rect']} ===", flush=True)
        manifest = train_pilot(name, entry, weights, data, args.project, settings)
        print(json.dumps(manifest, indent=2), flush=True)
    print(f"\nartifacts in {args.project}")


if __name__ == "__main__":
    main()
