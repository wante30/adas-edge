"""Run the frozen-holdout comparison matrix for the road-domain pilot.

Five configurations, all measured on the same 1000-image BDD100K holdout slice
with the historical matching protocol (per class, one-to-one, conf 0.25, IoU 0.5):

===========================  ====================================================
BASE-320                     speed baseline: COCO ``yolo11n.pt`` at 320x320
BASE-416                     quality baseline: COCO ``yolo11n.pt`` at 416x416
BASE-512x288                 input-geometry diagnostic: COCO ``yolo11n.pt`` at 512x288
Road-FT-320                  road-domain adaptation at the existing square geometry
Road-FT-512x288              road-domain adaptation at 16:9
===========================  ====================================================

Pairwise reading: BASE-320 -> Road-FT-320 isolates the data-domain variable;
Road-FT-320 -> Road-FT-512x288 isolates the input-geometry variable;
BASE-320 -> BASE-512x288 isolates geometry with the COCO weights held fixed.

Each configuration is evaluated in its own subprocess so one failure cannot
corrupt the rest, and every per-model JSON plus a combined matrix is written.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CLASSES = ["car", "truck", "bus", "person", "motorcycle", "bicycle", "traffic light"]


def default_matrix(weights: Path, road: Path, geometry: str = "fixed") -> list[dict]:
    """The five planned configurations.

    Geometry is explicitly ``fixed``: the deployed RKNN graph has one frozen
    input shape, so the holdout must be measured with that same canvas. Ultralytics
    defaults to ``rect=True`` for prediction, which would silently turn "320" into
    a 192x320 tensor and make the label mean something the deployment never runs.
    """
    return [
        {"label": "BASE-320", "model": weights, "imgsz": "320", "geometry": geometry},
        {"label": "BASE-416", "model": weights, "imgsz": "416", "geometry": geometry},
        {"label": "BASE-512x288", "model": weights, "imgsz": "512x288", "geometry": geometry},
        {"label": "Road-FT-320", "model": road / "road-ft-320.pt", "imgsz": "320", "geometry": geometry},
        {"label": "Road-FT-512x288", "model": road / "road-ft-512x288.pt", "imgsz": "512x288", "geometry": geometry},
    ]


def run_one(entry: dict, bdd_root: Path, out_dir: Path, limit: int, device: str, evaluator: Path) -> Path:
    report = out_dir / f"{entry['label']}.json"
    markdown = out_dir / f"{entry['label']}.md"
    command = [
        sys.executable,
        str(evaluator),
        "--model", str(entry["model"]),
        "--bdd-root", str(bdd_root),
        "--out", str(report),
        "--markdown", str(markdown),
        "--imgsz", str(entry["imgsz"]),
        "--limit", str(limit),
        "--device", device,
        "--geometry", entry.get("geometry", "fixed"),
        "--label", entry["label"],
    ]
    print(f"\n=== {entry['label']} :: {entry['model']} @ {entry['imgsz']} ===", flush=True)
    completed = subprocess.run(command, text=True, capture_output=True)
    sys.stdout.write(completed.stdout)
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr[-4000:])
        raise SystemExit(f"{entry['label']} evaluation failed with exit code {completed.returncode}")
    return report


def assemble(reports: list[tuple[str, dict]], out_dir: Path) -> dict:
    lines = [
        "# Road-domain pilot: frozen BDD100K holdout matrix",
        "",
        f"- holdout images: {reports[0][1]['images']}",
        "- matching: one-to-one per class, conf >= 0.25, IoU >= 0.5",
        "",
        "## Per-class recall / F1",
        "",
        "| class | " + " | ".join(f"{label} R / F1" for label, _ in reports) + " |",
        "| --- | " + " | ".join("---" for _ in reports) + " |",
    ]
    for name in CLASSES:
        cells = []
        for _, report in reports:
            value = report["classes"][name]
            cells.append(f"{value['recall']:.3f} / {value['f1']:.3f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append(
        "| **macro** | "
        + " | ".join(f"**{report['macro_recall']:.3f}** / **{report['macro_f1']:.3f}**" for _, report in reports)
        + " |"
    )

    lines += ["", "## Small-object recall (COCO area buckets on original pixels)", "", "| bucket | " + " | ".join(label for label, _ in reports) + " |", "| --- | " + " | ".join("---" for _ in reports) + " |"]
    for bucket in ("small", "medium", "large"):
        cells = [f"{report['size_buckets'][bucket]['recall']:.3f}" for _, report in reports]
        lines.append(f"| {bucket} | " + " | ".join(cells) + " |")
    lines.append("| small GT count | " + " | ".join(str(report["size_buckets"]["small"]["gt"]) for _, report in reports) + " |")

    lines += ["", "## Input geometry actually used (wall-clock throughput)", "", "| metric | " + " | ".join(label for label, _ in reports) + " |", "| --- | " + " | ".join("---" for _ in reports) + " |"]
    lines.append("| requested | " + " | ".join(str(report["size"]) for _, report in reports) + " |")
    lines.append("| geometry | " + " | ".join(report["geometry"] for _, report in reports) + " |")
    lines.append(
        "| measured network input | "
        + " | ".join(
            " / ".join(f"{shape[1]}x{shape[2]}" for shape in report["observed_input_shapes"])
            for _, report in reports
        )
        + " |"
    )
    lines.append(
        "| pixel budget | "
        + " | ".join(
            " / ".join(str(shape[1] * shape[2]) for shape in report["observed_input_shapes"])
            for _, report in reports
        )
        + " |"
    )
    lines.append(
        "| wall ms/frame | "
        + " | ".join(f"{report['wall_ms_per_image']:.2f}" if report["wall_ms_per_image"] is not None else "n/a" for _, report in reports)
        + " |"
    )
    lines.append("| detections | " + " | ".join(str(report["detections"]) for _, report in reports) + " |")

    matrix = {
        "holdout_images": reports[0][1]["images"],
        "conf": reports[0][1]["conf"],
        "iou_match": reports[0][1]["iou_match"],
        "labels": [label for label, _ in reports],
        "per_class": {
            name: {label: report["classes"][name] for label, report in reports} for name in CLASSES
        },
        "macro_recall": {label: report["macro_recall"] for label, report in reports},
        "macro_f1": {label: report["macro_f1"] for label, report in reports},
        "geometry": {label: report["geometry"] for label, report in reports},
        "observed_input_shapes": {label: report["observed_input_shapes"] for label, report in reports},
        "wall_ms_per_image": {label: report["wall_ms_per_image"] for label, report in reports},
        "size_buckets": {label: report["size_buckets"] for label, report in reports},
        "per_class_size_recall": {label: report["per_class_size_recall"] for label, report in reports},
        "detections": {label: report["detections"] for label, report in reports},
        "models": {label: report["model"] for label, report in reports},
        "sizes": {label: report["size"] for label, report in reports},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "matrix.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bdd-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("yolo11n.pt"), help="COCO baseline checkpoint")
    parser.add_argument("--road-dir", type=Path, default=Path("D:/Datasets/adas_edge_training/road_pilot"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--device", default="0")
    parser.add_argument("--only", help="comma separated labels to run a subset of the matrix")
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="reuse <out-dir>/<label>.json when present instead of re-evaluating that row",
    )
    parser.add_argument(
        "--geometry",
        choices=("fixed", "minrect"),
        default="fixed",
        help="fixed = the deployed frozen canvas (default); minrect = Ultralytics' minimum rectangle",
    )
    args = parser.parse_args()

    evaluator = Path(__file__).with_name("evaluate_bdd_yolo.py")
    entries = default_matrix(args.weights, args.road_dir, args.geometry)
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        entries = [entry for entry in entries if entry["label"] in wanted]
        missing = wanted - {entry["label"] for entry in entries}
        if missing:
            parser.error(f"unknown matrix label(s): {sorted(missing)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports: list[tuple[str, dict]] = []
    for entry in entries:
        cached = args.out_dir / f"{entry['label']}.json"
        if args.reuse and cached.exists():
            print(f"=== {entry['label']} :: reusing {cached} ===", flush=True)
            reports.append((entry["label"], json.loads(cached.read_text(encoding="utf-8"))))
            continue
        if not Path(entry["model"]).exists():
            raise SystemExit(f"missing checkpoint for {entry['label']}: {entry['model']}")
        path = run_one(entry, args.bdd_root, args.out_dir, args.limit, args.device, evaluator)
        reports.append((entry["label"], json.loads(path.read_text(encoding="utf-8"))))

    matrix = assemble(reports, args.out_dir)
    print("\n" + (args.out_dir / "matrix.md").read_text(encoding="utf-8"))
    print(f"macro recall: {matrix['macro_recall']}")


if __name__ == "__main__":
    main()
