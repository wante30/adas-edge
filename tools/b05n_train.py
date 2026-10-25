"""B-05N train full-frame ten-class YOLOv8n / YOLOv8n-P2 on BDD100K.

Initialization choices:
  coco    - COCO yolov8n.pt
  bdd     - formal BDD-YOLOv8n checkpoint (10 classes)
  b05l-p2 - B-05L single-class P2 specialist (feature transfer only)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = Path(r"D:\Datasets\adas_edge_training\b05n_bdd_multiclass\data.yaml")
PROJECT = Path(r"D:\Datasets\adas_edge_training\b05n_runs")
COCO = Path(r"C:\Users\JingyuWang\adas-edge\yolov8n.pt")
BDD = Path(r"D:\Datasets\adas_edge_training\b03b_pretrained_cache_20261021\bdd100k-yolov8n-best.pt")
B05L_P2 = Path(r"D:\Datasets\adas_edge_training\b05l_runs\v8n_p2\weights\best.pt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(model: Any) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in model.state_dict().items()}


def transfer_stats(dest: Any, source_path: Path) -> dict[str, Any]:
    import torch

    source = torch.load(source_path, map_location="cpu", weights_only=False)["model"]
    src, dst = signature(source), signature(dest)
    matched = {key for key in dst if key in src and src[key] == dst[key]}
    total = sum(value.numel() for value in dest.state_dict().values())
    matched_params = sum(dest.state_dict()[key].numel() for key in matched)
    return {
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "source_classes": int(getattr(source, "nc", 0)),
        "destination_classes": int(getattr(dest, "nc", 0)),
        "matched_layers": len(matched),
        "destination_layers": len(dst),
        "matched_parameter_fraction": matched_params / total if total else 0.0,
        "unmatched_examples": sorted(set(dst) - matched)[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("yolov8n", "yolov8n-p2"), required=True)
    parser.add_argument("--init", choices=("coco", "bdd", "b05l-p2", "scratch"), default="bdd")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--init-path", type=Path, default=None)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--p2-yaml", default="yolov8-p2.yaml")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    args = parser.parse_args()

    import torch
    import torchvision

    _original_nms = torchvision.ops.nms

    def nms_cpu_fallback(boxes, scores, iou_threshold):
        if boxes.is_cuda:
            return _original_nms(boxes.cpu(), scores.cpu(), iou_threshold).to(boxes.device)
        return _original_nms(boxes, scores, iou_threshold)

    torchvision.ops.nms = nms_cpu_fallback
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    source = args.init_path or {"coco": COCO, "bdd": BDD, "b05l-p2": B05L_P2}.get(args.init)
    if args.init != "scratch" and (source is None or not source.exists()):
        raise FileNotFoundError(source)

    if args.arch == "yolov8n":
        model = YOLO(str(source)) if args.init != "scratch" else YOLO("yolov8n.yaml")
        if args.init in {"coco", "bdd"}:
            stats = transfer_stats(model.model, source)
        else:
            stats = {"note": "standard YOLOv8n cannot use single-class P2 head; scratch or COCO/BDD expected"}
    else:
        model = YOLO(args.p2_yaml)
        stats = transfer_stats(model.model, source) if source is not None else {"note": "scratch"}
        if source is not None:
            model.load(str(source))
            stats["loaded_with"] = f"YOLO.load({source.name})"
    params = sum(value.numel() for value in model.model.state_dict().values())
    result = model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, rect=True,
        batch=args.batch, device=args.device, workers=args.workers, project=str(args.project),
        name=args.name, seed=0, deterministic=True, pretrained=True, patience=args.patience,
        close_mosaic=10, plots=True, verbose=True, exist_ok=args.exist_ok, cache=False, amp=True,
    )
    save_dir = Path(result.save_dir)
    best = save_dir / "weights" / "best.pt"
    meta: dict[str, Any] = {
        "arch": args.arch, "init": args.init, "init_path": str(source) if source else None,
        "data": str(args.data.resolve()), "epochs": args.epochs, "batch": args.batch,
        "imgsz": args.imgsz, "seed": 0, "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version,
        "platform": platform.platform(), "parameters": params, "transfer": stats,
        "save_dir": str(save_dir), "best": str(best), "best_sha256": sha256(best),
        "results_dict": {key: float(value) for key, value in result.results_dict.items()},
    }
    if args.export:
        exported = YOLO(str(best)).export(format="onnx", imgsz=(288, 512), opset=12, simplify=True,
                                          dynamic=False, device=args.device)
        path = Path(str(exported))
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        meta["onnx"] = {
            "path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size,
            "input": {"name": session.get_inputs()[0].name, "shape": session.get_inputs()[0].shape},
            "output": {"name": session.get_outputs()[0].name, "shape": session.get_outputs()[0].shape},
        }
    (save_dir / "b05n_run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    print("B05N_RUN_META " + json.dumps(meta, ensure_ascii=True))


if __name__ == "__main__":
    main()
