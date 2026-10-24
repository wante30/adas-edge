"""B-05L train one YOLOv8-family traffic-light Specialist and export fixed ONNX.

Examples:
  python tools/b05l_train.py --arch yolov8n --epochs 2 --name trial_v8n
  python tools/b05l_train.py --arch yolov8n --epochs 60 --name v8n
  python tools/b05l_train.py --arch yolov8n-p2 --epochs 60 --name v8n_p2

The primary comparison uses the same 8513/487 split as the production YOLO11n.
Weights are written under D:\Datasets\adas_edge_training\b05l_runs and are not
committed to Git.
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

DEFAULT_DATA = Path(r"D:\Datasets\adas_edge_training\b05l_data\primary_clean.yaml")
DEFAULT_PROJECT = Path(r"D:\Datasets\adas_edge_training\b05l_runs")
DEFAULT_YOLOV8N = Path(r"C:\Users\JingyuWang\adas-edge\yolov8n.pt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_signature(model: Any) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in model.state_dict().items()}


def transfer_stats(dest: Any, source_path: Path) -> dict[str, Any]:
    import torch

    source = torch.load(source_path, map_location="cpu", weights_only=False)["model"]
    src = state_signature(source)
    dst = state_signature(dest)
    matched = {key for key in dst if key in src and src[key] == dst[key]}
    matched_params = sum(dest.state_dict()[key].numel() for key in matched)
    total_params = sum(value.numel() for value in dest.state_dict().values())
    return {
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "destination_layers": len(dst),
        "matched_layers": len(matched),
        "destination_parameters": total_params,
        "matched_parameter_fraction": matched_params / total_params if total_params else 0.0,
        "unmatched_examples": sorted(set(dst) - matched)[:20],
        "note": "Exact key+shape matches before Ultralytics class-head remapping.",
    }


def transfer_stats_after(dest: Any, source_path: Path) -> dict[str, Any]:
    """Count exact tensor equality after Ultralytics' own load/remap step."""
    import torch

    source = torch.load(source_path, map_location="cpu", weights_only=False)["model"]
    src = source.state_dict()
    dst = dest.state_dict()
    matched = []
    for key, value in dst.items():
        if key in src and src[key].shape == value.shape and torch.equal(src[key].cpu(), value.cpu()):
            matched.append(key)
    total_params = sum(value.numel() for value in dst.values())
    matched_params = sum(dst[key].numel() for key in matched)
    return {
        "matched_layers": len(matched),
        "destination_layers": len(dst),
        "matched_parameters": matched_params,
        "destination_parameters": total_params,
        "matched_parameter_fraction": matched_params / total_params if total_params else 0.0,
        "unmatched_layers": sorted(set(dst) - set(matched))[:30],
    }


def build_model(arch: str, source: Path, *, p2_yaml: str):
    from ultralytics import YOLO

    if arch == "yolov8n":
        model = YOLO(str(source))
        stats = transfer_stats(model.model, source)
        stats["after_load"] = transfer_stats_after(model.model, source)
        return model, stats
    if arch == "yolov8n-p2":
        model = YOLO(p2_yaml)
        stats = transfer_stats(model.model, source)
        model.load(str(source))
        stats["loaded_with"] = "YOLO.load(yolov8n.pt)"
        stats["after_load"] = transfer_stats_after(model.model, source)
        return model, stats
    raise ValueError(f"unsupported arch: {arch}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=("yolov8n", "yolov8n-p2"), required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--source", type=Path, default=DEFAULT_YOLOV8N)
    parser.add_argument("--p2-yaml", default="yolov8-p2.yaml")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    args = parser.parse_args()

    import torch
    import torchvision

    # The installed torchvision 0.27.1+cpu has no CUDA NMS kernel.  Ultralytics
    # validation calls torchvision.ops.nms, so use its CPU kernel and move the
    # selected indices back to the original device.  Training itself is unaffected.
    _original_nms = torchvision.ops.nms

    def _nms_with_cpu_fallback(boxes, scores, iou_threshold):
        if boxes.is_cuda:
            indices = _original_nms(boxes.cpu(), scores.cpu(), iou_threshold)
            return indices.to(boxes.device)
        return _original_nms(boxes, scores, iou_threshold)

    torchvision.ops.nms = _nms_with_cpu_fallback

    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training run")
    if not args.source.exists():
        raise FileNotFoundError(args.source)
    if not args.data.exists():
        raise FileNotFoundError(args.data)

    model, transfer = build_model(args.arch, args.source, p2_yaml=args.p2_yaml)
    params = sum(value.numel() for value in model.model.state_dict().values())
    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        rect=True,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        seed=args.seed,
        deterministic=True,
        pretrained=True,
        patience=args.patience,
        close_mosaic=10,
        plots=True,
        verbose=True,
        exist_ok=args.exist_ok,
        cache=False,
        amp=True,
    )
    result = model.train(**train_kwargs)
    save_dir = Path(result.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)

    meta: dict[str, Any] = {
        "arch": args.arch,
        "name": args.name,
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
        "parameters": params,
        "transfer": transfer,
        "save_dir": str(save_dir),
        "best": str(best),
        "best_sha256": sha256(best),
        "results_dict": {key: float(value) for key, value in result.results_dict.items()},
    }
    if args.export:
        exported = YOLO(str(best)).export(
            format="onnx", imgsz=(192, 512), opset=12, simplify=True, dynamic=False, device=args.device
        )
        exported_path = Path(str(exported))
        if not exported_path.exists():
            raise FileNotFoundError(exported_path)
        import onnxruntime as ort

        session = ort.InferenceSession(str(exported_path), providers=["CPUExecutionProvider"])
        meta["onnx"] = {
            "path": str(exported_path),
            "sha256": sha256(exported_path),
            "size_bytes": exported_path.stat().st_size,
            "input": {"name": session.get_inputs()[0].name, "shape": session.get_inputs()[0].shape},
            "output": {"name": session.get_outputs()[0].name, "shape": session.get_outputs()[0].shape},
        }
        if session.get_inputs()[0].shape != [1, 3, 192, 512]:
            raise ValueError(f"unexpected exported input shape: {session.get_inputs()[0].shape}")
    (save_dir / "b05l_run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    print("B05L_RUN_META " + json.dumps(meta, ensure_ascii=True))


if __name__ == "__main__":
    main()
