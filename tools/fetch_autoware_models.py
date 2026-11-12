"""Fetch pinned Autoware traffic-light ONNX artifacts without committing weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


FILES = {
    "classifier": "traffic_light_classifier_mobilenetv2_batch_1.onnx",
    "lamp_recognizer": "traffic_light_lamp_recognizer_comlops.onnx",
    "labels": "lamp_labels.txt",
    "lamp_params": "lamp_recognizer_ml.param.yaml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "adas-edge-stage3/1.0"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", default="v4.0")
    parser.add_argument("--base-url", default="https://huggingface.co/AutowareFoundation/traffic_light_classifier/resolve")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, name in FILES.items():
        destination = args.output_dir / name
        url = f"{args.base_url.rstrip('/')}/{args.revision}/{name}"
        if not destination.exists():
            try:
                fetch(url, destination)
            except Exception as exc:
                rows.append({"key": key, "file": name, "url": url, "status": "download_failed", "error": str(exc)})
                continue
        rows.append({"key": key, "file": name, "url": url, "status": "available", "sha256": sha256(destination), "bytes": destination.stat().st_size})
    manifest = {"revision": args.revision, "source": "AutowareFoundation/traffic_light_classifier", "files": rows}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
