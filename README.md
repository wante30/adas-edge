# ADAS Edge

PC baseline and Rockchip deployment path for the 2026 FPGA ADAS contest entry.

## Competition software closeout scope

The canonical competition path is `adas_edge.competition`: a frozen v1 frame
descriptor is validated, decoded from BGR/RGB/NV12/RGB565, passed through an
injectable detector and deterministic ADAS policy, then encoded as a result
descriptor. This is a software closure and mock-E2E boundary, not a claim of
physical FPGA/PCIe completion.

PCIe DMA, BDD100K download/training, and custom network changes are deliberately
out of scope for this phase.

## Quick start

Use a clean Python 3.10--3.12 environment. The host Python detected during
development was 3.11; RKNN conversion is **not** supported from Windows.

```powershell
cd C:\Users\JingyuWang\adas-edge
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Run an image, video, or an image directory. `yolo11n.pt` may be a local COCO
weight path or an Ultralytics model name.

```powershell
.\.venv\Scripts\python.exe -m adas_edge.cli --source C:\data\road.mp4 --model yolo11n.pt --show --output-video outputs\road.mp4 --output-jsonl outputs\road.jsonl --benchmark-csv outputs\benchmark.csv
.\.venv\Scripts\python.exe -m adas_edge.cli --camera 0 --camera-id front --model yolo11n.pt --show
```

Press `q` in the preview window to stop. Per-frame JSON is written to stdout
and, when requested, to JSON Lines. The rendered frame contains boxes, action,
latency, and stream FPS.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe tools\check_environment.py
```

See [architecture](docs/architecture.md), [competition software closeout](docs/competition_software_closeout.md),
and the [RKNN deployment boundary](deployment/rknn/README.md).
