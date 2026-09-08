# ADAS Edge — 2026 FPGA Competition Software

Software-side implementation for the 2026 National Undergraduate Embedded Chip and System Design Competition, FPGA Innovation Design track, Topic 3. The intended system connects FPGA video capture and OSD with an RK3566 detector through PCIe metadata transport.

## What is included

- RK3566-oriented YOLO11n INT8 conversion, split-output decoding, and benchmark helpers.
- `CompetitionDetector`: preprocessing, RKNN output decoding, NMS, source-coordinate restoration, and stream/frame/timestamp propagation.
- A versioned FPGA ↔ RK3566 frame/result descriptor contract and software mock transport.
- Deterministic ADAS rules for pedestrians, traffic lights, lane-change restrictions, and warnings.
- Competition-focused tests, model-build records, external BDD100K image validation notes, and the project handover document.

## What is deliberately excluded

Weights, RKNN binaries, calibration images, raw datasets, road videos, rendered videos, local outputs, and benchmark artifacts are excluded because of file size, licensing, or reproducibility constraints. The public repository contains code and documentation only.

The current BDD100K result is a PC FP32 image-level pre-acceptance check; it is not an RK3566 INT8 accuracy result or an FPGA/PCIe end-to-end claim. See [the project handover](docs/competition_project_handover.md) for evidence, known gaps, and the execution order.

## Quick start

Use Python 3.10–3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
```

RKNN conversion needs a Rockchip-supported Toolkit2 environment. The original COCO `yolo11n.pt` checkpoint and legal calibration data must be supplied locally; no model artifact is bundled.

## Current status

- 320 and 416 INT8 split-output candidates have completed RK3566 single-image load/inference/decode smoke tests.
- Competition detector, scene policy, and PCIe software-mock tests pass.
- Real FPGA capture, PCIe endpoint/DMA, OSD return, continuous video validation, four-stream testing, and final demonstration remain pending.

## License

No project-wide license has been selected. Do not assume permission to reuse code or data beyond GitHub's normal viewing rights. Third-party datasets, model weights, and toolkits retain their own terms.
