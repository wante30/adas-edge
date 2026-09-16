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
- The round-2 road-domain pilot is complete. It does not clear the RKNN INT8 gate, because traffic light regresses to zero; no board deployment follows from it.
- Competition detector, scene policy, and PCIe software-mock tests pass.
- Real FPGA capture, PCIe endpoint/DMA, OSD return, continuous video validation, four-stream testing, and final demonstration remain pending.

## Round-2 road-domain pilot

The second experiment replaces the failed COCO-subset fine-tune with real
road-domain data and a non-square input. Method: [the road-domain pilot
note](docs/road_domain_pilot.md). Results and the deployment gate:
[the results note](docs/road_domain_pilot_results.md).

- BDD100K train was unreachable from the development host (official portal TLS
  mismatch, `dl.cv.ethz.ch` NXDOMAIN, S3 buckets 404/403, no labels-bearing
  mirror on the Hub), so the pilot trains on **nuImages** keyframes via
  `tools/prepare_nuimages_road_subset.py`. nuImages has no `traffic light`
  category, which turns out to matter - see the gate below.
- The 80-class pretrained head is preserved (labels keep their original COCO ids),
  so the pilot stays comparable with BASE-320/416:
  `Transferred 499/499 items from pretrained weights`.
  `tools/transfer_coco_head.py` narrows the head to seven classes for deployment
  without discarding the pretrained class trunk, verified against the COCO logits
  it claims to keep.
- `tools/run_road_pilot.py` trains Road-FT-320 and Road-FT-512x288;
  `tools/run_road_matrix.py` compares all five models on the frozen 1000-image
  holdout with macro recall and COCO small/medium/large buckets.
- Every evaluation records the input shape the network actually received.
  Ultralytics predicts with `rect=True` by default, so a request for `320` feeds
  `192x320`; the holdout is therefore measured on the deployed fixed canvas
  (`--geometry fixed`).

Outcome: the input-geometry change is worth **+0.078 macro recall** and a 20x
gain in small-object recall, and it reproduces on two different starting points.
Road-domain training on nuImages adds only ~+0.005 macro recall and costs macro
F1, and it drives traffic-light predictions to **zero** (141 -> 0 true positives)
even though those head rows were never reset. The RKNN INT8 gate therefore does
not pass, and no board deployment follows from this round.

## License

No project-wide license has been selected. Do not assume permission to reuse code or data beyond GitHub's normal viewing rights. Third-party datasets, model weights, and toolkits retain their own terms.
