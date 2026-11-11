# FPGA road segmentation candidate (Issue #15)

Isolated Python experiment. No runtime, service, PCIe, HUD, Tesla BEV or teammate
MobileNet files are modified. Source audit: [reports/phase0.md](reports/phase0.md).
Run all commands **from the repository root**. Examples use PowerShell.

## Setup and evidence labels

Python 3.10, torch 2.5.1+cu121 tested on RTX 4060 Laptop. Use your CUDA-specific
torch installation, then install `requirements.txt` here. `candidate` is a new
8,508-parameter model, not a pretrained TwinLiteNet. `twin-small` and `twin-nano`
preserve upstream two-head architecture; `twin-four` is a four-logit extension.
The original network is preserved, but the portable masked BCE/Dice harness,
training schedule and 320x192 preprocessing are **not the original paper recipe**.
`sigmoid(fg-bg)` matches each original two-class softmax exactly.

States: **已运行通过** = recorded software checks; **需要真实数据/GPU** = actual
labelled train/validation still required; **等待 FPGA 队友确认** = hardware contract;
**板端实测** = no results for this candidate yet. Synthetic loss/IoU are never road
accuracy. Historical FPS in Issue #15 are not this model's performance.

## Real data and frozen manifests

Download data yourself under its license. Do not commit data/weights. Create a
JSON index; paths are relative to that index (absolute paths also work). Require
explicit split, original video/sequence group ID, and license provenance for every
sample. Do not split neighbouring frames by random filenames. Existing BDD
holdout images remain test-only and cannot be PTQ calibration samples.

```json
[
  {
    "id": "bdd-video1-frame1",
    "source": "bdd100k",
    "image": "bdd/images/train/video1.jpg",
    "label_json": "bdd/labels/train/video1.json",
    "split": "train",
    "group_id": "bdd-video1",
    "license": "BDD100K accepted data terms; record version/location"
  },
  {
    "id": "vistas-001",
    "source": "mapillary_vistas",
    "image": "vistas/training/images/001.jpg",
    "label_png": "vistas/training/labels/001.png",
    "config": "vistas/config_v2.0.json",
    "mapping": {"crosswalk": ["marking--crosswalk-zebra"]},
    "split": "train",
    "group_id": "vistas-verified-sequence001",
    "license": "record exact downloaded Vistas license and allowed use"
  }
]
```

This is a schema example, not an existing usable dataset. Omit unavailable task
paths; never fabricate an empty lane JSON for an unannotated frame. An explicitly
annotated empty lane task is valid negative supervision. Raw drivable PNG uses
0=direct, 1=alternative, 2=background, 255=ignore, not a thresholded binary PNG.
Raw lane `poly2d` L/C paths accept modern `vertices/types` objects and the older
flat `[x,y,type]` serialization, including frame `objects` and
`attributes.style`. An optional `label_json` supplies both drivable polygons and
lane objects. The previous MTL lane PNG encoding is accepted as `lane_png`;
because its background is 255, it contributes positive-only pixels and does not
invent negative supervision. Unsupported RLE/schema fails closed.
Line paths have default native width 8px and resize with the image; dataset
coordinates are never interpreted in network-pixel space. Use 4/8/12 native px
conversion runs and compare thin-line F1/recall before selecting width. Rasterize
curves, not their Bezier control polygons. No automatic filling of dashed gaps.

BDD crosswalk sparse curves have **zero crosswalk validity**. Vistas config label
names resolve IDs dynamically; unmapped channels have zero validity, void pixels
are ignored. Mapping road into drivable is an explicit semantic proxy choice,
not proof that road is safe to drive. Vistas generic line markings cannot infer
solid/dashed. Dense crosswalk from Vistas and BDD drivable/lane can train jointly.
Overlapping masks stay independent. Visualization precedence is drivable, solid,
dashed, crosswalk; it does not alter labels or loss.

```powershell
python -m experiments.training.fpga_road_seg.dataset_adapter --index D:/data/road/index.json --output outputs/road-converted-w8 --line-width 8
python -m experiments.training.fpga_road_seg.train --manifest outputs/road-converted-w8/manifest.jsonl --config experiments/training/fpga_road_seg/configs/twin_small.json --output outputs/twin-small-real
python -m experiments.training.fpga_road_seg.train --manifest outputs/road-converted-w8/manifest.jsonl --config experiments/training/fpga_road_seg/configs/candidate.json --output outputs/candidate-real
python -m experiments.training.fpga_road_seg.eval --checkpoint outputs/candidate-real/best.pt --manifest outputs/road-converted-w8/manifest.jsonl --split test --output outputs/test-metrics.json
python -m experiments.training.fpga_road_seg.infer --checkpoint outputs/candidate-real/best.pt --image D:/data/road/example.jpg --output outputs/example.png
```

Converted manifest and provenance include image/index hashes, fixed splits,
native raster width and per-channel positive/valid pixel counts. Full manifests
are validated for sequence leakage, repeated paths/IDs and duplicate image bytes.
Training refuses missing supervised channels or zero positive training labels.
Loss averages BCE (configurable positive class weights) plus masked Dice over
available channels. Metrics accumulate TP/FP/FN/TN globally over valid pixels;
undefined IoU/recall is JSON null. Crosswalk recall and solid/dashed F1 are included.
Padding is excluded from loss/metrics. Nearest label downsampling can erase very
thin labels; inspect masks and compare 320x192 versus 512x288 on genuine GT.

## Executable smoke check

```powershell
python -m experiments.training.fpga_road_seg.smoke_data --output outputs/road-fixture
python -m experiments.training.fpga_road_seg.train --manifest outputs/road-fixture/manifest.jsonl --config experiments/training/fpga_road_seg/configs/twin_small.json --output outputs/twin-small-smoke --epochs 2 --synthetic-smoke
python -m experiments.training.fpga_road_seg.train --manifest outputs/road-fixture/manifest.jsonl --config experiments/training/fpga_road_seg/configs/candidate.json --output outputs/candidate-smoke --epochs 12 --synthetic-smoke
python -m pytest experiments/training/fpga_road_seg/tests -q --basetemp outputs/road-pytest
python -m experiments.training.fpga_road_seg.profile --output outputs/road-shapes.json
```

Use fresh output directories; weights/large outputs stay ignored by Git. Original
Small/Nano reject 384x216 because CAAM patch dimensions are incompatible. Their
320x192, 512x288 and 640x384 probes pass. Candidate accepts all four sizes.
Profiles count Conv/ConvTranspose/Linear/MatMul MACs, FLOPs=2*MACs; exclude
activation/pooling/elementwise costs. They cannot be compared unqualified to the
upstream README's 1.40G FLOPs with unspecified arithmetic convention.

## Export, benchmark and offline Q4 mock

FP32 export, train-only static QDQ calibration, layer inventories and comparison:

```powershell
python -m experiments.training.fpga_road_seg.export --checkpoint outputs/candidate-real/best.pt --manifest outputs/road-converted-w8/manifest.jsonl --output outputs/candidate-export --calibration-count 256
python -m experiments.training.fpga_road_seg.benchmark --device cuda --output outputs/gpu-throughput.json
```

Synthetic checkpoints/manifests require the explicit `--synthetic-smoke` flag.
QDQ is a Conv-only quantization experiment: Add/Resize/I/O remain float32 and
the graph is not a PG2L200H artifact. The report and calibration manifest record
operator/tensor shapes, hashes, scales, zero-points and illustrative bias/requant
contracts.

`offline_q4` takes an explicit `q4_native` image or bottom-right Q4 crop from a
2x2 mosaic, a four-channel `.npy` mock mask and same-frame detection JSON. This
tool never imports the running FAST-Q4 service and cannot send data to hardware.
The draft protocol and coordinate requirements are documented in
[`docs/fpga_road_seg_handoff.md`](../../../docs/fpga_road_seg_handoff.md).
