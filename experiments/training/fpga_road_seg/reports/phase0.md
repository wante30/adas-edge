# Phase 0: source and repository audit

Date: 2026-11-12 (Asia/Shanghai). Issue: https://github.com/wante30/adas-edge/issues/15

## Confirmed repository boundary

Base `origin/main`: `9b396e8b346bfbbfa9856950937e1cd3f54516f4` (verified with GitHub API).
The user's `live/stage2` checkout is dirty; this work uses a separate worktree.
Open PRs #2–#8, #11–#14 include unmerged lane telemetry, MTL/Seg-256,
FAST-Q4, Tesla BEV and signal changes. No cherry-pick or stable service change.
Read README, deployment guide, model training report, existing AI bridge frame
selection, and latest signal branch Q4-native coordinate conventions.
Local `bench_seg_experiments.py` compares Seg-256 to MTL predictions, not to GT;
that agreement is not segmentation accuracy. No teammate MobileNet weights used.
PG2L200H CNN RTL, timing/resource reports and operator capabilities are absent.
Existing AVM RTL in this repository does not establish CNN accelerator support.

## Pinned software reference

TwinLiteNet+: https://github.com/chequanghuy/TwinLiteNetPlus
commit `90f1b8695ae311d5123b05f8534b2e11e42499d2`.
Vendored model/config are exact upstream files except the package-relative config
import. MIT license retained. Small: p=2, q=3, channels=8/16/32/64/128;
Nano: p=1, q=1, channels=4/8/16/32/64.
Both have depthwise/pointwise and dilation up to 16, PReLU, ConvTranspose,
CAAM attention with MatMul/Linear/Softmax, pooling, Add, Concat, reshape and transpose.
These are software references, not a claim of PG2L200H compatibility.

Upstream README reports Small 90.6% drivable mIoU / 29.3% lane IoU, 0.12M
parameters / 1.40G FLOPs; Nano 87.3% / 23.3%, 0.03M / 0.57G. These are
**upstream results**, not our reproduction. Official Small and Nano weights were
downloaded to ignored local output, loaded strictly, and run on six upstream
demo images. Torch/ONNX max logit errors were 0.0001374 (Small) and 0.0008965
(Nano). This is export parity, not accuracy evaluation; no GT accompanies the demos.
Upstream loader uses RGB uint8 converted to float /255 during training,
640x384 letterboxing and a 12-pixel top/bottom output crop against 640x360 labels.
Its original loss/loader is CUDA/path coupled. Our portable harness preserves the
unmodified network but is not an exact reproduction of that training recipe.
384x216 and 512x288 need explicit shape probes: CAAM's 2x4 bins impose additional
divisibility beyond a normal encoder stride. Never silently reshape/resize it.

## Label evidence and policy

BDD toolkit pinned at `9ac17c6c7c51d2fc83065fccd707cd5b1882a293`:

- https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst
- https://github.com/bdd100k/bdd100k/blob/master/bdd100k/configs/lane_mark.toml
- https://github.com/bdd100k/bdd100k/blob/master/bdd100k/label/to_mask.py
- https://github.com/bdd100k/bdd100k/blob/master/bdd100k/label/label.py

Drivable PNG: direct=0, alternative=1, background=2, ignore=255. Lane raw JSON:
`poly2d.vertices/types/closed`, attributes `laneStyle=solid|dashed`, category
crosswalk/road curb/single white/etc (older exports may prefix `lane/`).
Do not assume Bezier control points are ordinary polygon vertices.
The toolkit encodes lane PNG as category + (direction << 5) + (style << 4).
Documentation bit numbering and newer converter differ; converter inputs must
explicitly specify their schema rather than guessing from values.

BDD crosswalk belongs to the **lane marking** task. Its sparse curves are not
dense painted-stripe/whole-crosswalk ground truth. Four-mask conversion therefore
sets crosswalk validity to zero for every BDD sample. Lane styles are valid only
where a lane task was explicitly annotated; unknown styles are ignored locally.
Drivable and lane channels may overlap. Road curb and crosswalk are excluded
from solid/dashed road lane targets. Line widths are weak raster labels, not
measured paint widths; retain native geometry and measure 4/8/12px sensitivity.

Mapillary Vistas is a supplemental **user-supplied licensed data** entry point:
https://www.mapillary.com/dataset/vistas . Read the downloaded config's label names
instead of hardcoding version-dependent IDs. Dense `marking--crosswalk-zebra`
(and version-specific crosswalk labels explicitly mapped by the user) can supervise
crosswalk; generic lane marking cannot supervise solid/dashed. Respect the exact
downloaded license; no assumption of unrestricted commercial redistribution.

## Available compute and data

RTX 4060 Laptop 8GB, Python 3.10, torch 2.5.1+cu121, ONNX 1.17.0,
ONNX Runtime 1.23.2; CUDA available. Suitable for measured smoke/throughput runs.
The previous MTL training record in PR #4 (`research/b05n_p2_mtl/README.md`)
confirms a real BDD100K segmentation run on AutoDL using
`/root/autodl-tmp/bdd100k`: 100k 1280x720 images, lane/drivable JSON, and
rendered `label_mtl/lane`, `label_mtl/drivable`, plus split indices. Its old
lane renderer used background 255 and encoded category/style/direction values.
This is a confirmed historical source, **not proof that the rented server or its
data still exists or is reachable**. The current Windows machine has validation
images/boxes, historical predictions/checkpoints, and a small detection-only
BDD subset; a complete verified segmentation train/val/test set has not yet been
located locally. Previous historical IoUs belong to that prior sparse-lane task
and are not results for dense crosswalk or this four-mask task.

The adapter now accepts old raw `objects`/`style`/flat `[x,y,type]` JSON, old
rendered lane PNG as positive-only supervision (255 background is unknown), and
drivable polygons. Existing BDD test/holdout frames must stay test-only and out
of PTQ calibration. The BDD lane PNG alone cannot supervise background negatives.
Mapillary dense crosswalk data remains user-supplied and unverified. Thus no new
real-data training or road accuracy is claimed until source bytes, task coverage,
grouped splits and label semantics are validated.

Measured synthetic-tensor GPU throughput suggests 70k samples x 50 epochs would
take about 0.95h for Candidate 320x192 and 3.83h for TwinLiteNet+ Small 320x192
on the local RTX 4060 Laptop under resident-tensor microbenchmark conditions.
These are optimistic lower-bound estimates excluding image decode, augmentation,
validation, checkpointing and data transfer. A 4090 rental is not justified
before the old annotated dataset is recovered and real loader throughput measured.

## Evidence categories

`已运行通过`: executable tests and reports produced by this work.
`需要真实数据/GPU`: real-data learning/evaluation (GPU itself is available).
`等待 FPGA 队友确认`: operator support, memory/clock/protocol/RTL weights.
`板端实测`: none for this new model. Issue's 5.925/14.295 AI FPS and 22.7 Qt FPS
are historical numbers only, not a new measurement.
