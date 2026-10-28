# Road-domain pilot (round 2)

Scope: answer one question. **Can genuine road-domain training plus a 16:9 input
raise small-object detection while keeping four-stream RK3566 speed usable?**

Round 1 is frozen and is *not* reopened: BASE-320 and BASE-416 stay as-is, the
1000-image BDD100K holdout is never trained on, and the failed FT-320/FT-416
checkpoints are kept only as negative results. No further COCO-subset training.

## Data: what is reachable and what is not

| source | status | evidence |
| --- | --- | --- |
| `bdd-data.berkeley.edu` portal | unreachable | TLS certificate hostname mismatch through the local proxy |
| `dl.cv.ethz.ch` official mirror | unreachable | DNS returns NXDOMAIN for `dl.cv.ethz.ch` locally; CONNECT tunnel then EOF |
| `bdd100k*` S3 buckets | unusable | `bdd-data-storage-release` 404, `bdd100k` (ap-south-1) 403, `bdd100k-data` access disabled |
| Hugging Face BDD100K mirrors | no usable train set | 55 datasets checked: images-only (`quandelagl/bdd-100k`, `Xing1210`, `Myszz`, `yiyi159/bdd100k-train`), captions (`minhbui/bdd5k`), FiftyONE val (`dgural`, `ShantyCam/bdd100k-det`), gated 401 (`xmu-xiaoma666-dataset/detection_bdd100k`, `keremberke/*`) |
| Kaggle | no credentials | `kaggle` CLI absent, no `kaggle.json` |
| **`finedet/nuimages`** | **used** | nuImages v1.0 keyframe images with COCO `[x,y,w,h]` boxes, train split 67,279 frames in 27 parquet shards |

So this round trains on **nuImages**, which was the stated second choice, not on
BDD100K. The consequence is recorded rather than hidden:

- **nuImages has no `traffic light` category.** Traffic-light rows in the head
  are therefore untouched by this pilot; the holdout traffic-light numbers
  measure "does road-domain training hurt traffic light", not an improvement.
- nuImages keyframes come from six cameras. Shards 0-4 were probed and are
  entirely `CAM_BACK`; those are excluded (`--skip-shards`) together with
  `--front-only`, which keeps `CAM_FRONT` and unnamed-camera frames.

Dataset builder: `tools/prepare_nuimages_road_subset.py`.

```
python tools/prepare_nuimages_road_subset.py \
  --output D:/Datasets/adas_edge_nuimages_road_5k \
  --limit 5000 --max-shards 10 --skip-shards 0,1,2,3,4 --front-only \
  --shard-cache D:/Datasets/_nuimages_shards \
  --holdout-names D:/Datasets/adas_edge_training/bdd_holdout_1000_names.txt
```

Balanced selection is rarest-class-first (bicycle -> bus -> motorcycle -> truck
-> person -> car) with hard floors, then a capped share of car/person-only
frames, so the "person 9000 / bicycle 200" skew cannot reappear. `--holdout-names`
makes the script abort if any frozen holdout filename ever appears.

## Head initialisation: option A in the pilot, option B for deployment

**Option A (used for both pilots).** Keep the pretrained 80-class head. The
nuImages labels carry the original COCO ids and `dataset.yaml` declares all 80
COCO names, so Ultralytics finds `nc == 80` and loads every head weight instead
of re-initialising. This is also what keeps Road-FT directly comparable with
BASE-320: same architecture, same head, only data and input geometry differ.

Confirmed in the run log rather than assumed - the trainer prints:

```
Detect [80, 16, None, [64, 128, 256]]
Transferred 499/499 items from pretrained weights
```

499 of 499 tensors transferred means no head row was left at its random
initialisation, which is exactly the failure mode round 1 had.

**Option B (`tools/transfer_coco_head.py`).** For the eventual seven-class RKNN
graph. A naive row copy into a fresh `nc=7` head does not work, and the reason is
architectural, not a shortcut:

```
c3 = max(ch[0], min(nc, 100))          # 80 for nc=80, but 64 for nc=7
cv3[b] = Sequential(
    Sequential(DWConv(c_in -> c3, 3), Conv(c3 -> c3, 1)),
    Sequential(DWConv(c3 -> c3, 3),   Conv(c3 -> c3, 1)),
    Conv2d(c3 -> nc, 1),
)
```

The class trunk width is derived from `nc`, so there are no rows to copy for the
inner tensors - the layouts disagree outright. The tool therefore **keeps the
80-wide pretrained trunk** and narrows only the terminal 1x1 convolution to the
seven stored rows (`person 0, bicycle 1, car 2, motorcycle 3, bus 5, truck 7,
traffic light 9`).

Measured on `yolo11n.pt` (`yolo11n-road7.pt`):

| check | result |
| --- | --- |
| class-trunk features, all three branches | `max_abs_diff = 0.0` (bit identical) |
| surviving 7 logits vs COCO logits | `max_abs_diff <= 5.7e-06`, `max_relative_diff <= 1.4e-07` |
| tensors changed | 6 (the three `cv3.<b>.2.weight/bias` pairs), 0 unexpected |
| real-image detections, same frame at 512x288 | identical to the COCO model (`car` 12, `truck` 1) |

So option B narrows the head without discarding the pretrained classifier, and it
is verified against the very logits it claims to preserve.

## Rectangular input: the contract that had to exist first

`input_size: int` was genuinely insufficient, so the contract was widened and
tested before any training:

| layer | change |
| --- | --- |
| `tools/profile_rknn_single_stream.py` | `normalize_input_shape()` (pure) + `validate_model_shape()` (policy: >=32, multiple of 32); `letterbox_bgr` builds a `(width, height)` canvas; `decode_yolo11_raw` uses independent `stride_x`/`stride_y` and asserts the grid divides the input |
| `deployment/rknn/export_onnx.py` | `--imgsz` accepts `512x288` |
| `competition_ai/detector.py` | `input_size` accepts `int or (width, height)` |
| `tools/evaluate_bdd_yolo.py` | `--imgsz` accepts `512x288`; records the shape the network really received |
| RKNN INT8 on RK3566 | still a Linux/WSL step, unchanged |

Ultralytics itself cannot *train* a two-element `imgsz`: `trainer.py` calls
`check_imgsz(..., max_dim=1)`, which silently collapses `(512, 288)` to `512`.
Rectangular training therefore uses `rect=True`, which shapes each batch to the
batch aspect ratio. Because every image in this subset is exactly 1600x900, the
batch shape is constant:

```
ceil([900/1600, 1] * 512 / 32) * 32 = [288, 512]
```

`rect=True` disables mosaic inside `YOLODataset`, so mosaic is disabled in
**both** pilots (`mosaic=0.0`) - otherwise the geometry comparison would silently
also change the augmentation policy.

### The "no wasted padding" premise needs correcting

The premise was that a 512x288 input beats 416x416 because it has fewer pixels
and wastes less letterbox padding. Measured on this pipeline, the first half is
right and the second half is not:

| requested `imgsz` | tensor actually fed to the network | pixels |
| --- | --- | --- |
| 320 | 192x320 | 61,440 |
| 416 | 256x416 | 106,496 |
| 512 | 288x512 | 147,456 |
| `(288, 512)` | 288x512 | 147,456 |

Ultralytics sets `rect=True` by default for prediction, so `predict` **already**
letterboxes to the minimum stride-aligned rectangle: a 16:9 road frame asked for
as "320" reaches the network as 192x320. There is no square padding waste in the
baselines to remove, and `512x288` costs **2.4x** the pixels of the actual
BASE-320 input and **1.38x** of BASE-416.

So the comparison that matters is a pixel-budget comparison on a 16:9 canvas, not
"square versus rectangular". This is also why `evaluate_bdd_yolo.py` now defaults
to `--geometry fixed` and records the observed input shape: a deployed RKNN graph
has one frozen canvas, and the holdout has to be measured on that same canvas.
The `minrect` mode stays available as a diagnostic.

The remaining real win for 16:9 is on the **board**, where
`tools/profile_rknn_single_stream.py` letterboxes to a fixed *square* canvas and
therefore pads a 16:9 frame with ~40% grey at 320x320. The rectangular contract
now lets the board run a fixed 320x192 / 416x256 / 512x288 canvas instead, which
is the same content for a fraction of the pixels. That change belongs to the
deployment step, and it is measured there, not here.

### Throughput measurement correction

`Results.speed` is not a usable latency source: it is a per-batch profiler reading
that comes out identical for every image and reported a bogus 3.4x penalty for
the rectangular shape (19.46 ms vs 5.73 ms). Wall-clock measurement with explicit
CUDA synchronisation (`tools/benchmark_yolo_shapes.py`, 60 timed frames each)
gives 320 = 17.19 ms, 416 = 15.24 ms, 512x288 = 15.38 ms - effectively flat,
because per-image decode/letterbox/NMS overhead dominates the CNN at these sizes.
The evaluator now reports wall-clock ms per image for the whole pass.

## Pilot recipe

Both pilots start from the pretrained `yolo11n.pt` and the same 5k road subset:

| pilot | imgsz | rect | train input | isolates |
| --- | --- | --- | --- | --- |
| Road-FT-320 | 320 | False | 320x320 | data domain, vs BASE-320 |
| Road-FT-512x288 | 512 | True | 512x288 | input geometry, vs Road-FT-320 |

Runner: `tools/run_road_pilot.py --data <dataset.yaml> --epochs 25 --batch 16`.

## Comparison matrix

`tools/run_road_matrix.py` evaluates every row on the frozen 1000-image holdout
with the historical protocol (per class, one-to-one, conf >= 0.25, IoU >= 0.5)
and reports per-class recall/F1, macro recall, small/medium/large GT recall and
latency.

| model | role | pairwise reading |
| --- | --- | --- |
| BASE-320 | current speed baseline | vs Road-FT-320 -> isolates data domain |
| BASE-416 | current quality baseline | - |
| BASE-512x288 | input-geometry diagnostic | vs BASE-320 -> isolates geometry, weights fixed |
| Road-FT-320 | road-domain adaptation | vs Road-FT-512x288 -> isolates geometry |
| Road-FT-512x288 | the candidate worth testing | - |

## Gate

RKNN INT8 conversion for RK3566 only happens if road-domain training plus the
16:9 input actually win on the holdout. Step 6 is therefore the decision point,
not step 7.

**Outcome: the gate does not pass.** Results, analysis and the two follow-ups that
should be split apart are in
[the results note](road_domain_pilot_results.md). Short version: the 16:9 / pixel
budget change is worth +0.078 macro recall and a 20x gain in small-object recall,
road-domain nuImages training adds ~+0.005 and costs macro F1, and both road-domain
models emit **zero** traffic-light boxes against 141 true positives for
BASE-512x288 - so the round's headline is that an untrained class does not survive
fine-tuning even when its head rows are never reset.

One design assumption from this round is now known to be wrong: preserving the
80-class head keeps the *weights*, but it does not keep the *behaviour* of a class
that never appears in the training labels. Any future fine-tune needs
"traffic-light recall does not regress on the frozen holdout" as a hard gate.
