# Road-domain pilot (round 2): results and gate decision

Every number below is measured on the **frozen 1000-image BDD100K holdout**,
never used for training, with the historical matching protocol (per class,
one-to-one, conf >= 0.25, IoU >= 0.5). Artifacts:

- per-model reports and `matrix.json` / `matrix.md`:
  `D:/Datasets/adas_edge_training/road_pilot/matrix_fixed/`
- checkpoints: `road-ft-320.pt`, `road-ft-512x288.pt` in the same directory
- training recipes and provenance: `road-ft-320.json`, `road-ft-512x288.json`

## Matrix

| class R / F1 | BASE-320 | BASE-416 | BASE-512x288 | Road-FT-320 | Road-FT-512x288 |
| --- | --- | --- | --- | --- | --- |
| car | 0.234 / 0.356 | 0.299 / 0.429 | 0.336 / 0.468 | 0.258 / 0.372 | 0.338 / 0.461 |
| truck | 0.140 / 0.194 | 0.205 / 0.263 | 0.219 / 0.273 | 0.156 / 0.203 | 0.215 / 0.277 |
| bus | 0.177 / 0.229 | 0.221 / 0.275 | 0.260 / 0.310 | 0.122 / 0.159 | 0.193 / 0.261 |
| person | 0.098 / 0.170 | 0.166 / 0.265 | 0.216 / 0.333 | 0.158 / 0.237 | 0.266 / 0.363 |
| motorcycle | 0.037 / 0.066 | 0.056 / 0.100 | 0.111 / 0.182 | 0.037 / 0.048 | **0.222** / 0.247 |
| bicycle | 0.011 / 0.021 | 0.053 / 0.100 | 0.063 / 0.114 | 0.011 / 0.020 | 0.053 / 0.093 |
| traffic light | 0.012 / 0.022 | 0.032 / 0.059 | 0.052 / 0.092 | **0.000** / 0.000 | **0.000** / 0.000 |
| **macro R / F1** | **0.101** / 0.151 | **0.147** / 0.213 | **0.180** / 0.253 | **0.106** / 0.148 | **0.184** / 0.243 |

| | BASE-320 | BASE-416 | BASE-512x288 | Road-FT-320 | Road-FT-512x288 |
| --- | --- | --- | --- | --- | --- |
| small-object recall | 0.001 | 0.010 | 0.023 | 0.006 | **0.024** |
| medium-object recall | 0.193 | 0.308 | 0.382 | 0.245 | 0.377 |
| large-object recall | 0.704 | 0.787 | 0.804 | 0.701 | 0.786 |
| traffic light TP / FP / FN | 32 / 99 / 2702 | 87 / 152 / 2647 | 141 / 177 / 2593 | **0 / 0 / 2734** | **0 / 0 / 2734** |
| network input | 320x320 | 416x416 | 288x512 | 320x320 | 288x512 |
| pixel budget | 102,400 | 173,056 | 147,456 | 102,400 | 147,456 |

## Reading the four cells

**1. Input geometry is worth ~+0.078 macro recall, and it repeats exactly.**
BASE-320 -> BASE-512x288 gains +0.078. Road-FT-320 -> Road-FT-512x288 gains
+0.078. The same delta on two different starting points means the effect belongs
to the pixel budget, not to the weights. Small-object recall moves 0.001 -> 0.023
and 0.006 -> 0.024, i.e. a 20-24x relative gain on the bucket that was previously
dead. This is the round's one unambiguous win.

**2. Road-domain training on nuImages adds almost nothing to macro recall.**
BASE-320 -> Road-FT-320: 0.101 -> 0.106 (+0.005, about one false positive per
class per 1000 images - noise at this sample size).
BASE-512x288 -> Road-FT-512x288: 0.180 -> 0.184 (+0.004).
Macro **F1 gets worse** in both cases (0.151 -> 0.148 and 0.253 -> 0.243): the
road-domain models trade precision for recall. Person FP goes 122 -> 302 and
motorcycle FP 6 -> 31 at 512x288.

**3. The one real data-domain signal is motorcycle, and it is strong.**
At 512x288, motorcycle recall goes 0.111 (COCO weights) -> 0.222 (road-domain
weights), TP 6 -> 12 with FP 6 -> 31. nuImages carries far more motorcycles than
the old COCO road subset did, and that transfers to BDD. So road-domain data does
help on classes where the source distribution was previously starved - it is just
too narrow a training set (nuImages, 4,500 frames, 6 classes) to move the macro
average.

**4. Traffic light is not merely untouched - it is destroyed.**
Both road-domain models emit **zero** traffic-light boxes at conf >= 0.25 across
1000 images, against 141 true positives for BASE-512x288 on the same frames. That
is a complete collapse, not a small regression.

This overturns the assumption behind head option A. Keeping the 80-class head and
never writing a traffic-light label does *not* preserve the class, because
fine-tuning moves the shared backbone, neck and class trunk out from under the
untrained output row, and the row goes silent. "The head weights were not reset"
is not the same guarantee as "the class still works".

## Gate decision

The stated gate was: only proceed to RKNN INT8 for RK3566 if road-domain training
plus a 16:9 input actually win.

**The gate does not pass, for one narrow but blocking reason: traffic light is a
required ADAS class and it now outputs nothing.** Shipping Road-FT-512x288 to the
board would trade a meaningful small-object gain for the loss of the single class
the earlier rounds already flagged as weakest (`TRAFFIC_LIGHT_STATE_UNKNOWN`).

Do **not** run `PT -> ONNX -> RKNN INT8 -> RK3566` on Road-FT-512x288 yet, and do
not re-run RKNN latency / 4x6 / 4x8 measurements, because the model they would
measure should not be the deployed model.

Two separate things are true and should be acted on apart:

- **Free win, no model change:** the board's `letterbox_bgr` pads a 16:9 frame
  into a *square* canvas, so 320x320 wastes ~40% of its pixels on grey. Feeding a
  fixed 320x192 / 416x256 canvas keeps the content and drops the pixel budget by
  40%. Accuracy survives that switch - BASE-320 measured 0.101 macro recall at
  320x320 against 0.099 at Ultralytics' minimum rectangle - so this is efficiency
  for free. The rectangular contract from this round already supports it.
- **Blocking fix, needs data or a different recipe:** traffic light has to survive
  fine-tuning. Options, in the order worth trying: include a traffic-light source
  in the road-domain mix (nuImages has no such category); freeze the backbone and
  neck and train only the head; or make "traffic-light recall does not regress on
  the frozen holdout" a hard acceptance gate for any future fine-tune. Any of
  these can be tested with the harness built this round.

## Caveats that limit how much weight these numbers carry

- **One seed per pilot.** A +0.005 macro-recall difference is not distinguishable
  from noise with a single run and 1000 images. Only the geometry effect (+0.078,
  reproduced twice) and the traffic-light collapse are large enough to act on.
- **nuImages is multi-view.** Only shards 0-4 were identifiable, and they are
  entirely `CAM_BACK`, so they were skipped; the remaining shards carry no camera
  tag. The training mix is front-facing plus unknown cameras, which is a weaker
  match to a front-facing deployment than BDD100K train would have been.
- **Wall-clock latency is not comparable across models.** BASE-320 and
  Road-FT-320 use the identical 320x320 canvas and the identical architecture, yet
  measured 24.94 ms and 18.52 ms - a 26% spread from host load alone. Read the
  throughput column as "proportional to pixel budget", not as a per-model result.
- **BDD100K train was never obtained.** This round trained on nuImages because
  every reachable BDD100K source turned out to be val-only or labels-free. The
  user's first choice, a real BDD100K-train road-domain fine-tune, remains
  untested and needs a licensed copy from the official portal or an equivalent
  credentialed mirror.
- **`mosaic=0` in both pilots**, forced by the need to keep the geometry
  comparison clean. These numbers are not the best this recipe can reach; they are
  the cleanest comparison this round could make.
