# Competition software performance

This document records measurements that are actually available in the checkout. It keeps hardware claims separate from software-only checks.

## Available measurements

| Measurement | Input/model | Result | Scope |
|---|---|---:|---|
| RK3566 single-image smoke | split INT8 320 | about 24.14 ms | RKNN load, one image after warmups, decode success; rough latency only |
| RK3566 single-image smoke | split INT8 416 | about 38.35 ms | RKNN load, one image after warmups, decode success; rough latency only |
| Historical NPU-only profile | raw-head INT8 640 | 70.692 ms mean | `rknn.inference` stage in the historical pipeline profile; excludes preprocess/postprocess |
| Historical split smoke | split INT8 640 | 96.206 ms mean | standalone `rknn.inference()` loop; excludes preprocess/postprocess; three warmups recorded |
| Traffic-light HSV adapter | host Python | UNKNOWN | no isolated timing measurement recorded |
| ADAS policy | host Python | UNKNOWN | no isolated timing measurement recorded |
| Frame/result serialization | host Python | UNKNOWN | no isolated timing measurement recorded |
| Software replay on RK3566 | 30-frame existing video | NOT COMPLETED | SSH invocation produced no remote process or output directory; no result is claimed |

The 70.692 ms and 96.206 ms values are not interchangeable: they use different 640 artifacts and output contracts. The former is the historical six-output raw-head pipeline stage; the latter is the split-output model's standalone Lite inference loop. Runtime/toolkit context is RKNN 2.3.2-era on RK3566 for both, but the historical 640 profile does not record exact warmup count, governor state or exact model filename. See `docs/competition_640_benchmark_note.md`.

## Interpretation

The detector, HSV adapter, ADAS policy, metadata protocol and four-stream mock are implemented and covered by tests. CPU overhead for the new scene/serialization stages is still `UNKNOWN`; it must be measured on the target software stack before being used in a budget. The 320/416 board smoke results establish load/inference/decode viability only. They do not establish accuracy, empty-frame behavior on approved road clips, or end-to-end PCIe latency.

## Remaining measurement

After an approved road-video subset is available, run the same clips through 320 and 416 and record per-stage timings, empty-frame ratio, class counts and ADAS actions. A successful RK3566 replay should be stored under `outputs/competition_software_replay_rk3566/` and labeled `SOFTWARE PIPELINE TEST ON RK3566`. Until that artifact exists, video replay status remains `NOT COMPLETED`.
