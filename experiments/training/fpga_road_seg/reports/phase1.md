# Phase 1 evidence: runnable training baseline

All runs used the generated 24-image synthetic fixture on the RTX 4060 Laptop
GPU (16 train, 4 validation, 4 test). They verify the scripts and metrics only;
the fixture is not road imagery.

| Model | Run | Train loss | Validation loss | Synthetic validation observation |
|---|---:|---:|---:|---|
| TwinLiteNet+ Small, original two heads | 2 epochs | 1.54560 -> 1.43225 | 1.43862 -> 1.42066 | Drivable IoU 0.4173; lane IoU 0.0 |
| RoadCandidate, four heads | 12 epochs | 1.55949 -> 1.35686 | best/final 1.44903 | Drivable IoU 0.5338; solid IoU 0; dashed IoU 0.0070; crosswalk IoU 0 |

IoUs above are calculated against the **synthetic fixture** and are not ADAS
accuracy. The short candidate run did not learn solid/crosswalk positives.
An additional 1-epoch CUDA smoke after the historical BDD compatibility change
completed at train loss 1.55949 and validation loss 1.53230.

An actual local BDD validation image was also passed through the synthetic-trained
candidate for the four-color overlay. The image is real, but model and mask
predictions are synthetic-smoke evidence, not a trained BDD result. The committed
PNG and JSON provenance label this distinction.

The training CLI performs data loading, masked BCE+Dice, validation, checkpoint
writing and JSON reporting. Conversion/split leakage, unknown BDD styles, ignored
pixels, missing categories and zero-positive non-synthetic channels fail closed.
Full real train/validation/test remains blocked on recovery and audit of the old
AutoDL BDD source and a dense crosswalk source.
