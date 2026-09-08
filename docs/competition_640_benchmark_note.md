# 640 benchmark note

The 70.692 ms figure is the `npu_infer_ms` mean from the 300-frame pipeline profile and is measured on RK3566 with the historical raw-head INT8 model; the pipeline also measures decode, preprocess and CPU DFL/NMS separately. The 96.206 ms figure is a separate 30-run smoke benchmark of `yolo11n_rk3566_int8_split.rknn`, reporting only the Lite `inference()` call but using a different model artifact/run context. Their files, dates and output contracts differ, so they are not contradictory measurements of one identical binary.

Known metadata: both use RKNN 2.3.2-era runtime/tooling and RK3566, and B explicitly has 3 warmup calls. The A report does not record warmup count, governor state, or an exact model filename; those fields are **UNKNOWN**. A's row is stage timing from a video pipeline (its `npu_infer_ms` excludes preprocess and postprocess); B is a standalone `rknn.inference()` loop (also excludes preprocess and postprocess). Split vs raw-head is different: A is documented as raw-head six-output, B as split `boxes`/`scores` two-output.

For the current competition report, use the 70.692 ms value only when describing the historical raw-head 640 pipeline profile, and use 96.206 ms only for the split-output 640 artifact. Neither should be silently substituted for the other; both are **MEASURED ON RK3566**.
