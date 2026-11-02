# P1 full-mode bounded optimization experiments

All production functions, weights, model resolution, CRC and real PCIe retained. Profiling runs used no-send authorization, NPU 600 MHz. The raw windows are retained in /home/cat/adas_issue9_20261102/src/evidence.

1. Compared default Tesla animation with hidden-scene stop + 33ms animation. Before: full Mosaic AI 13.7006, weighted Qt 24.2878; MTL AI 6.1473, weighted Qt 24.9910. After: full Mosaic AI 13.9384, sampled Qt 23.71; MTL AI 6.0826, sampled Qt 25.86. Gains are inconsistent. Both animation changes were discarded; only Lite explicitly stops animation.
2. Raised only GUI/render scheduling to nice -5 for 60 seconds, then restored original nice values. MTL AI 6.0423, sampled Qt 27.42 and texture 25.46, GUI update 13.86ms. GUI gained while full-system throughput targets remained unmet; not enabled by default.
3. Diagnostic inference-thread retuning from its measured original nice -19 to -3 with GUI -5: MTL AI dropped to 5.141 while Qt rose to 29.74. Rejected, all priorities restored. A subsequent Mosaic trial changed model during the sample (FAST-MOSAIC plus Lite); that mixed window is INVALID for performance comparison and is not used as a Mosaic result. No source or assertion was changed to hide it.

Current full-mode targets Qt>=28, MTL>=6.5 and Mosaic>=13.8 are not all achieved. The whole-system CPU contention remains: capture 84-94 percent of one core; OCR 52-59; render 23-32; source 21-23; inference 32-48, plus kernel/GUI/driver overhead. Changing priorities trades inference for UI throughput. No unproven scheduling policy is shipped as an improvement. Lite is the measured accepted performance mode; complete ADAS remains available.

See perf_after_summary.json, gui_nice5_summary.json and gui_infer_mtl_summary.json for distributions/stage/runqueue evidence. The final 600-second run is required to report final build behavior separately.
