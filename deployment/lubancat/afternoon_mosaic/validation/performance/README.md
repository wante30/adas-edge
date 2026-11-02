# Evidence scope

`baseline/summary2.json` contains the corrected A/B/C/D baseline used in the report. `baseline/summary.json` is an invalid preliminary diagnostic (all-task sampler overhead; D also failed QML import) and is not used for conclusions.

Raw JSONL, process logs, cProfile data, failure records and original XWD screenshots are preserved in the separate real-PCIe evidence archive. JSON summaries and selected PNG screenshots are committed here. Runtime packaging excludes live sessions and raw recordings.

The board wall clock reads 2024. All performance intervals, source ages and frame rates use monotonic clocks; the task date is 2026-11-02. `source_fps` is accepted input, not Qt scene-graph FPS. CPU percentages in process/thread tables use one core = 100%; whole-system CPU uses all four cores = 100%.

`final_acceptance` completed menu/restart checks but its subsequent long run was aborted after a screenshot-close click hit the source toolbar. It is excluded from final performance acceptance. The corrected full run is `final610`. Priority-trial measurements interrupted by PCIe failure and all-layer-off/cProfile trials are diagnostic only.

`final610` is a complete 612s functional run with NPU governor restored but actual frequency held at 900MHz after a diagnostic. It is valid as a separately labeled frequency trial, not the equal-frequency A/B. `final600` runs from a freshly extracted runtime with actual NPU frequency reset to 600MHz and is the final equal-frequency acceptance.
