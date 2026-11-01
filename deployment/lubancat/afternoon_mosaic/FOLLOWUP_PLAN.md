# PR #6 follow-up execution

Base: d397af411d592447197fa2f194144dca52c765dd. Independent branch: feature/adas-risk-control-20261102.
Board development root: /home/cat/adas_risk_dev_20261102. PR #6 root and runtime archive remain recoverable.
User steering: no simulator UDP port; replay uses an in-process DRY_RUN adapter with labelled simulated acknowledgements. Hardware interfaces remain separate.

## Ordered checkpoints

1. Freeze PR #6 sources, manifest and runtime; establish isolated checkout/deployment and recovery.
2. Make FAST-MOSAIC an explicit selectable model with mode/epoch handshake and last-road-model restoration.
3. One event protocol/lifecycle drives colors, panel, BEV/HUD and control eligibility.
4. Bound clear/downgrade latency and associate by measured target identity, source and epoch.
5. Add lightweight front/pedestrian/two-wheeler/cut-in/oncoming/brake-approach rules and per-rule timings.
6. Consume existing MTL crosswalk semantics; latest-slot background semantics <=2 Hz; no invented mosaic segmentation.
7. Unified fresh-Q1 critical decisions, dry-run requests/ACK/clear and isolated crosswalk pulse.
8. Adapt existing Windows ByteTrack-Lite; render-only prediction/interpolation/fades.
9. Instrument scheduling and latency; compare same 1350-frame input and alternating mode schedule; >=600 seconds.
10. Package RK3566 runtime and scripts; publish a separate PR with actual and synthetic evidence clearly distinguished.

Each checkpoint has its own verification and Git commit. Preserve navigation 2 Hz, MTL segmentation, HUD/LVT1/BEV, existing model options and native libraries. Do not modify FPGA/PCIe/runtime or enable full RoadScene/HazardFusion.

## Performance acceptance

Reference: mosaic 14.534 FPS, road MTL 6.899 FPS, Qt 29.444 FPS.
Minimum: mosaic 13.8 FPS, road MTL 6.5 FPS, Qt 28 FPS, continuous >=600 seconds.
Report source/texture/scene/AI counters independently; log p95 rule times, guiBusy fraction, request spacing, frame write, RKNN, transport and clear/UI delays.

## Progress

- Stage 1: base commit verified against PR #6; independent Git worktree created. Board baseline remains running pending isolated deployment.

Stage 3 checkpoint: 22 Python tests, native association/source/expiry test, native build and 280 actual replay completions passed. Unified event receiver removes Qt latch and maps warning/critical to yellow/red; full scenario and latency acceptance follows.

Stage 4 checkpoint: shared per-camera measured IDs feed binary-box metadata and event association. Explicit IDs never fall back to a different target; seam detections cannot control. Epoch resets and stable IDs covered by tests. Native receiver change awaits the combined stage 8 build.

Stage 5 checkpoint: numerical measured-track rules replace the legacy asynchronous HazardEvents/BackgroundMotion engine entirely in this bridge. Six scenario tests cover static-near versus expansion, pedestrian/two-wheel, lateral entry, independent direction requirements and camera isolation. Per-rule switches/count/mean/p95 are reported. Paired lamp evidence is supplied by stage 6 worker.
