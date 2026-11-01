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

Stage 6 checkpoint: existing MTL lane-category masks and paired target-lamp crops are processed by a latest-slot worker <=2 Hz; stale/context-invalid semantics disappear. Three semantic tests passed, including missing mosaic masks and no semantic-only critical event.

Stage 7 checkpoint: one critical-only nonblocking controller checks source, identity, freshness, context, path, vehicle availability and authorization, with per-event decision/request/ACK/rejection/timeout logs. DRY_RUN vehicle state is explicitly virtual; HMI hardware HUD sends are suppressed. Synthetic pulse helper requires measured crossing evidence and produces one 300 ms simulated pulse per event.

Stage 8 checkpoint: recovered Windows ByteTrack-Lite now accepts completion timestamps and expires by 600 ms, retains two-stage/family isolation, and publishes measured boxes plus bounded visual velocities. Native Tesla predicts at render time for <=250 ms, interpolates at frame cadence, fades birth/death over 180/350 ms, expires at 600 ms, always filters Q1, and expires hazard color separately. Native combined build/replay validation follows in stage 9.

Stage 9 checkpoint: 610.672 s actual replay, alternating ~60 s ROAD/MOSAIC; PR6 completion counting gives MTL-FAST 6.6475 FPS, FAST-MOSAIC 14.0374 FPS, Qt overall 28.0948 FPS. 39 tests passed locally and on RK3566; native association test passed. Zero runtime errors, duplicate completions, event/box identity-color mismatches or invalid control requests. 246 critical decisions / 73 DRY_RUN requests and ACKs, no timeouts. UI confirmation/clear p95 204.54/204.40 ms. Raw archive SHA matches frozen log digests. Synthetic numerical and native UI evidence is separate. Hardware PCIe/LVT1/brake execution remains unavailable.
