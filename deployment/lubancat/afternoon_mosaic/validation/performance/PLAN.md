# Real PCIe performance development — 2026-11-02

Baseline: a9054ec21fb4a97637f06bbf0ece1d36f0f08c2e. Physical forward source Q4.

1. Preserve baseline executable and Git source; diagnose real PCIe C/B/A/D with identical sampling and monotonic clocks. Record thread run time and runqueue wait separately.
2. Implement FAST-Q4 with the existing independent BDD C ABI, packed Q4 crop, real model APPLIED handshake and unavailable MTL semantics. Keep default MTL-FAST.
3. Fix measured Qt/CPU/OCR hot paths in separate commits; compare identical full feature configurations.
4. Verify switching, coordinates, restart and controls without actuator sends; run >=600 s real PCIe acceptance across all three modes.
5. Package executable/source/restore scripts, raw evidence, screenshots, measured report and independent PR.

All performance sessions explicitly set ADAS_CONTROL_AUTHORIZED=0. Real telemetry and decision logging stay enabled. No RTL, driver or runtime changes.

Preflight: PCIe revision ff, capture_io and offline LVT1 reproduced before any performance edit. Saved board preflight evidence; forced reboot restored endpoint and capture (53.8 FPS initial sample). Original baseline executable saved under /home/cat/adas_perf_20261102/baseline.
