# P1 Lite lifecycle and real PCIe A/B

FAST-MOSAIC-LITE is a separate formal menu profile using the identical model, 512x288 input, C API and mosaic adapter as FAST-MOSAIC. The backend stops and reaps OCR/IPC and the semantics worker; bypasses risk, dynamic HUD generation and automatic brake decisions. Qt stops the Tesla animation timer and auto HUD sender. Detection identities and native boxes remain. Switching out recreates OCR and semantics, restores the previous control enabled/authorized state, and uses a new HMI epoch.

Three real-menu windows (60+ seconds each) passed in lite_ab_issue9_acceptance.json: full Mosaic -> Lite -> MTL-FAST. Lite had no OCR PID or active semantics, control enabled=false; full tasks and OCR PID restored on exit. Source is real PGV_LIVE Q4-forward, no synthetic input. Performance sends unauthorized throughout. Full Python suite: 62 tests passed on RK3566.

Raw measurements remain on board in /home/cat/adas_issue9_20261102/src/evidence/lite_ab_{0,1,2}_system.jsonl. Summary is lite_ab_summary.json, including weighted Qt FPS and counter-based texture FPS (do not equate render FPS with new video frames). Full Mosaic AI completion 13.7006 FPS, Lite 15.6546 FPS, MTL 6.1473 FPS. Sample mean Qt 24.17/32.17/24.93; sample mean texture 22.79/30.10/23.94. CPU 95.37/88.84/92.65 percent. Full-system performance targets still require improvement.

P0 real AI evidence was also collected before no-send profiling: 29 real Q4 critical requests and 29 FPGA accepted ACKs. p0_ai_control.json correlates event request_id to the hardware journal. This is distinct from the separately labeled three manual probes.
