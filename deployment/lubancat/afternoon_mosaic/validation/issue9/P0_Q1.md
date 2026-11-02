# Q1 localization

Capture boot: 8161ae7d-7d47-49f2-baa0-ed293d3d117d. Temporarily stopped the capture service, ran the unchanged capture executable with `--frames 60 --fps60 --out .../raw --publish-dir .../published --profile`, then restarted the service successfully.

60 FPGA-CRC-validated cached DMA snapshots contain an all-zero Q1. Q2/Q3/Q4 remain populated; per-frame change/nonzero/CRC data is in quadrant_diagnosis.json. Final frame 43734 generation 8 has CRC 8494728b; the published PGVF payload is byte-identical to the corresponding saved DMA snapshot, with valid header and CRC. `pgv_snapshot_frame` copies the leased DMA slot to cached memory without quadrant processing; publisher writes that payload unchanged. Source identity and mapping are 1280x720 RGB565, Q1 top-left/Q2 top-right/Q3 bottom-left/Q4 bottom-right.

The first observed zero stage is the CRC-validated DMA payload, before publisher/PGVF/Qt. This rules out application crop, publisher offset, and GUI clearing as the origin. It does NOT distinguish source camera/input-valid from the FPGA compositor/routing/zero-fill logic. No live FPGA internal input-valid debug registers or verified matching RTL build are available in this test; no unsupported RTL/driver patch was made. Minimal hardware next step: inspect Q1 input clock/de/valid and compositor quadrant-select/zero-fill at the DMA pixel producer, correlate output frame_id 43734 and CRC, check top-left mux routing. Never replace Q1 with another channel.

Capture executable SHA256: d559f87966423ef31dfa317ab2500e342b99fc2939d97990f1047078671ad6ab. Original capture/driver remain unchanged. Raw 60-frame snapshots and published final frame retained at /home/cat/adas_issue9_20261102/q1; analysis can be reproduced with tools/diagnose_quadrants.py.
