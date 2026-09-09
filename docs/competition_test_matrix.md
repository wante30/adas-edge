# Competition test matrix

| Area | Current evidence | Level |
|---|---|---|
| PCIe v1 golden bytes, malformed headers, version/size/CRC | `tests/test_competition_closeout.py` | SOFTWARE UNIT TEST |
| BGR/RGB/NV12/RGB565 adapters | synthetic color vectors | SOFTWARE UNIT TEST |
| bbox clipping, empty/multiple detections | injected detector tests | SOFTWARE UNIT TEST |
| red/yellow/green/unknown, hazards, lane, navigation, speed | deterministic policy tests | SOFTWARE UNIT TEST |
| FPGA->RK3566->FPGA metadata round-trip | mock pipeline/transport | SOFTWARE E2E MOCK |
| four-stream identity and drop-oldest accounting | round-robin transport | SOFTWARE UNIT TEST |
| temporal traffic-light stabilizer | synthetic state sequence | SOFTWARE UNIT TEST |
| installed package import | editable install plus pytest | SOFTWARE UNIT TEST |
| RKNN inference | existing single-image smoke records | RK3566 BOARD TEST |
| physical endpoint/DMA, capture, OSD/HDMI, four cameras | not in repo | FPGA/PCIe HARDWARE TEST: BLOCKED |
