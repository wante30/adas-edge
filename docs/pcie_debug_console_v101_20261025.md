# RK3566 PCIe V1.0.1 debug console

Protocol source: `deployment/lubancat/qt_gpu_probe/rk3566_pcie_protocol_v1.json`
and `RK3566_PCIe协议开发交接.md`.

## Status

| Area | Software | Test |
|---|---|---|
| HUD request/read | implemented | SIMULATION_PASS |
| SCENE shadow/time/commit | implemented | SIMULATION_PASS |
| NAV shadow/control/active | implemented | SIMULATION_PASS |
| ISP four tiles / field encode / query | implemented | SIMULATION_PASS |
| ALGO 64-bit fields / query | implemented | SIMULATION_PASS |
| DIAG / behavior / HSST read | implemented | SIMULATION_PASS |
| Legacy/video DMA | read-only status; writes blocked | BLOCKED |
| PIO BAR0 | not implemented | BLOCKED |
| Ethernet action mailbox | not present in this repo | BLOCKED |
| Real BAR1 windows | cannot open without PCIe | HARDWARE_NOT_TESTED |
| Board web deployment | SSH timed out at deployment time | HARDWARE_NOT_TESTED |

`HARDWARE_PASS` is intentionally not claimed.

## Service

Single worker; no second BAR1 writer:

```bash
python3 pcie_protocol_service.py \
  --simulation --port 38264 --web-host 127.0.0.1 --web-port 8080
```

Real hardware (when available and driver ownership is confirmed):

```bash
python3 pcie_protocol_service.py --port 38264 --web-host 127.0.0.1
```

LAN access explicitly:

```bash
python3 pcie_protocol_service.py --web-host 0.0.0.0 --token '<random secret>'
```

The browser must send `X-Token`. Do not expose an unauthenticated MMIO API.

The old `pcie_isp_service.py` is now a thin launcher for the same worker so an
old and new service cannot both own BAR1.

## UI

`pcie_debug_ui.html`, served by the service:

- HUD: full state composition, takeover, blink validity, accepted/applied/effective.
- SCENE: shadow, BCD time, commit sequence, three-stage readback, vector.
- NAV: direction/distance, apply, active, clear.
- ISP: four tile forms generated from the protocol JSON, all documented fields.
- ALGO: stage bits 56–59 and full parameter fields.
- DIAG: behavior, HSST priority source fields, video status, DMA read-only.
- RAW: every service-side MMIO operation and JSON report export.

The Qt FPGA CTRL page has a `WEB DEBUG` button opening
`http://127.0.0.1:8080/`. Existing Qt `PcieIspControl` UDP requests remain
compatible with `query`, `apply_isp`, `apply_algo`.

## Safety

- Writes are composed in the service; browser has no arbitrary register write API.
- SCENE/HUD/NAV/ISP/ALGO use their documented complete DWORD transactions.
- HUD vector `0x608=0x00011103`, release `0x00020000`.
- SCENE sequence is COMMIT low 16; NAV sequence is CONTROL high 16.
- DMA and PIO writes are blocked; `VIDEO_DMA_CONTROL_UNAVAILABLE`.
- `0x650/0x654` are never used as action writes.
- `rev ff`, BAR all-ones, or magic mismatch produces DISCONNECTED; no automatic
  switch to simulation in a production launch.

## Tests

```bash
python -m pytest tests/test_pcie_protocol_service.py -q
```

9 tests cover: 123-register uniqueness/alignment, HUD vectors and blink guard,
SCENE BCD/clear, NAV vector, ISP signed offset/round-trip, ALGO bits 56–59,
DIAG/DMA read-only, and concurrent HUD serialization.
