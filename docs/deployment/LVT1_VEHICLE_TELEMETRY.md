# LVT1 RK3566 vehicle telemetry integration (offline staging)

**Status: software decoder and fixture tests; NOT board validated and NOT connected to production Qt.**

Source of truth: teammate's `rk3566_pcie_vehicle_telemetry_v1.json` (2026-10-28), schema `pango.rk3566.vehicle_telemetry_handoff.v1`. The handoff references an FPGA build and files `rk3566_image_control/vehiclemon.c`, `include/pango_vehicle_regs.h`, and a modified kernel driver; **none of those three files was found in the synced `main` checkout at review**. Obtain and compare their actual versions before deploying.

## Contract / transport

* PCIe Endpoint 0755:0755; BAR1, 32-bit little-endian DWORD **read only** through `/dev/pango_pci_driver` with `PCI_IMAGE_MMIO_READ_CMD = 0xC008530B` (`_IOWR('S', 11, struct pango_image_mmio)`, two u32 fields, 8 bytes).
* LVT1: `0x690 MAGIC=0x3154564c`, `0x694 VERSION=0x00010000`, `0x698 GENERATION`, `0x69c CAPS & 7 == 7`.
* Snapshot: `0x6a0 sample_seq`, `0x6a4 HSST payload`, `0x6a8 lane status`, `0x6ac action status`, `0x6b0 behavior result`, `0x6b4 action accept count`.
* Constants: `0x6b8 action hold nominal 1001ms`; `0x6bc HSST stale nominal 500ms`.
* Identity check → g0 → six words (plus two static threshold words) → g1 → identity recheck. Publish **only** for g0 == g1, max five attempts/poll. `generation=0` and `0xffffffff` are legal; g0 and g1 compared for equality, not signed order. A disconnect/FPGA reset breaks the session; generation is not a boot ID.
* The old `0x680–0x68c` lane mailbox does not provide atomic cross-register snapshots. Prefer new LVT1. No BAR0 fallback, no writes, no assumptions that an old kernel allowlist can read new offsets.

## Offline code

`deployment/lubancat/qt_gpu_probe/vehicle_telemetry_v1.py` is a **parallel reader**, not a replacement for the teammate's C `vehiclemon` or the kernel module. It deliberately does not touch current video/HUD processes.

```bash
# Cross-platform fixture tests; no hardware required:
python -m pytest -q tests/test_vehicle_telemetry_v1.py

# ONLY after the new FPGA build and driver are installed on a test RK3566:
python3 deployment/lubancat/qt_gpu_probe/vehicle_telemetry_v1.py --once
python3 deployment/lubancat/qt_gpu_probe/vehicle_telemetry_v1.py --interval-ms 25
# Optional atomic latest-status file, for a FUTURE Qt consumer:
python3 deployment/lubancat/qt_gpu_probe/vehicle_telemetry_v1.py \
    --interval-ms 25 --output-json /dev/shm/adas_vehicle_status.json
```

The script prints one complete JSON status per coherent poll. Standard output is data; diagnostics go to stderr. Exit 0 = normal or one-shot success, 2 = invalid CLI args, 3 = device/ioctl/identity/output error, 4 = one-shot coherence retry exhaustion. A live poll skips incoherent snapshots rather than emitting partial data.

**Do not launch this reader until the kernel module actually accepts read ioctl for 0x690–0x6bc.** The board's existing `hud_bridge.py` maps BAR1 directly for a different, existing HUD interface. Do not install a second BAR1 writer or change its permissions to workaround failed LVT1 ioctl reads.

## Semantic requirements

| Field | Correct interpretation |
|---|---|
| `left_line_count/right_line_count` | Number of detected boundary lines on each side, **not** road lane count or self lane ID |
| Count null vs 0 | null = unknown/invalid; 0 = an explicitly valid zero |
| `left_nearest_type/right_nearest_type` | `solid/dashed/unknown`; no geometric pixel coordinates |
| `lane_source` | `hsst/rk/none`, based on FPGA effective source and freshness |
| HSST stale | invalidate both sides; never backfill one HSST side from an RK type |
| `last_action` | historical accepted type; does not mean the key is still held |
| `current_action` | `none` when `action_active=false` |
| `decision` | priority idle → reason 8 warning → deny bit → allow bit → unknown |
| `accept_count` | 32-bit accepted-action counter; not lossless action queue |
| `hsst_sample_seq` | sender diagnostic, **not** generation/coherence lock or clock |

The example fixture in the teammate handoff must decode: **left=2 solid, right=1 dashed, action=brake active, decision=warning(reason8), last seq=65535, accept_count=7**. This is a fixture, NOT an observed board result.

## HMI integration — proposed next step, not yet active

* Existing Qt 5.12.8 app: `deployment/lubancat/qt_gpu_probe/main.cpp` + `main.qml` + `video_item.cpp`. Current active binary is documented under `docs/hmi/TESLA_ADAS_HMI.md`; deployed `main.qml` and `video_item.cpp` may differ from GitHub.
* Add one read-only `VehicleStatus` QObject, independent of `VideoItem` render thread and NPU. Consume either verified C `vehiclemon` JSONL output or a single latest JSON file from this Python reader, not both in production.
* Use an **explicit Qt-side freshness deadline** (suggest 250ms starting value) based on local receipt/modification age. Even a file containing `online:true` is OFFLINE if updates stop. FPGA's 500ms HSST stale flag and RK-side reader freshness are different checks. Do not keep displaying green permission after timeout.
* Present left/right nearest type + valid count, `lane_source`, `current_action`, `decision`, `reason`, `sensor_stale`, and separate history fields. Do not draw geometrically accurate lanes or invent physical speed from this packet.
* Reuse the existing QML '车道状态' card and Tesla ego brake/turn lights **only for explicit state indications**. Do not equate `brake_allow` to an actuated brake or `lane_left_allow` to a completed lane change.
* Keep new HMI optional/feature-gated until source drift and ABI compatibility are resolved; do not replace `main.qml` or live `qt_gpu_probe` blindly.

## Deployment gate and rollback

1. Obtain teammate's **complete driver and vehiclemon C sources**, kernel module build recipe, FPGA bitstream build SHA and final timing evidence. Compare with on-board kernel 4.19 module ABI and running service stack; back up existing .ko and all local changed files.
2. Before any change, record `lspci -vv`, old ioctl responses, video 49.49 FPS baseline, AI ~9.98Hz, Qt self-report ~56FPS, `hud_bridge` applied/reject/error, and active service parameters. These are historical baselines, **not guaranteed current**.
3. Build the new module against the exact running RK kernel headers. Establish a rollback procedure and obtain owner approval **before unload/reload/reboot or any service interruption**.
4. After the new FPGA image and driver are present, test LVT1 identity, `--once`, continuous 25ms reads, injected action types, short action hold, stale HSST, counter replay, unplug/reconnect, and same-session generation churn.
5. Only after live reads pass, integrate read-only status into Qt; compare fps/latency and test offline/timeout display. Merge the PR after independent code review and on-board validation.

## Known blockers

1. Only the JSON handoff was supplied, not the teammate's actual C source, driver or FPGA image. Hardware cannot be claimed functional from host mocks.
2. Authorized remote Windows machine `DESKTOP-2HNBT2L` was **offline** during this implementation. No RK3566 SSH / kernel headers / Qt compile were available.
3. No board MMIO, interactive Ethernet actions, live HMI, or performance regression test has been performed on this new branch.
4. LVT1 only describes FPGA→RK state. YOLO→FPGA hazard event transport remains a **separate** protocol/algorithm development task.
