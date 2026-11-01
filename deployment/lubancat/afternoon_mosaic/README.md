# ADAS integration from the afternoon snapshot

Base: `9b396e8b346bfbbfa9856950937e1cd3f54516f4` (`adas-afternoon-20261101`).

`ai/` and `hmi/` are recovered from the two archives in
`releases/adas_afternoon_20261101/`. No failed mode implementation is imported.
`baseline_hashes.json` records the files before integration changes.

The original archived HMI binary SHA-256 is
`27945c8037a8689251e6a2f07ceb2f528b0f34e268b1df6c678c05770f03ff4d`.
The original backup's compiled Qt resource contains exactly the archived
`main.qml`, and that compressed resource occurs in the archived executable.
This establishes the QML source provenance independently of timestamps.

Build in a new directory with `qmake ../hmi/qt_gpu_probe.pro` then `make -j2`.
Never use a Makefile recovered from an archive: it contains old absolute paths.
The board clock can reset; use current timestamps in the independent source
copy before configuring a new build.

The verified road configuration uses `ADAS_DISABLE_ROAD_SCENE=1`,
`ADAS_LEGACY_HAZARD=1`, `ADAS_AUTO_BRAKE=1`, and `--ocr-hz 2`.
The test harness supports an isolated simulator at port 38362. After the user
explicitly requested normal commands on 2026-11-01, the active integration
session and supplied launch scripts use the existing HUD Bridge at port 38262.
The original decision conditions and AutoBrakeController are preserved.


## Implemented behavior

| Function | Afternoon baseline | Integration |
|---|---|---|
| Road detectors | B / B-CAPI / MTL-FAST / MTL-FULL | Same paths, backends, segmentation and model menu |
| Four views | Video mosaic | One complete-frame BDD inference, global boxes |
| Navigation | Q2 OCR requested at 2 Hz | Same independent OCR process and quadrant selection |
| Hazards / brake | Legacy Q1 events and existing controller | Same logic; other mosaic quadrants excluded |
| HUD / LVT1 / BEV | Existing widgets and adapters | Preserved |
| Requests | Fixed 10 Hz | Default adaptive, maximum 30 Hz; fixed mode retained |
| Displayed AI FPS | Process lifetime average | Unique accepted results over the latest 1.2 s, sampled every 200 ms |

See MODE_PROTOCOL.md for coordinates, epochs, reconnect and stale-result rules,
and fast_mosaic/UPSTREAM.md for the exact PR #5 source provenance.

## Build and run on RK3566

Use an independent directory such as /home/cat/adas_mosaic_dev_20261101.
The recovered sources require the board's existing Qt5, Mali, RKNN runtime,
models, OCR assets and production road C API library. No driver, runtime,
bitstream or model installation is performed by these scripts.

```sh
sh tools/build.sh
sh tools/start_local_replay.sh /home/cat/qt_gpu_probe_ai_20261019/local_quad_full.rgb
# When the physical PCIe input is available again:
sh tools/start_pcie.sh
# Restore the original service and original HMI autostart loop:
sh tools/restore_afternoon.sh
```

The launcher records boot ID, original HMI arguments/environment, original
autostart parent, service state and owned test PIDs in
evidence/integration_session.json. It stops the previous owned session before
launching another. Formal autostart and archived executables stay untouched.
For an already active integration session the record is reused. Normally one
original HMI supplies the recovery arguments/environment. With local replay,
if no original HMI is running, an archived restored-session record may supply
those launch parameters; the current autostart parent and service state are
rediscovered. This requires an existing recovery record on that board. An
active session record from another boot is rejected before any PID signalling.
The runtime package does not carry live PID records to another installation.

The local source is the full 1350-frame, 1280x720 RGB565 recording, played in a
loop at a requested 60 FPS. Its achieved source/render/AI rates are measured
independently. Shared AI snapshots use an independent /dev/shm file.

## Adaptive requests

--ai-hz 0 enables completion-driven requests with one outstanding frame and
a latest-frame slot; --ai-hz 10 retains the previous pacing cap.
ADAS_AI_MAX_HZ overrides the adaptive cap (default 30; allowed 1–60).
A request expires after 1.5 s. When Qt overlay updates exceed 12 ms or rendering
falls below 25 FPS, the scheduler adds 30 ms of pacing. Model changes pause
new frames until acknowledged and reset tracking and rate history. A mode
change cancels a superseded queued road-model request.

The 200 ms metrics timer also checks for AI silence. No new result packets
are required to reach zero FPS or clear offline targets. Inference counters
use monotonic pipeline counts instead of bounded timing-buffer lengths.

## Evidence and limits

validation/stage1.json and stage1.png document the recovered afternoon road
baseline. validation/stage2_mode_switches.json records epoch switching before
BDD integration. validation/stage3_switches.json and stage3_mosaic.png record
actual complete-frame BDD integration before adaptive pacing.

Later acceptance uses local replay at the user's request after the physical
PCIe feed was removed. Current replay results must not be described as fresh
PCIe throughput or proof of a physical brake actuation. Earlier live tests
received successful normal brake acknowledgements. A disconnected BAR can
return all-ones values in the original vehicle-status producer; such records
are invalid hardware evidence even if its legacy online field is true.

Validation commands:

```sh
python3 -m unittest discover -s tests -v
ADAS_VALIDATION_LABEL=local_replay python3 tools/validate_integration.py
```


## Final local replay acceptance, 2026-11-01

The final `final_replay` run completed **612.179 seconds** continuously after
model switches, road/mosaic switches, an intentional AI stop, zero-FPS/offline
feedback and reconnection. The HMI process remained running. There were no
duplicate result identities or inference/listener/crash markers in the logs.

| Measured quantity | Result |
|---|---:|
| Local recording source, requested 60 FPS | 54.603 FPS achieved |
| Texture uploads | 28.676 FPS |
| Qt scene graph | 29.444 FPS mean |
| ROAD_Q1 / MTL-FAST | 6.899 unique completions/s |
| MOSAIC_FULL / FAST-MOSAIC | 14.534 unique completions/s |
| CPU, normalized across four cores | 23.856% mean |
| GPU driver load | 36.239% mean |
| NPU driver load | 100% reported; a driver reading, not an independent utilization measurement |

Short profile-switch checks also completed for B (32 results, 9.285 FPS),
B-CAPI (29 results, 13.739 FPS) and MTL-FULL (12 results, 5.124 FPS). These are
short functional checks, not ten-minute benchmarks for each model.

The continuous interval contains 384 results with hazards. The downloaded
replay log contains 125 normal brake requests and 125 timeouts: no physical
PCIe endpoint/HUD bridge was present. Physical brake execution, fresh live
LVT1 and PCIe hot-plug recovery remain unverified. The disconnected board was
rebooted to clear its stale hardware state; no PCIe driver recovery fix is
claimed. Local replay screenshots retain some original PCIe-labelled UI
fields, which are not hardware-connectivity evidence.

Navigation code remains unchanged and OCR is requested at 2 Hz. On this
recording, the Q2 navigation card is smaller/differently positioned: 83 results
were VALID, 4 STALE and 6468 UNREADABLE, predominantly Q2_ARROW_NOT_FOUND.
Navigation is therefore not claimed to pass throughout this replay.

Reviewable evidence:

- [Acceptance report](validation/REPORT.md), [derived metrics](validation/final_summary.json)
- [Full runner report](validation/stage5_acceptance.json), [raw log snapshots](validation/stage5_logs.zip)
- [Four-view hazard screenshot](validation/final_mosaic.png), [road/MTL](validation/stage5_model_MTL-FAST.png)
- [BEV](validation/stage5_bev.png), [AI offline](validation/stage5_offline.png)
- [Runtime hashes](validation/runtime_manifest.json)

Derive the measurements independently from the archived log snapshots:

```sh
mkdir -p /tmp/adas-final-evidence
unzip validation/stage5_logs.zip -d /tmp/adas-final-evidence
python3 tools/analyze_acceptance.py /tmp/adas-final-evidence --label final_replay
```

The RK3566 runtime ZIP includes `build-stage4/qt_gpu_probe`, the independent
BDD shared library, sources/assets and launch/recovery scripts. Extract it
into an independent directory. Existing board models, Qt/Mali/RKNN libraries,
OCR dependencies and the raw replay recording are prerequisites and are not
bundled. The package preserves evidence hashes but excludes machine-specific
active/restored PID records; capture an original HMI session first on a new
installation. Restore with `sh tools/restore_afternoon.sh`; without PCIe the
original autostart can only resume waiting for its input.
