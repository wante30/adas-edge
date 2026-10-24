# P2 LOCAL_REPLAY / FBO baseline (2026-10-25)

All measurements are LOCAL_REPLAY. PCIe endpoint is currently `rev ff`; no PCIe result is claimed.

## Implemented

- `pgv_ram_publisher`: 120 distinct RGB565 frames preloaded, file PGVF benchmark source.
- `pgv_ring_publisher`: 3-slot shared-memory ring, 1,843,200-byte slots.
  - writer invalidates slot sequence, writes payload, publishes slot/frame/timestamp, then latest sequence;
  - reader validates slot sequence before/after copy;
  - consumer latest-only remains unchanged.
- `VideoItem --pgv-ring PATH`: same Frame, DisplayQueue, AI latest slot, FBO renderer.
- Existing `--pgv-file` and PCIe paths remain untouched.

## Startup

```bash
# Publisher (60 Hz)
./pgv_ring_publisher --raw local_quad_full.rgb \
  --ring /dev/shm/pgv_bench/ring.bin --fps 60 --count 120 --seconds 120

# Qt, same downstream path
./tools/run_qt_supervised.sh --fps 60 --duration 60 \
  --pgv-ring /dev/shm/pgv_bench/ring.bin --source-label pgv-local \
  --pgv-layout quad --ai-frame-file /dev/shm/qt_gpu_probe_ai.mfr1 --ai-hz 10
```

## Publisher truth

```text
pgv_ring_publisher alone: 60.07 FPS / 12 s
with Qt+AI: ~48 FPS (not 60; not claimed as 60)
```

## Old PGVF vs shared-memory, same FBO

| Group | Source | Producer FPS | Unique upload FPS | render callback | Queue dropped | source→upload p95 | CPU | Mali | AI FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A old file | PGVF | 34.8 | 22.9 | 40.6 | 708 | 104 ms | 28% | 47% | — |
| A shared ring | ring | 38.8 | 28.5 | 51.0 | 207 | 39.6 ms | 17.9% | 52% | — |
| E old file | PGVF | 34.9 | 15.7 | 40.8 | 1158 | 115 ms | 97% | 23% | 5.0 |
| E shared ring | ring | 35.7 | 16.25 | 20.1 | 394 | 74.8 ms | 15.1% | 95% | 7.3 |

These are 20 s short runs, not the required 60 s × 2. They are baseline evidence, not final acceptance.

## Findings

- File PGVF `stat/open/pread` plus per-frame file publication is the first upstream loss. Ring removes that path and improves A producer 34.8→38.8 and upload 22.9→28.5.
- Ring Publisher still drops to ~48 FPS when Qt reads/copies concurrently; 60 FPS is not stable under HMI. Do not announce 60 FPS replay.
- A: render callback 51 but upload 28.5; DisplayQueue latest-only drops 207; upload is limited below render by update/queue/consumer cadence, not only texture upload (texture p95 ~2.4 ms).
- E: Mali load 95%, render callback 20 FPS, upload 16.25; AI + Tesla/HUD makes GPU/render the dominant cap.
- QSG direct texture branch was not implemented in this round. Therefore no FBO/QSG comparison is claimed and FBO is not proven to be the root cause.

## Gaps

- 60 s × 2 A/E runs not completed.
- QSG branch absent.
- Source writer under concurrent consumer is ~48 FPS, still below target.
- Full AI source/OCR timing not re-baselined in this run.

## Rollback

- Qt: use `--pgv-file /dev/shm/pgv_bench/latest.pgv` with old publisher; or `--pgv-file /dev/shm/pango-video/latest.pgv` for PCIe.
- Publisher: omit `--pgv-ring` in Qt; no change to PCIe ABI/driver.
