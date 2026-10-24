# P4 frame handoff / FBO consumption (2026-10-25)

LOCAL_REPLAY only. PCIe is not accessed. Default renderer remains FBO.

## 1. Ring correctness fix

`pgv_ring_publisher` before the fix used plain `memcpy` for metadata stores.
Now all slot/frame/sequence metadata uses GCC `__atomic_store_n` release and
readers use `__atomic_load_n` acquire:

1. Writer sets `slot_seq = 0` before memcpy.
2. Writer copies payload.
3. Release fence.
4. Writer publishes `slot_seq`, frame_id, timestamp.
5. Writer publishes `latest_seq` last.

Reader reads `latest_seq`, slot version, copies payload, then reloads slot
version; mismatches are discarded. No `slot_conflict` occurred in the 60 s
runs (2 runs).

## 2. New counters

- Publisher: `published`, `deadline_missed`.
- Reader: `ring_reader_attempted`, `ring_slot_conflict`,
  `ring_frame_unchanged`, `ring_pool_no_free`, `ring_enqueued`.
- Queue/renderer: `display_queue_*`, `renderer_pending_skipped`,
  `frame_upload_total`, `unique_frame_id`.

## 3. 60 s x 2 results

All runs: `--fps 60`, 1280x720 RGB565, 120 distinct frames.

### Ring + FBO

| Run | Publisher | Producer | Unique upload | Queue drop | p95 source→upload | CPU | Mali | AI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 60.01 | 58.98 | 57.70 | 50 | 9.30 ms | 23.0% | 41% | — |
| A2 | 60.01 | 58.99 | 56.74 | 134 | — | 19.0% | 48% | — |
| E1 | 59.82 | 58.00 | 37.61 | 1216 | 23.71 ms | 23.8% | 34% | 9.83 |
| E2 | 59.71 | 58.28 | 40.18 | 1078 | — | 24.2% | 33% | 9.85 |

Publisher `deadline_missed=0` in all ring runs.

### In-process RAM + FBO

| Run | Producer | Unique upload | Queue drop/overwrite | p95 | CPU | Mali | AI |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | 59.59 | 45.85 | 778/53 | 14.98 ms | 17.6% | 41% | — |
| A2 | 59.58 | 44.75 | 843/- | — | 19.0% | 36% | — |
| E1 | 59.36 | 41.95 | 988/69 | 19.26 ms | 21.8% | 9%* | 10.02 |
| E2 | 59.30 | 42.38 | 963/- | — | 23.6% | 36% | 9.98 |

`*` end-of-run sample, not a sustained average.

## 4. Frame conservation (representative runs)

Ring A1:

```text
reader_attempted 3572
- pool_no_free       60
= ring_enqueued     3512
- queue dropped       50
= texture uploads   3462  (57.70 FPS)
ring_slot_conflict        0
```

Ring E1:

```text
reader_attempted 3545
- pool_no_free       87
= ring_enqueued     3458
- queue dropped    1216
= texture uploads  2242  (37.61 FPS)
```

In-process A1:

```text
producer 3608
- queue overwrite 53
- queue drop     778
= uploads       2776
```

## 5. Findings

- Ring publisher reached 60 FPS with `deadline_missed=0`; the prior 48 FPS is
  not reproducible after the atomic/metadata fix and removal of the unused QSG
  QML object.
- Qt Producer reaches 58–59 FPS with Ring. The old 31–39 FPS was caused by the
  ring publication race/metadata path and the hidden QSG item's QML overhead.
- A: upload 56.7–57.7 FPS, essentially producer-bound; `renderer_pending_skipped=0`.
  No evidence that `Renderer::synchronize()` pending behavior is the limiter.
- E: producer ~58, upload 37–40; queue drops 1078–1216. The limiter is the
  Qt/QML/Scene Graph consumption cadence under full AI, not texture upload
  (`texture_upload_p95` ~4.7 ms) and not sustained Mali saturation (33–34% in
  these runs).
- The previously observed Mali 95% was not reproduced in the fixed 60 s E runs;
  it should not be treated as the steady-state cause.
- In-process RAM did not improve A and only marginally changed E; ring transfer
  is not the current A bottleneck.

## 6. Not completed

- Tesla/bbox/HUD ablation matrix was not completed. No claim is made about
  each overlay's GPU cost.
- `frameSwapped` independent counter was not added.
- QSG remains opt-in code only; default is FBO.

## 7. Commands

```bash
# Ring publisher
./pgv_ring_publisher --raw local_quad_full.rgb \
  --ring /dev/shm/pgv_bench/ring.bin --fps 60 --count 120 --seconds 70

# Ring + FBO
./tools/run_qt_supervised.sh --fps 60 --duration 60 \
  --pgv-ring /dev/shm/pgv_bench/ring.bin --source-label pgv-local \
  --pgv-layout quad --output-json /tmp/ring.json

# In-process RAM + FBO
./tools/run_qt_supervised.sh --fps 60 --duration 60 \
  --inproc-ram local_quad_full.rgb --source-label inproc-ram \
  --pgv-layout quad --output-json /tmp/inproc.json
```

## 8. Rollback

- Use `--pgv-file` with the old PGVF publisher, or `--pgv-ring` for the new
  ring path. No PCIe ABI/RTL/model change.
- Revert this commit to return to the P3 tree.
