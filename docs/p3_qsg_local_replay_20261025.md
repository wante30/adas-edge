# P3 QSG direct-texture branch (2026-10-25)

LOCAL_REPLAY only. PCIe endpoint is not used.

## Implementation

- `QsgVideoItem` (`QQuickItem::updatePaintNode`) with a persistent
  `QSGDynamicTexture` subclass.
- RGB565 texture, `glTexImage2D` once, `glTexSubImage2D` thereafter.
- `QSGSimpleTextureNode`, source rectangle converted from normalized
  `VideoItem.sourceViewport` to 1280x720 texture pixels.
- VideoItem FBO is hidden only in QSG mode; producers, DisplayQueue, AI latest
  slot, HUD, Tesla, bbox and QML overlays are unchanged.
- Switch:
  - default: FBO (`--video-qsg` absent)
  - QSG: `--video-qsg` or `QT_VIDEO_QSG=1`
- QSG mode is opt-in; FBO remains default.

Functional check: QSG image displays upright, correct colors, HUD/bbox/Tesla
overlays still visible, process exits normally (`duration`).

## Short A/E comparison (15-20 s, shared-memory ring)

| Metric | A FBO | A QSG | E FBO | E QSG |
|---|---:|---:|---:|---:|
| Publisher FPS | ~42-48 | ~48 | ~47.7 | ~48.8 |
| Producer FPS | 38.8 | 33-35 | 31.2 | 31.3 |
| Unique upload FPS | 28.5 | 25.7 | 17.7 | 17.3 |
| render callback | 51.0 | 38.9 | 24.3 | 7.0 |
| queue dropped | 207 | ~ | 187 | 212 |
| p95 source→upload | 39.6 ms | 66.1 ms | 91.6 ms | 116.4 ms |
| AI FPS | — | — | 3.6 | 6.9 |
| Mali | 52% | 43% | 95% | 0-45% |

The QSG E render-callback counter is not comparable: QSG bypasses the FBO
renderer counter. The authoritative comparison is unique upload FPS, which did
not improve.

## Conclusion

- QSG direct texture was implemented and visually validated.
- It did **not** improve complete-AI unique-frame upload in these short runs:
  E QSG 17.3 vs FBO 17.7 FPS; A QSG 25.7 vs FBO 28.5 FPS.
- Keep FBO as default. Do not claim a QSG win.
- Formal 60 s × 2 runs were not completed because the QSG short run failed the
  performance criterion; results above are short-run evidence only.
- Publisher alone is 60.07 FPS; with Qt it is ~48 and Qt Producer ~31-39. The
  remaining gap is source/consumer CPU and ring-copy contention, plus FBO/GPU
  load in the complete AI case.

## Commands

```bash
# QSG short run
./tools/run_qt_supervised.sh --fps 60 --duration 15 --video-qsg \
  --pgv-ring /dev/shm/pgv_bench/ring.bin --source-label pgv-local \
  --pgv-layout quad --ai-frame-file /dev/shm/qt_gpu_probe_ai.mfr1 --ai-hz 10
```

## Rollback

```bash
git revert <this-commit>
```

Default FBO is unchanged; omitting `--video-qsg` restores the baseline.
