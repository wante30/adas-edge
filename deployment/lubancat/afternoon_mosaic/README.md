# ADAS integration from the afternoon snapshot

Base: `9b396e8b346bfbbfa9856950937e1cd3f54516f4` (`adas-afternoon-20261101`).

`ai/` and `hmi/` are recovered from the two archives in
`releases/adas_afternoon_20261101/`. No failed mode implementation is imported.
`baseline_hashes.json` records the files before integration changes.

The HMI binary SHA-256 is
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
Brake output during tests is redirected to a simulator; production defaults
and decision conditions remain unchanged.
