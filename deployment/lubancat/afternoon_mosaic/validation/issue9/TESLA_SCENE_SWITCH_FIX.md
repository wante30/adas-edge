# Tesla scene switch restored — 2026-11-02

The QML root property `sceneEnabled` shadowed the identically named C++ context property. Its self-binding left the scene disabled. Rename the QML property to `sceneRenderingEnabled` and retain the C++ `--scene-off` option. Keep visibility declarative across reparenting and Lite transitions.

Deployed only this QML fix to `/home/cat/adas_issue9_20261102/src/hmi/main.qml`; preserved the preceding file as `main.qml.before_scene_fix_2128988933689`. Restarted only the owned HMI (4533 → 93961); AI PID 4532 remained running.

Read-only observations after deployment:

- Startup log changed from `SCENE_WORKLOAD enabled false active false` to `SCENE_WORKLOAD enabled true active true`.
- `/dev/shm/tesla_live_objects.json` advanced to frame 128884 and contained actual tracked cars, people and a truck instead of an empty frame-zero document.
- Scene telemetry reported 20 live tracks and approximately 24.97 scene FPS. This is a running-state observation, not a performance acceptance result.
- Control remained HARDWARE, enabled and authorized, with forward source Q4.
- No QML errors appeared in the inspected startup tail.

No benchmark, manual control probe, or acceptance test was started. Full P2 smoothing acceptance remains outstanding.
