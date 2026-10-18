# Mali EGL RGB565 Video Benchmark

This is an independent fullscreen X11/EGL display path for the LubanCat RK3566 Mali-G52. The default video modes do not start GTK/PIL, AI, OCR, FPGA mailbox writes, KMS takeover, or the PCIe driver. `--ai-bridge` optionally starts the separately managed Python AI sidecar described below. Every run creates one X11 window and closes it when the run ends.

## Input contract

The T05 PGV1 capture ABI is 1280x720 RGB565 little-endian. The pixel payload is 1,843,200 bytes in row-major order with no row padding, so the source row stride is 2,560 bytes. A coherent DMA slot is 1,847,296 bytes after its completion record and 4 KiB alignment. The capture application snapshots the active coherent slot into cached process memory before returning its lease. The EGL upload consumes only the packed pixel payload and does not retain a DMA lease.

The current PGVF file publisher uses a 40-byte little-endian header followed by that packed payload. The existing Python PgvLiveCapture/PgvLatestCapture remains the live-input and GTK fallback path. This benchmark does not modify it, the PGV protocol, the PCIe driver, or the system graphics configuration.

## Build on the board

~~~sh
cd /home/cat/competition_cockpit_port_20261018/deployment/lubancat/mali_video
g++ -std=c++17 -O3 -Wall -Wextra -pthread mali_video.cpp \
  -L/lib/aarch64-linux-gnu \
  -Wl,-rpath,/lib/aarch64-linux-gnu \
  -l:libmali.so.1 -lX11 -o mali_video_egl_hmi
~~~

The program requires the active X11 mode to be 1280x720 and refuses to run on a non-Mali renderer. It uses eglSwapInterval(1).

## Run

Synthetic RGB565 input; the background is prepared once, then a source worker generates a moving road scene and frame-ID marker into a bounded four-buffer pool. The EGL thread only consumes the newest ready frame:

~~~sh
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
  ./mali_video_egl_hmi --source synthetic --fps 60 --duration 60 \
  --output-json /home/cat/competition_cockpit_port_20261018/output/mali_video_synthetic/summary.json
~~~

Replay previously captured PCIe payloads. Each input file must be exactly 1,843,200 bytes; --file can be repeated:

~~~sh
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
  ./mali_video_egl_hmi --source replay --fps 60 --duration 60 \
  --file /home/cat/pcie_test/pango_video_t05/frames_first/frame_g8_0000131574.rgb565 \
  --file /home/cat/pcie_test/pango_video_t05/web_evidence/browser_frame.rgb565 \
  --file /home/cat/pcie_test/pango_video_t05/t07_fifo_opt_retest/first/frame_g8_0000005285.rgb565 \
  --output-json /home/cat/competition_cockpit_port_20261018/output/mali_video_local_replay/summary.json
~~~

Cache and draw the static cockpit UI texture without AI:

~~~sh
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
  ./mali_video_egl_hmi --source replay --fps 60 --duration 60 --overlays \
  --file /home/cat/pcie_test/pango_video_t05/frames_first/frame_g8_0000131574.rgb565 \
  --file /home/cat/pcie_test/pango_video_t05/web_evidence/browser_frame.rgb565 \
  --output-json /tmp/mali_video_hmi.json
~~~

## Asynchronous AI and Dynamic HMI

`ai_bridge.py` runs the existing YOLO11n, traffic-light Specialist, navigation OCR, and hazard pipeline in a separate process. It reads an identity-checked `MFR1` snapshot of the exact submitted RGB565 frame, uses a bounded latest-only work slot, and returns detection boxes plus a separately updated RGBA status texture. OpenGL resources stay on the EGL thread. The sidecar does not open PCIe devices, write FPGA registers, or change the PGV protocol.

For real input, start the sidecar with the shared snapshot and dynamic HMI paths. It stays alive while `PGV_LIVE` has no frames and keeps the disconnected/unknown panel visible; stale detection, navigation, and hazards are cleared:

~~~sh
export PROJECT=/home/cat/competition_cockpit_port_20261018
export ADAS_EDGE_ROOT=$PROJECT
export AI_FRAME=/dev/shm/mali_video_ai_frame.rgb565
export DYNAMIC_HMI=/dev/shm/mali_video_dynamic_hmi.rgba

python3 $PROJECT/deployment/lubancat/mali_video/ai_bridge.py \
  --live-frame-file $AI_FRAME \
  --dynamic-hmi-file $DYNAMIC_HMI \
  --source PGV_LIVE --layout front --display-view front \
  --main-model /home/cat/adas-edge-rect/rect_models/yolo11n_rk3566_int8_split_512x288.rknn \
  --specialist-model /home/cat/adas-edge-rect/tl_check/tl_specialist_512x192_hybrid2.rknn \
  --summary /tmp/mali_video_ai.json
~~~

In a second terminal, run the EGL renderer against the existing PGVF publisher:

~~~sh
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
  $PROJECT/deployment/lubancat/mali_video/mali_video_egl_hmi \
  --source pgv-live --pgv-file /dev/shm/pgv_live_codex_20261018_postreboot/latest.pgv \
  --layout front --view front --fps 60 --duration 1200 --ai-bridge \
  --ai-frame-file $AI_FRAME --dynamic-hmi-file $DYNAMIC_HMI \
  --output-json /tmp/mali_video_pgv_hmi_20min.json
~~~

For local replay, prepare a directory of distinct 1,843,200-byte RGB565 frames, pass every file to `mali_video_egl_hmi --source replay --layout front --view front`, and configure the sidecar as `--source LOCAL_REPLAY`. A 30 FPS source replayed at 30 FPS is reported as 30 FPS; it is not counted as 60 unique frames/s. A replay must never be labeled `PGV_LIVE`.

## Board Verification

The saved reports in `results/` are from the RK3566 Mali-G52 at HDMI 1280x720@60Hz. The refresh mode is a display setting, not a scanout measurement.

| Stage | Evidence and result |
|---|---|
| A, video only, 60 s final run | `SYNTHETIC`; the producer generated 3,601 source IDs and the EGL path submitted 3,598 unique frames (59.968 FPS), input 60.001 FPS, 2 latest-only skips; each ID changes the moving road geometry and visible 24-bit marker; input interval p50/p95/p99 16.664/17.507/18.381 ms, submit interval 16.702/19.106/21.553 ms; generation p95 1.750 ms, texture upload p95 2.134 ms, draw-call p95 9.882 ms, swap p95 5.070 ms; CPU 8.44% of four-core machine, GPU p95 35%, RSS peak 56.9 MB, max temperature 27.22 C; RGB565 red readback passed. See `results/synthetic_source_final_60s_20261019.json`. |
| B, cached static HMI, 60 s | `SYNTHETIC`; input 60.002 FPS and unique submissions 59.907 FPS, 6 skips; GPU p95 36%, RSS peak 67.8 MB, max temperature 30 C. |
| C, GPU detection layer, 60 s | `SYNTHETIC`; unique submissions 59.869 FPS, 8 skips, 3,589 frames with asynchronous synthetic demo boxes; box draw p50/p95 0.949/1.575 ms. The boxes are a rendering probe, not real detections. |
| E, AI plus dynamic HMI, 20 s final run | `LOCAL_REPLAY`; 120 distinct RGB565 road images looped at 30 FPS; input 30.000 FPS, 599 source IDs submitted (29.984 FPS), EGL swaps 29.984/s, 1 latest-only skip; input interval p50/p95/p99 33.335/34.240/38.645 ms, submit interval 33.251/43.703/49.140 ms; 133 frames drew RKNN boxes, result age p50/p95 235.6/244.2 ms; YOLO 7.26 FPS, Specialist/OCR 2.52 FPS each; GPU p95 47%, NPU p95 100%, display RSS peak 273.8 MB, AI RSS peak 133.5 MB, max temperature 31.25 C, inference errors 0. |
| F, full AI soak, 20 min | `LOCAL_REPLAY`; the same 120 road images looped at 30 FPS; 36,001 source IDs produced, 35,993 submitted (29.994 FPS), 7 latest-only skips; input interval p50/p95/p99 33.335/34.125/38.279 ms, submit interval 33.281/44.857/51.995 ms; 9,049 frames drew RKNN boxes, result age p50/p95 235.3/238.6 ms; YOLO 7.02 FPS, Specialist/OCR 2.49 FPS each; GPU p95 46%, NPU p95 100%, display RSS peak 279.9 MB, AI RSS peak 225.7 MB, max temperature 33.75 C, inference errors 0. |
| Empty PGV source, 6 s | `PGV_LIVE`; 0 source frames and AI requests, 361 unavailable polls, 55 UI-only EGL swaps; the screen reports PCIe disconnected, FPGA unknown, signal unknown, and navigation stopped. No local video fallback occurred. |

The board home directory and project tree contain no source road-video container. The available replay set has 120 distinct RGB565 payloads; the 20-second and 20-minute AI tests cycle that set, so those runs do not represent 3,600 unique real-road images. Stage A is a generated moving-road source, not recorded footage; its producer creates a different scene and source-ID marker for each submitted frame. The synthetic 60 FPS path meets the unique-frame count for display validation, while real-road long-duration AI validation remains limited to the 120-frame replay set.

Screenshot: [LOCAL_REPLAY dynamic HMI](results/local_replay_dynamic_hmi_final2_20s_20261019.png). The live HDMI scanout unique FPS is not measured. `eglSwapBuffers` completion and the configured 60 Hz mode are not evidence that 60 distinct frames reached the panel. The real PGV path, DMA ownership and RGB565/stride mapping are adapted but still need an FPGA-powered end-to-end run.

## Metrics

The JSON summary separates source frames produced, unique source IDs submitted, EGL swap calls, latest-only skipped IDs, and actual HDMI scanout FPS. actual_scanout_fps deliberately remains null: EGL swaps and display refresh rate do not reveal which distinct source frames reached scanout.

Per-frame p50/p95 values cover waiting for the next input, source-to-submit age, RGB565 texture upload, draw-call submission, swap blocking, and the combined host-call time. These are CPU-side API timings. GPU load/frequency come from the Mali devfreq sysfs nodes; temperatures come from thermal-zone sysfs. Process CPU time and RSS are also reported. Input data is always labeled LOCAL_REPLAY or SYNTHETIC.

Board validation reports for 2026-10-19 are preserved in `results/`.
