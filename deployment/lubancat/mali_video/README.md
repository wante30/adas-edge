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
  -l:libmali.so.1 -lX11 -o mali_video
~~~

The program requires the active X11 mode to be 1280x720 and refuses to run on a non-Mali renderer. It uses eglSwapInterval(1).

## Run

Synthetic RGB565 input; the 60-frame image pool is built before timing begins:

~~~sh
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
  ./mali_video --source synthetic --fps 60 --duration 60 \
  --output-json /home/cat/competition_cockpit_port_20261018/output/mali_video_synthetic/summary.json
~~~

Replay previously captured PCIe payloads. Each input file must be exactly 1,843,200 bytes; --file can be repeated:

~~~sh
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
  ./mali_video --source replay --fps 60 --duration 60 \
  --file /home/cat/pcie_test/pango_video_t05/frames_first/frame_g8_0000131574.rgb565 \
  --file /home/cat/pcie_test/pango_video_t05/web_evidence/browser_frame.rgb565 \
  --file /home/cat/pcie_test/pango_video_t05/t07_fifo_opt_retest/first/frame_g8_0000005285.rgb565 \
  --output-json /home/cat/competition_cockpit_port_20261018/output/mali_video_local_replay/summary.json
~~~

Cache and draw the static cockpit UI texture without AI:

~~~sh
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
  ./mali_video --source replay --fps 60 --duration 60 --overlays \
  --file /home/cat/pcie_test/pango_video_t05/frames_first/frame_g8_0000131574.rgb565 \
  --file /home/cat/pcie_test/pango_video_t05/web_evidence/browser_frame.rgb565 \
  --output-json /tmp/mali_video_hmi.json
~~~

## Asynchronous AI Sidecar

`ai_bridge.py` reads the same local RGB565 replay files, converts them with the existing PGV BGR565 conversion, and runs the existing RKNN YOLO11n, traffic-light Specialist, navigation OCR, and hazard pipeline in a separate Python process. The renderer sends only frame IDs and source timestamps over loopback UDP. The sidecar uses a single latest-only request slot and returns a bounded box/status packet; the EGL loop never waits for inference. The sidecar does not open PCIe devices or write FPGA registers.

Start the sidecar first on the board, supplying three or more known RGB565 frames and the board's existing RKNN model files:

~~~sh
cd /home/cat/competition_cockpit_port_20261018
export ADAS_EDGE_ROOT=/home/cat/competition_cockpit_port_20261018
python3 deployment/lubancat/mali_video/ai_bridge.py \
  --file /home/cat/pcie_test/pango_video_t05/frames_first/frame_g8_0000131574.rgb565 \
  --file /home/cat/pcie_test/pango_video_t05/web_evidence/browser_frame.rgb565 \
  --file /home/cat/pcie_test/pango_video_t05/t07_fifo_opt_retest/first/frame_g8_0000005285.rgb565 \
  --main-model /home/cat/adas-edge-rect/rect_models/yolo11n_rk3566_int8_split_512x288_mmse.rknn \
  --specialist-model /home/cat/adas-edge-rect/tl_check/tl_specialist_rk3566_int8_split_512x192.rknn \
  --summary /tmp/mali_video_ai.json
~~~

Then run the renderer with the same files and `--ai-bridge`:

~~~sh
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
  /tmp/mali_video_hmi_ai_20261019/mali_video \
  --source replay --fps 60 --duration 60 --ai-bridge \
  --file /home/cat/pcie_test/pango_video_t05/frames_first/frame_g8_0000131574.rgb565 \
  --file /home/cat/pcie_test/pango_video_t05/web_evidence/browser_frame.rgb565 \
  --file /home/cat/pcie_test/pango_video_t05/t07_fifo_opt_retest/first/frame_g8_0000005285.rgb565 \
  --output-json /tmp/mali_video_render.json
~~~

Both summaries identify their input as `LOCAL_REPLAY`; replaying still images measures scheduling and resource load, not detector accuracy or motion-based hazard validity. `actual_scanout_fps` remains null until HDMI scanout is measured independently.

## Metrics

The JSON summary separates source frames produced, unique source IDs submitted, EGL swap calls, latest-only skipped IDs, and actual HDMI scanout FPS. actual_scanout_fps deliberately remains null: EGL swaps and display refresh rate do not reveal which distinct source frames reached scanout.

Per-frame p50/p95 values cover waiting for the next input, source-to-submit age, RGB565 texture upload, draw-call submission, swap blocking, and the combined host-call time. These are CPU-side API timings. GPU load/frequency come from the Mali devfreq sysfs nodes; temperatures come from thermal-zone sysfs. Process CPU time and RSS are also reported. Input data is always labeled LOCAL_REPLAY or SYNTHETIC.

Board validation reports for 2026-10-19 are preserved in `results/`.
