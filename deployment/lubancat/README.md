# LubanCat competition console

This launcher runs the current native competition console on the LubanCat
RK3566 desktop. It uses the existing board-side 512x288 main RKNN model,
512x192 traffic-light specialist model, and `front_eval_intersection_20261001.mp4`.

The prerecorded-video UI opens paused on the intersection scene. For live PCIe
input, set `PGV_FILE` to the PGV publisher's `latest.pgv`. Press B and drag a
box around the selected lane's illuminated lamp housing to bind it; Esc exits.
The binding is explicit, and the lamp color comes from current pixels inside
that tracked housing.

By default the launcher does not write to FPGA. For a live perception-only V2
mailbox test, set `FPGA_HUD=1` together with `PGV_FILE`. It verifies HUD magic,
version, and capabilities before opening the sender, then sends observed HUD
perception fields only; it does not send turn-signal or vehicle-control
commands. Do not pass a simulated `--mode` with this option.

The GTK3 backend is used because the board's OpenCV build has no HighGUI
backend. The logical 1600x900 cockpit is resampled to the 1024x576 drawable
area on the 1024x600 display; mouse coordinates still map to the logical
canvas. Pillow 7 compatibility is retained without changing the board's
system Python packages. If RapidOCR is absent, navigation returns an
unconfirmed observation instead of stopping the UI. The optional spatial view
uses the bundled renderer only; its geometry is illustrative and uncalibrated.

## Board measurement

On 2026-10-18, visible GTK playback processed 90 frames from a 30 FPS, 3-second
intersection clip in 9.76 seconds (9.22 processed FPS). The same path took
17.94 seconds (5.02 FPS) before moving the GTK resize from Cairo to a
pre-sized pixel buffer, an observed 1.84x throughput improvement. The optimized
run measured 40.22 ms median main-model inference, 24.30 ms specialist
inference, 53.22 ms cockpit rendering, and 4.17 ms GTK event-wait time. All 90
source frames were retained; this playback is still slower than real time.
Model inputs and confidence/NMS thresholds were unchanged.

This is a board-side GUI/playback measurement, not an accuracy evaluation or
the full FPGA/PCIe competition loop. The lamp group starts unbound; bind the
visible lamp housing before interpreting its color. Live FPGA perception
transport is separately opt-in through `FPGA_HUD=1`.

Run `run_competition_console.sh` from the desktop launcher or a terminal. Each
run writes its trace and summary to a new timestamped directory under `output`.

For the live navigation and analysis workflow, launch `run_navigation_analysis.sh`
or use the `比赛导航与分析` desktop icon. It reads the board's current PGV1
publisher from `/dev/shm/pgv_live_codex_20261018_postreboot/latest.pgv`, uses
Q3 for road perception and Q1 for the navigation screen, enables the experimental
hazard analysis, and sends observed perception fields through the FPGA HUD V2
mailbox. It does not send vehicle-control commands. The publisher path can be
overridden with `PGV_FILE` if the capture service uses a different directory.
