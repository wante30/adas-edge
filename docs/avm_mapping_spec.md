# AVM mapping specification

- Source references: [A2D2 official tutorial notebook](https://aev-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/tutorial.ipynb); [OpenCV fisheye camera model](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html).

## Frozen coordinates

- Vehicle frame: +X right, +Y forward, +Z up; ground is Z=0. A2D2 source axes are +x forward, +y left, +z up.
- BEV: 480 x 640 pixels, x range [-7.500, 7.500] m, y range [-10.000, 10.000] m. Pixel centers are used.
- Input working image: 640 x 320 RGB565 little-endian at the FPGA boundary. The retained PC reference remains RGB888.

## Per-pixel operation

For each output pixel, inverse-project the ground point into selected A2D2 camera(s), apply the official fisheye/telecam distortion model, bilinearly sample the source, and blend the two highest-scoring valid cameras. This fused inverse map is equivalent to undistort -> ground projection -> remap while avoiding a full intermediate image in FPGA.

## Selection

- BEST_4CH IDs: front_right, side_left, side_right, rear_center. Coverage 0.952650, hole 0.047350, overlap 0.245475 of the BEV.
- BEST_6CH IDs: 0..5. Coverage 0.954440, hole 0.045560, overlap 0.460827 of the BEV.

## LUT contract

- `avm_selected_96b_le.bin` and `.hex` use one 12-byte entry per row-major output pixel. The exact field offsets and reserved bits are in `outputs/avm_fpga_package/lut/entry_layout.md`.
- Direct-coordinate versus linear-address checks report 0 and 0 mismatches over valid A/B samples.
