# AVM previous-reference audit

## Confirmed before this MATLAB run

- `tools/adas_hmi_sim/configs/surround_bev.json` is `APPROXIMATE_PLANAR`: 640x360 inputs, a 640x640 canvas, and hand-authored polygons. It is a PC visual compositor, not A2D2-calibrated AVM and not an FPGA LUT.
- `tools/adas_hmi_sim/backend/calibrated_surround.py` is a four-role OpenCV remap path driven by an external YAML. It uses one K/D model per role and writes NPZ maps; it does not consume the A2D2 six-camera calibration or RGB565 fixed-point contract.
- `docs/adas_hmi_surround_rk3566.md` records that the former four videos were not a synchronized fisheye rig and that the old output was an approximate planar CPU result.
- `woodscape_devkit/omnidet/data/LUTs.pkl` is a neural-network inverse-polynomial fisheye lookup table, not the AVM ground-plane compositor LUT frozen here.
- The historical single-frame probe used A2D2 names but sampled raw images with undistorted K and did not apply the raw distortion model. Its 79.0844%/91.4253% figures are historical only, not this run result.

## This run

- Source: A2D2 `20180810_150607`, native 1920x1208, six-camera calibration copied to `outputs/avm_fpga_package/calibration/cams_lidars.json`.
- Mapping uses the official A2D2 view axes, `CamMatrixOriginal` plus `Distortion` for the fused inverse map, and ground plane Z=0.
- Four original camera folders have timestamp-alignment evidence; front-left/front-right use a source-index sequence without timestamp sidecars, so six-camera skew remains UNKNOWN.
- Selected extent: 15.000 m x 20.000 m; BEST_4CH coverage 0.952650 and BEST_6CH coverage 0.954440.
