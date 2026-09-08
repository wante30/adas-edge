# Competition model build

Date: 2026-09-06. Both candidates were built with the existing split-output conversion pipeline, without changing the YOLO11n architecture or checkpoint.

| input | ONNX source | RKNN output | size | SHA-256 | board smoke |
|---:|---|---|---:|---|---|
| 320 | `tmp/rk3566_profile/yolo11n_320.onnx` (opset 12, `[1,84,2100]`) | `deployment/rknn/yolo11n_rk3566_int8_split_320.rknn` (`boxes [1,4,2100]`, `scores [1,80,2100]`) | 3,893,761 bytes | `60F17E9BFE20B27DC7D0BE068BC9852024E4A941847F881E1890BF96B2C977DA` | **YES**, RK3566 load/inference/decode path; rough inference 24.14 ms |
| 416 | `tmp/rk3566_profile/yolo11n_416.onnx` (opset 12, `[1,84,3549]`) | `deployment/rknn/yolo11n_rk3566_int8_split_416.rknn` (`boxes [1,4,3549]`, `scores [1,80,3549]`) | 3,954,625 bytes | `87DFD6D480760D105E0ABD7D04829D5059C8C5721BD7E3D6C53784DE9793BB43` | **YES**, RK3566 load/inference/decode path; rough inference 38.35 ms |

Source checkpoint: `C:\Users\JingyuWang\yolo11n.pt`, SHA-256 `0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1` (COCO YOLO11n). Calibration: `tmp/rk3566_profile/calibration_wsl.txt`, 25 COCO128 images. Toolkit: `rknn-toolkit2 2.3.2` in WSL Python 3.10; target `rk3566`; `mean_values=[[0,0,0]]`, `std_values=[[255,255,255]]`; INT8 `do_quantization=True`; static NHWC uint8 runtime input. The generated models were loaded on LubanCat RK3566 with `rknn-toolkit-lite2/librknnrt 2.3.2`, driver 0.8.2. Smoke used one COCO128 image after three warmups; it is not a full benchmark or accuracy claim.
