# Competition final detector status

Final detector is **provisionally YOLO11n INT8 split-output 320**: `deployment/rknn/yolo11n_rk3566_int8_split_320.rknn`, COCO 80 classes, confidence 0.25, NMS IoU 0.45. RK3566 smoke inference is about 24.14 ms on one COCO128 frame. 416 remains fallback (`...split_416.rknn`, about 38.35 ms) until identical road clips establish a small-object recall gain. No fine-tuning is justified without failure evidence.

Four-stream planning target: 6 FPS per stream round-robin. This is a planning budget, not a PCIe end-to-end claim. Final freeze requires road-video comparison and hardware loop closure.
