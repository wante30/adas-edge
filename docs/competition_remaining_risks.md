# Competition remaining risks

- Real FPGA bitstream, video capture, PCIe endpoint/DMA, FPGA OSD/HDMI output, and four physical inputs are not present.
- The vendor Host reference remains a frozen-frame bring-up ABI, not continuous metadata transport.
- No approved road-video acceptance set or final 1080P demonstration is in the repo.
- COCO YOLO11n has no dedicated road-obstacle/zebra semantics and traffic-light color still needs road-video validation.
- Replay and mock tests prove software behavior only; they do not replace `RK3566 BOARD TEST` or `FPGA/PCIe HARDWARE TEST`.
