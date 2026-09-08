# Competition remaining risks

- FPGA bitstream, four physical inputs, PCIe endpoint/driver and HDMI OSD are absent from this checkout.
- The supplied Host reference is a frozen-frame BAR0 PIO/DMA bring-up tool, not a continuous-stream driver; continuous metadata transport still needs endpoint integration.
- No official现场 U-disk clips or legal road-video acceptance set is present; 320 vs 416 quality is unresolved.
- COCO YOLO11n has no dedicated cone/road-obstacle/zebra classes and no traffic-light color state.
- HSV traffic-light classification is conservative and requires road-video validation, especially for tiny/night signals.
- Historical RKNN metrics are not end-to-end FPGA→RK3566→FPGA measurements.
- Final demo recording and submission package remain pending hardware integration.
