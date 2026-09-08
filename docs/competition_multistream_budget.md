# Competition multistream budget

Measured RK3566 NPU-only service rates from existing reports: 320 ≈43.72 FPS, 416 ≈28.80 FPS, 640 ≈12.65 FPS. A stable four-stream starting point is round-robin at 6 FPS per stream (24 FPS aggregate), leaving margin for decode/postprocess; 8 FPS per stream (32 aggregate) is a stress target. 4×10 FPS is above the measured 320 service capacity and must not be promised. Display FPS and AI inference FPS are separate.

These are NPU/service measurements, not four-camera PCIe end-to-end benchmarks. Final rates require the actual FPGA transport and road clips.
