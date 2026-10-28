# AVM fixed-point report

- RGB888 floating output is the primary reference. RGB565-expanded floating output isolates coordinate/interpolation quantization from input quantization.
- Scan F = [4 6 8], alpha bits = [6 8 10], guard bits = [0 4].
- Chosen point: F=4, alpha=6, guard=0; gate=PASS_RGB565_REFERENCE_ONLY.
- Chosen vs RGB888: MAE 1.287060, PSNR 43.501232 dB, SSIM 0.989815, max abs 9.
- Chosen vs RGB565-expanded reference: MAE 0.192169, PSNR 55.137482 dB, SSIM 0.999094, max abs 5. The selected gate is relative to this reference only; the RGB888 comparison is reported separately and is not silently treated as a pass.
- This is a software equivalence gate; it is not FPGA timing or resource evidence.
