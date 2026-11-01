# FAST-MOSAIC provenance

Selected files from PR #5, commit f98f3c9272ba8daa97b2db14a9421ee3e1cee2f5. The PR's history is not merged.

- p2capi_generic.c Git blob: ebc7ebcb28dbe89cced16e87684dfb2b04a6119f
- scripts/capi_detector.py Git blob: cda012b7c6cbc4af1f7af1af1879ec2cc590c95a

Both match upstream byte for byte after Git newline normalization. Only the Makefile's default RKNN include directory is adapted to ../../ai. The detector preprocessing, raw heads, DFL and NMS are reused.

The shared library lives at fast_mosaic/capi/libp2capi_generic.so; the road library /home/cat/qt_gpu_probe_ai_20261019/libp2capi.so is untouched. BDD uses 512x288 uint8 RGB NHWC, nine outputs and three scales. Python verifies the 24016-byte P2Result ABI and queries output and scale counts before accepting the model.

Environment overrides: ADAS_MOSAIC_MODEL and ADAS_MOSAIC_LIBRARY.
