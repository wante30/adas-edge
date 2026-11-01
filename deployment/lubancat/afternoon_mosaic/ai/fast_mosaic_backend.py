"""Load the PR #5 generic ABI in an independent shared library and namespace."""
import ctypes
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("ADAS_MOSAIC_MODEL", "/home/cat/cityscapes_arch_compare_20261027/models/bdd_yolov8n_512x288_raw_heads_int8_capi.rknn"))
LIBRARY = Path(os.environ.get("ADAS_MOSAIC_LIBRARY", str(ROOT / "fast_mosaic/capi/libp2capi_generic.so")))


def available():
    return MODEL.is_file() and LIBRARY.is_file()


class FastMosaicDetector:
    def __init__(self):
        spec = importlib.util.spec_from_file_location("adas_fast_mosaic_capi", ROOT / "fast_mosaic/scripts/capi_detector.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.LIB_PATH = str(LIBRARY)
        if ctypes.sizeof(module.P2Result) != 24016:
            raise RuntimeError("Unexpected P2Result ABI layout")
        self.inner = module.CapiDetector(str(MODEL), (512, 288))
        self.detector = self
        self.last_timing = None
        self.single_model = True
        try:
            for name in ("p2_output_count", "p2_scale_count"):
                function = getattr(self.inner._lib, name)
                function.argtypes = [ctypes.c_void_p]
                function.restype = ctypes.c_int
            self.output_count = self.inner._lib.p2_output_count(self.inner._ctx)
            self.scale_count = self.inner._lib.p2_scale_count(self.inner._ctx)
            if (self.output_count, self.scale_count) != (9, 3):
                raise RuntimeError(f"BDD requires 9 outputs / 3 scales, got {self.output_count}/{self.scale_count}")
        except BaseException:
            self.inner.close()
            raise
        print(f"FAST_MOSAIC_READY model={MODEL} library={LIBRARY} outputs=9 scales=3 result_bytes=24016", flush=True)

    def detect(self, frame):
        rows, elapsed = self.inner.detect(frame)
        self.last_timing = dict(self.inner.last_timing, output_count=self.output_count,
                                scale_count=self.scale_count, backend="FAST-MOSAIC")
        return rows, elapsed

    def close(self):
        self.inner.close()
