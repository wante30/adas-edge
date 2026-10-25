"""Debug RKNN vs ONNX raw output on one image."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def to_linux(path: str) -> Path:
    if path.startswith("/mnt/"):
        return Path(path)
    drive, rest = os.path.splitdrive(path)
    return Path("/mnt") / drive[0].lower() / rest.lstrip("\\/").replace("\\", "/")


def main() -> None:
    from rknn.api import RKNN
    import onnxruntime as ort

    onnx_path = "/mnt/d/Datasets/adas_edge_training/b05n_final/b05n_p2_multiclass.onnx"
    calib = sorted(Path("/mnt/d/Datasets/adas_edge_training/rknn_b05o/calib_full").glob("*.jpg"))
    listing = Path("/mnt/d/Datasets/adas_edge_training/rknn_b05o/debug_calib.txt")
    listing.write_text("\n".join(str(p) for p in calib) + "\n", encoding="utf-8")

    image = sorted(Path("/mnt/d/Datasets/adas_edge_training/b05n_bdd_multiclass/images/test").glob("*.jpg"))[200]
    bgr = cv2.imread(str(image))
    rgb = cv2.cvtColor(cv2.resize(bgr, (512, 288), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

    rknn = RKNN(verbose=False)
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3566",
                quantized_method="channel")
    print("load", rknn.load_onnx(model=onnx_path))
    print("build", rknn.build(do_quantization=True, dataset=str(listing)))
    print("init", rknn.init_runtime())
    rk = rknn.inference(inputs=[np.ascontiguousarray(rgb[None])])[0]
    rknn.release()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    on = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    print("shapes", rk.shape, on.shape)
    rk2 = rk.reshape(on.shape)
    for name, arr in (("rknn", rk2), ("onnx", on)):
        print(name, "box ch", arr[0, :4, 0], "class ch", arr[0, 4:, 0])
        print(name, "cls min/max", float(arr[0, 4:].min()), float(arr[0, 4:].max()),
              "box min/max", float(arr[0, :4].min()), float(arr[0, :4].max()))
    diff = np.abs(rk2.astype(np.float32) - on)
    print("diff mean/max", float(diff.mean()), float(diff.max()))
    print("corr class ch", float(np.corrcoef(rk2[0, 4:].ravel(), on[0, 4:].ravel())[0, 1]))


if __name__ == "__main__":
    main()
