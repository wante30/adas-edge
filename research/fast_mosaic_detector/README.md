# FAST-MOSAIC / BDD C API 独立研究

日期：2026-11-01
目标：在 RK3566 上把 BDD-YOLOv8n Raw-head INT8 迁移到 C API，并用一次推理覆盖整张 1280×720 四宫格画面。
范围：模型/C API/离线验证；不修改正式 Qt、`ai_bridge.py`、systemd、生产 `.so` 或默认模型 B。

---

## 1. 结论

- BDD-YOLOv8n Raw-head 实际是 **9 个输出、3 个尺度（P3/P4/P5）**，不是 12 个输出。
- BDD C API 完整检测：
  - C 层 300 帧：**25.52 FPS**，单帧 p50 **38.98 ms**。
  - Python ctypes 路径 300 帧：**29.99 FPS**，单帧 p50 **32.48 ms**。
  - `rknn_run` p50 17.5 ms；`want_float=0` 输出获取 p50 2.45 ms。
- 同条件 BDD Python RKNN Lite：**21.43 FPS**，单帧 p50 46.0 ms。
- 同条件 P2 C API：**18.76 FPS**，单帧 p50 52.96 ms。
- 目标 **20 FPS 已达到**，因此本轮不重新训练轻量模型；继续压缩模型会引入新的精度/量化风险，收益不足。
- 真实 PCIe 四宫格：一次推理，最佳帧 17 个框，Q1 7 / Q2 3 / Q3 0 / Q4 7；Q3（车内）无误检。
- 真实 PCIe PGV 连续 100 帧：**25.61 FPS**，单帧 p50 31.32 ms。
- 合成 2×2 坐标测试：Q1/Q2/Q3/Q4 = 3/2/3/2 个框，一次推理 19.6 ms。
- 四宫格可视化：`results/mosaic_pcie_bdd_capi.png`；结构化结果：`results/mosaic_pcie_bdd_capi.json`。

## 2. 文件

| 路径 | 说明 |
|---|---|
| `capi/p2capi_generic.c` | 9/12/15 输出通用 C API；保留原 `P2Result` ABI |
| `capi/rknn_bench_generic.c` | 通用 3/4 尺度测速程序，`want_float=0/1` |
| `capi/Makefile` | 板端编译 |
| `scripts/capi_detector.py` | ctypes 封装，读取真实 `output_count` / `scale_count` |
| `scripts/bench_python_bdd.py` | BDD Python RKNN Lite 300 帧测速 |
| `scripts/bench_capi_bdd.py` | BDD C API Python 路径 300 帧测速 |
| `scripts/bench_resource_sampler.py` | CPU/NPU/GPU/温度采样 |
| `scripts/run_mosaic_pcie.py` | 真实 PCIe PGV 四宫格一次推理 + 可视化 |
| `scripts/test_bdd_capi.py` | BDD Python vs C API 同帧一致性 |
| `results/` | 测速日志、资源、四宫格 PNG/JSON |

## 3. 编译

板端需要 RKNN Runtime 2.3.2 和 `rknn_api.h`（仓库已有）：

```bash
cd research/fast_mosaic_detector/capi
make CC=gcc
```

等价：

```bash
gcc -O3 -fPIC -Wall -Wextra -I../../../deployment/lubancat/capi/include \
    -shared p2capi_generic.c -lrknnrt -lm -o libp2capi_generic.so
gcc -O3 -I../../../deployment/lubancat/capi/include \
    rknn_bench_generic.c -lrknnrt -lm -o rknn_bench_generic
```

本实验没有替换生产 `/home/cat/qt_gpu_probe_ai_20261019/libp2capi.so`。

## 4. 运行

板端测试目录示例：

```text
/home/cat/fast_mosaic_20261101/
  libp2capi_fast.so       ← libp2capi_generic.so 的副本
  rknn_bench_fast         ← rknn_bench_generic 的副本
  capi_detector.py
  ai_bridge.py / b05p_competition_bridge.py  ← 仅用于 Python 对照
  bench_*.py
```

```bash
# C 层，300 帧，int8
./rknn_bench_fast /home/cat/cityscapes_arch_compare_20261027/models/bdd_yolov8n_512x288_raw_heads_int8.rknn out.json 0

# Python C API，300 帧
ADAS_P2CAPI_LIB=$PWD/libp2capi_fast.so python3 bench_capi_bdd.py

# BDD Python RKNN Lite，300 帧
python3 bench_python_bdd.py

# 真实 PCIe 四宫格，一次推理
ADAS_P2CAPI_LIB=$PWD/libp2capi_fast.so python3 run_mosaic_pcie.py
```

`rknn_bench_generic.c` 默认读取 `/home/cat/b05g_113421_3s.rgb` 的前 60 帧，循环 5 轮。

## 5. 协议

四宫格 JSON 的每个框：

```json
{
  "class_name": "car",
  "confidence": 0.85,
  "bbox": [x1, y1, x2, y2],
  "center": [cx, cy],
  "quadrant": 1,
  "quadrant_name": "Q2",
  "crosses_boundary": false
}
```

- `bbox` 为原始 1280×720 全局坐标。
- 象限按框中心分配：Q1 左上、Q2 右上、Q3 左下、Q4 右下。
- 跨边界框保留全局坐标，`crosses_boundary=true`，不强行裁剪。
- 每帧只执行一次 `rknn_run()`；输入整张 1280×720 缩放到 512×288。

## 6. 约束

- 没有修改正式 Qt、`ai_bridge.py`、systemd、生产 C 库或默认 B。
- 测试期间 Qt/HMI 仍在运行，资源数据是整板负载，不是纯模型负载。
- 当前 PCIe 输入为 1280×720 2×2 拼接；四宫格内容以实际视频为准。
- 512×288 下每象限约 256×144，小目标漏检是已知代价；本轮没有改为四路独立推理。
