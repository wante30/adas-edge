# FAST-MOSAIC 结果

| 文件 | 说明 |
|---|---|
| `benchmark_summary.json` | BDD Python / BDD C API / P2 C API 的统一性能表 |
| `consistency.json` | BDD Python vs C API 同帧一致性 |
| `mosaic_pcie_bench.json` | 真实 PCIe PGV 100 帧连续 FAST-MOSAIC：25.61 FPS |
| `mosaic_synthetic_bdd_capi.png` | 四路道路帧拼成 2×2，一次推理，Q1–Q4 均有框 |
| `mosaic_synthetic_bdd_capi.json` | 合成四宫格的全局坐标/象限/类别 |
| `mosaic_pcie_bdd_capi.png` | 真实 PCIe 1280×720 四宫格一次推理可视化 |
| `mosaic_pcie_bdd_capi.json` | 全局坐标、类别、置信度、象限、跨边界标记 |
| `bdd_capi_bench.log` | BDD C API int8 300 帧原始计时 |
| `bdd_capi_float.log` | BDD C API `want_float=1` 诊断 |
| `p2_capi_bench.log` | P2 C API 300 帧对照 |
| `bench_python_bdd.log` | BDD Python 300 帧 |
| `bench_capi_bdd.log` | BDD C API Python ctypes 300 帧 |
| `bdd_capi_resource_resource.json` | CPU/NPU/GPU/温度采样 |

所有性能数据条件：完整 1280×720 输入、Qt/HMI 运行、AI Bridge 未运行；
不是 AI Bridge 10 Hz 结果率，也不是 Qt 实时显示 FPS。
