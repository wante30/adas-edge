# MTL / C API 实测证据

本目录保存 2026-10-31 的板端原始结果，均为 JSON / JSONL / 文本日志。

| 文件 | 条件 | 说明 |
|---|---|---|
| `bench_prod_b.json` | 本地 60 帧，10 Hz，Qt/HUD 运行 | B（Python RKNN Lite）完整 AI Bridge |
| `bench_prod_bcapi.json` | 同上 | B-CAPI |
| `bench_prod_full.json` | 同上 | MTL-FULL |
| `bench_prod_fast.json` | 同上 | MTL-FAST |
| `prod_switch.log` | 独立进程，生产安装文件 | B→B-CAPI→MTL-FULL→MTL-FAST→B 切换 |
| `mtl_switch3.log` | 测试副本，3 轮 | B→B-CAPI→B 循环 + MTL 尺寸/overlay 检查 |
| `tl_check.log` | 单帧 | base/adapter/B-CAPI 类别差异 |
| `qt_metrics_mtl.jsonl` | Qt 本地 PGV 回放 | scene_graph_fps / texture / CPU |
| `qt_phase.txt` | 同上 | B 与 MTL-FAST 的时间边界 |
| `validate_*.json` | 板端 | B-2a 输出/FP16 校验 |
| `mtl_integration_report_20261031.md` | 正式安装文件 + 本地直播回放 | 完整集成报告：C API、MTL、切换、Qt、回退 |
| `capi_bridge_report_20261031.md` | B-CAPI 早期接入 | 8.50 / 9.22 FPS 的历史运行；最终生产数据以 `bench_prod_*` 为准 |
| `bench_legacy_b60.json` / `bench_legacy_capi60.json` | 早期 60 帧回放 | 8.50 / 9.22 FPS 的原始 JSON |

测试输入：`/home/cat/b05g_113421_3s.rgb` 前 60 帧（1280×720 RGB565LE），
或 `/home/cat/b05n_mtl_b2a_bench_20261031/replay60/`。

注意：

- 这些是**本地回放**数据，不是 PCIe 实时整机性能。
- 独立 C API 14.33 / 4.95 / 5.97 FPS 与完整 Bridge 8.56 / 5.00 / 6.78 FPS
  是不同测试层级，不能混用。
- Qt 使用 llvmpipe 软件 OpenGL；scene_graph_fps 不代表 Mali 上限。
