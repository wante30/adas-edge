# Research 代码索引

本目录保存 B05N P2、B-2a MTL、B05L Specialist 的训练/评估/转换代码和实验证据。

| 目录 | 说明 | 运行方式 |
|---|---|---|
| `b05n_p2/` | P2 十类检测：训练、BDD50、ONNX/RKNN、INT8 分析 | 从仓库根目录运行；依赖 `tools/b05k_common.py`、`tools/b05k_runtime.py`、`tools/b05n_tl_eval.py`、`tools/b05l_eval.py` |
| `b05n_p2_mtl/` | B-2a 多任务：模型、损失、训练、评估、量化、Seg-256 | `cd research/b05n_p2_mtl`；`mtl/` 与 `scripts/` 自带 sys.path |
| `b05l_specialist/` | B05L 交通灯专模选型与评估 | 从仓库根目录运行；依赖 `tools/b05k_*` 与 `competition_ai/traffic_light_tracking.py` |

## 1. 依赖说明

B05N/B05L 的历史脚本原本在 `b05n_work/tools` / `b05l_work/tools` 工作树中运行，
依赖共享模块：

- `tools.b05k_common`、`tools.b05k_runtime`：已放入本仓库 `tools/`。
- `tools.b05n_tl_eval`、`tools.b05l_eval`：已放入本仓库 `tools/`，作为共享评估模块。
- `competition_ai.traffic_light_tracking`：已加入 `competition_ai/`（B-05E 低帧率检测 +
  LK 颜色跟踪）。

从仓库根目录运行：

```bash
PYTHONPATH=. python research/b05n_p2/tools/b05n_eval_multiclass.py --help
PYTHONPATH=. python research/b05n_p2/tools/b05n_tl_eval.py --help
PYTHONPATH=. python research/b05l_specialist/tools/b05l_eval.py --help
```

MTL 目录自带路径处理：

```bash
cd research/b05n_p2_mtl
python scripts/train_stage_b2a.py --help
python scripts/eval_mtl.py --help
```

## 2. 大文件

权重、ONNX、RKNN、数据集、演示 MP4 不在普通 Git。按
`artifacts/manifests/model_manifest.json` 的路径和 SHA256 从训练机/板端取回。
`research/b05n_p2/reports/assets/` 和 `research/b05l_specialist/reports/assets/`
只保留代表性 JPEG/PNG，不含原始视频帧。

## 3. 统一研究记录

`docs/research/MTL_RESEARCH_SUMMARY.md`：研究问题、模型结构、分阶段训练、
量化、Seg-256、C API、消融、系统集成和局限。
