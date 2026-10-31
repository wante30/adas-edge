# B05L 交通灯 Specialist（C/D 方案）


> 运行：从仓库根目录执行，或设置 `PYTHONPATH=.`。共享模块已放在 `tools/`，颜色分类依赖 `competition_ai/traffic_light_tracking.py`。

B05L 是基于 B05N P2 的单类交通灯专模，输入 512×192（top60 ROI），
用于 C（BDD + Specialist）和 D（B05N P2 + Specialist）双模型方案。
它不是当前默认模型的替代品；默认 B 仍是单模型 B05N P2。

## 1. 选型结果

BDD50、150 ROI GT、conf 0.25、IoU 0.5：

| 模型 | 预处理 | AP50 | Recall | 8–16px Recall |
|---|---|---:|---:|---:|
| B05L P2 | Resize 512×192 | **0.6107** | **0.5800** | **0.5750** |
| B05L P2 | Letterbox 512×192 | 0.4853 | 0.4733 | 0.4250 |
| B05N P2 全画面 | 512×288 | 0.5205 | 0.4533 | 0.4125 |

Resize 是 B05L 的正确预处理；Letterbox 会明显掉点。完整报告见
`reports/B05L_specialist_model_selection.md`。

## 2. 工具

| 文件 | 用途 |
|---|---|
| `tools/b05l_prepare.py` | 数据/ROI 准备 |
| `tools/b05l_train.py` | 训练入口 |
| `tools/b05l_eval.py` | BDD50/独立集评估 |
| `tools/b05l_onnx_audit.py` | ONNX 输出检查 |
| `tools/b05l_timing.py` | 延迟测量 |
| `tools/b05l_summary.py` | 结果汇总 |

## 3. 部署

- ONNX/RKNN：`b05l_v8n_p2_raw_heads_int8_500_20261026.rknn`
- SHA256：`c11b95b481b25ee34a093390001e821af8f2503f923a52159b8576c6a6f38b6f`
- 板端路径：`/home/cat/cityscapes_arch_compare_20261027/models/`
- 当前默认 B 不使用 specialist；C/D 方案按需加载。
