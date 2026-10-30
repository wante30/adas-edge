# 三种模式更新前后截图

日期：2026-10-29  
说明：before 是仓库历史运行截图；after 是本次独立测试目录的实板截图，使用本地 PGVF fixture + 第二 AI sidecar + LVT1 fixture，**不是 PCIe/LVT1 实板通过**。

| 模式 | Before | After |
|---|---|---|
| 单路检测 | `before/qt_ui_default_q2.png` | `../lvt1/single_live_targets_20261029.png` |
| 四路检测 | `before/qt_ui_optimized_quad.png` | `../lvt1/quad_live_targets_20261029.png` |
| 空间视图 | `before/qt_ui_optimized_spatial.png` | `../lvt1/spatial_live_targets_20261029.png` |
| 左变道拒绝（新增状态） | — | `../lvt1/single_left_change_denied_20261029.png` |

After 图中的 `SIM` 表示 LVT1 是 fixture；YOLO 目标是真实 RKNN 推理结果，但输入为本地视频。测试时生产 `pango-video-capture` 处于 `capture_io`，没有真实 PCIe 帧可用。
