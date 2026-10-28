# 2026-10 Qt HMI 一次性开发脚本归档

这些脚本是 2026-10-20 至 2026-10-21 开发 RK3566 Qt/Tesla HMI 时使用的**一次性补丁/探测脚本**，不是生产入口，也没有被核心模块导入。

保留原因：它们记录了当时对 QML、GPU 场景、PGV 快速读取、状态面板、导航 OCR、双缓冲等问题的实际修改过程，便于回溯。运行前请先读脚本内容；不要在生产板端直接执行。

生产入口请使用：

- `deployment/lubancat/qt_gpu_probe/`
- `deployment/lubancat/mali_video/`
- `tools/competition_console/`
