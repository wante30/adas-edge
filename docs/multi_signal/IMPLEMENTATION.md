# 多交通灯实现与验收记录

## 基线

2026-11-11 通过 SSH 核对实机进程、启动脚本、源码 SHA-256 与 Qt 二进制。实际 AI 为 `/home/cat/adas_issue9_20261102/src/ai`，Qt 为 `/home/cat/adas_issue10_lifecycle_20261103`；不是直接使用 GitHub main 或 afternoon 的历史快照。逐文件哈希见 `board_baseline_20261111.json`。本分支基线标签 `rollback/multi-signals-board-20261111`。

## 审查结论

- `competition_ai/traffic_light.py`：单 ROI HSV 可复用；`classify_traffic_lights` 是旧全局聚合，不能作为新方向输出。
- `ai/tools/competition_console/fpga_perception.py`：先按 IoU 去重，再汇总 RED/GREEN/OFF；黄色不受旧 HUD 协议支持。保留为兼容路径，不接入新信号控制。
- `ai/model_profiles.py`：实际有 FAST-Q4、FAST-MOSAIC、MTL-FAST 和含 Specialist 的模型。新模块不加载检测模型。
- `FrontPipeline`：主框 750 ms、Specialist 框 400 ms 缓存；新属性路径只接收本帧真正更新的检测，禁止把缓存框当成本帧观测。
- `MosaicRoadAdapter`：全 mosaic 框转换为指定前向相机局部坐标；新增路径固定物理 Q4，其他前向选择失效。跨接缝框另外验证。
- 现有 `AsyncTrafficLightTracker` 会缓存最多 60 张全帧及灰度图，不适合当前 CPU/内存预算；新路径使用小规模 IoU 匹配、单待处理任务和 2 Hz 属性分析，不额外做全帧光流。
- Qt 通过 MVA2 的 JSON status 扩展调试信息。正式 HUD、PCIe 寄存器、硬件许可和自动刹车不读取 multi_signal。

## 第一阶段验证

本地独立模块 7 项测试通过：多色独立、双模型去重、缓存框拒绝、过期、上下文切换、坐标转换、闪烁稳定。合成图仅用于这些单元测试。

隔离运行部署测试需设置 `ADAS_EDGE_ROOT` 和 `PYTHONPATH` 为 `deployment/lubancat/afternoon_mosaic/ai`，不能先载入仓库根的同名旧包。部署现有 62 项测试中 61 项通过；`test_auto_brake.py::ControlEligibilityTest::test_hardware_q4_rejects_nonphysical_and_wrong_context` 的旧 fixture 未设置实机版新增 physical_input 条件，返回 nonphysical_hardware_input。该控制模块和测试均未因本专项调整。
