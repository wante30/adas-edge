# 比赛需求矩阵（选题三）

更新日期：2026-10-28  
赛题：2026 全国大学生嵌入式芯片与系统设计竞赛 · 紫光同创 FPGA 创新设计赛道 · 选题三  
官方依据：`docs/competition_rk3566_delivery_plan_20261015.md` 对原始 PDF 第 14–16 页（印刷 12–14）的逐页核对；原始 PDF SHA256 `8581797a38a82bd7879a936e1c7a140c62c5285f432bfb8c18877cfe7974dcaf`。

状态定义：

- **DONE**：有实机/可复现证据，验收项完整。
- **PARTIAL**：关键链路已通，但缺少现场验收/完整数据/一方硬件。
- **TODO**：有代码/计划，未做端到端。
- **BLOCKED**：依赖不可访问的硬件/资料/数据。

---

## 1. 总表

| ID | 官方要求 | 现场验收 | FPGA 职责 | RK3566 职责 | 状态 | 真实证据 | 主要缺口/风险 |
|---|---|---|---|---|---|---|---|
| R0 | 紫光同创 FPGA 或 FPGA+MCU；外接 ARM 性能 ≤ RK3568 | 硬件清单 | 主控 | RK3566 | **DONE** | LubanCat-1 RK3566；PG2L100H 交接 | 需最终硬件清单/照片 |
| R1 | 图像采集、处理、LCD/HDMI 显示闭环 | 摄像头/HDMI 输入→HDMI 输出 | 采集/缓存/OSD/HDMI | 接收/显示 | **PARTIAL** | FPGA→PCIe→RK 视频 49.49 FPS；RK HDMI 截图 | FPGA HDMI OSD 未肉眼验证；完整闭环未录 |
| R2 | 基于 FPGA 的图像处理/感知 | 现场演示处理效果 | ISP/ALGO/AVM/车道 | 配置/展示 | **PARTIAL** | ISP/ALGO 寄存器事务历史通过；AVM 640 模块仿真 409,600/409,600 | 无完整 FPGA 工程/时序/资源；MobileNet 未证实 |
| R3 | 高速通信（如 PCIe） | 发帧/回传状态 | Endpoint/DMA | Root Complex/驱动 | **PARTIAL** | 0755:0755、5GT/s x1、pango_video、HUD applied、HSST 实时 | ADSF/ADSR 未实机；完整双向元数据/四路 DMA 未验收 |
| R4 | 至少四路并行采集/低延迟传输（高阶） | 四路展示 | 四路采集/DDR/PCIe | 四路调度/显示 | **PARTIAL** | HMI 显示 2×2 四宫格；采集 49.49 FPS | 当前是一路 1280×720 mosaic；四路独立 DMA/同步未证明 |
| R5 | MCU/端侧轻量 AI 实时识别 | 现场视频实时框 | 预处理/OSD | RKNN NPU | **PARTIAL** | A 模型 10 Hz；主 RKNN p50 29.15 ms；mAP/资源数据 | 比赛域精度/夜间/小目标/四路未验收 |
| R6 | 红灯停、绿灯行、强制红灯行驶报警 | 路口视频+按键 | LED/箭头/OSD | 检测+颜色+决策 | **PARTIAL** | 策略/测试/专模/HSV；HUD 字段 | 当前 HMI “信号灯未识别”；无比赛域红停绿行实机录像 |
| R7 | 虚线变道、实线直行、强制实线变道报警 | 变道视频+按键 | 车道/OSD | 决策 | **PARTIAL** | 策略测试；FPGA HSST 实时左右类型/数量 | HMI 未消费 HSST；无真实车道变道端到端验收 |
| R8 | 带导航的真实道路视频，根据导航模拟智驾 | 导航视频 | OSD/状态机 | 导航解析+决策 | **TODO** | OCR 链路存在；`q2_navigation_reader.py` | OCR p50 ~1.2 s，当前 `valid=0`；无导航跟随验收 |
| R9 | 其他 1–2 个道路场景（行人闯红灯刹车等） | 场景视频 | OSD/报警 | 危险事件 | **PARTIAL** | `hazard_events.py` 实验；HMI 危险卡 | 无真实正例/独立误漏报；不能宣称闯红灯/急刹已确认 |
| R10 | 硬件实物、无明显时序违例 | 现场检查 | 板卡/bitstream/时序 | 运行 | **PARTIAL** | 实板运行；PCIe/视频/HUD/HSST | 无 FPGA 工程/bitstream hash/STA 报告；不能证明时序闭合 |
| R11 | 文档、源码、性能报告 | 材料审查 | RTL/约束 | 软件/报告 | **PARTIAL** | 本仓库 docs/源码/测试 559 passed | FPGA 源码/约束/时序报告缺失；四路性能未验收 |
| R12 | 1080P、≤3 分钟演示视频 | 视频审查 | 全链路 | 全链路 | **TODO** | 无最终 1080P 三分钟成片 | 需录采集/PCIe/红停绿行/虚实线/导航/危险/回退 |

---

## 2. 逐项说明

### R0 硬件平台

- RK3566 满足“不高于 RK3568”的性能约束。
- FPGA 型号：PG2L100H-6FBG676（交接文档）。
- 需要补：最终硬件清单、板卡照片、接线图、供电/散热说明。
- 风险：如果现场更换板卡/位流，所有 hash/性能需重测。

### R1 图像采集/显示闭环

已证明：

- FPGA→PCIe→RK 视频采集 49.49 FPS；
- Qt HMI 在 RK HDMI 上显示 1280×720 画面；
- 四宫格、检测框、3D BEV 可见。

未证明：

- FPGA 端 HDMI OSD 的肉眼可见性（HUD applied 只证明寄存器锁存）；
- 从摄像头/HDMI IN 到最终 HDMI OUT 的完整“采集→处理→显示”链路；
- 现场 U 盘视频/未知视频的即插即用。

### R2 FPGA 图像处理

已证明：

- ISP/ALGO 寄存器事务（历史 HW_APPLIED）；
- AVM 640 模块级 RTL 仿真 409,600/409,600 像素、0 mismatch；
- HSST 车道邮箱实时更新。

未证明：

- 完整 PG2L100H 工程的综合/布局布线/时序；
- MobileNet 车道识别的模型/坐标/稳定性；
- 四路 AVM 在 DDR/HDMI 的完整集成。

### R3 PCIe 通信

已证明：

- 物理链路 5GT/s x1，驱动 `pango_video`；
- FPGA→RK 视频 PGVF；
- RK→FPGA HUD V2（applied seq、reject=0、error=0）；
- FPGA→RK HSST 车道邮箱。

未证明：

- ADSF/ADSR 实机；
- 四路 DMA 描述符/独立流；
- 完整双向控制/状态机闭环；
- 长时间稳定性/错误恢复。

### R4 四路视频

当前：

- FPGA 给 RK 一路 1280×720 的 2×2 mosaic；
- HMI 显示四个象限；
- 采集 49.49 FPS。

不能说：

- 四路独立 PCIe 流；
- 四路 AI 同时 10 Hz；
- 四路传感器同步/标定。

### R5 AI 推理

已证明：

- A 模型（BDD Raw-head INT8）10 Hz；
- 主 RKNN p50 29.15 ms；
- 检测框在实板 HMI 可见；
- B/C/D 板端回放。

未证明：

- 比赛域精度（当前主要是 BDD/Cityscapes/官方视频）；
- 夜间/雨天/隧道；
- 四路并发；
- 交通灯颜色/归属；
- 小目标/远处行人。

### R6 红绿灯

已有：

- YOLO TL 框 + HSV；
- B05L Specialist；
- `scenario_policy.py` 红停绿行/黄停/UNKNOWN 停；
- HUD V2 红/绿/行人字段；
- 历史 BDD50/视频结果。

缺口：

- 当前 HMI “信号灯未识别”；
- 无本车道控制灯自动归属；
- 箭头灯/夜间/远距离未验收；
- 无“红灯停/绿灯行/强制红行报警”的完整实机录像。

### R7 虚实线变道

已有：

- `scenario_policy.py` 实线拒绝/虚线许可/未知拒绝；
- FPGA HSST 实时左右数量/类型；
- HUD V2 车道字段。

缺口：

- HMI 未消费 HSST；
- 无本车横向位置/车道 ID；
- 无真实变道视频端到端验收；
- 路口转弯 vs 道路变道需继续区分。

### R8 导航

已有：

- `q2_navigation_reader.py`、OCR sidecar、导航卡；
- `scenario_policy.py` 导航提示。

缺口：

- OCR p50 ~1.2 s，当前 `valid=0`；
- 无实时导航源/路口映射；
- 无“根据导航完成模拟智驾”的验收；
- 不能把固定路线/手工箭头当导航解析。

### R9 危险场景

已有：

- `hazard_events.py` 图像空间事件；
- HMI 危险卡；
- 策略优先级。

缺口：

- 无真实正例/独立误漏报；
- 无本车速度/标定距离；
- 不能确认“急刹”；
- 不能从车辆灯色认定“行人闯红灯”；
- 需要至少 1–2 个可重复场景。

### R10 硬件实物/时序

已有：

- 实板运行；
- PCIe/视频/HUD/HSST。

缺口：

- FPGA 完整工程、bitstream hash、STA/时序报告、资源利用率；
- “无明显时序违例”无法证明；
- 现场演示脚本/回退方案。

### R11 文档/源码/性能

已有：

- 本仓库代码、文档、559 tests passed；
- 性能报告/模型 manifest；
- 部署指南。

缺口：

- FPGA RTL/约束/时序报告；
- 四路端到端性能；
- 最终演示材料。

### R12 1080P 三分钟视频

TODO。需要录制：

- 采集/预处理/PCIe 证据；
- 红停绿行/强制红行；
- 虚实线变道；
- 导航；
- 行人/障碍危险；
- 异常断流/回退；
- 不超过 3 分钟、1080P。

---

## 3. 比赛前优先级

| 优先级 | 工作 | 影响 |
|---|---|---|
| P0 | 红绿灯闭环（当前“未识别”） | 直接对应 R6，最容易现场失分 |
| P0 | 实线/虚线 + HSST 接入 HMI | 直接对应 R7 |
| P0 | 1080P 三分钟演示视频 | 材料硬门槛 R12 |
| P1 | 行人/障碍危险场景 1–2 个 | R9 |
| P1 | 导航可用（或明确降级） | R8 |
| P1 | FPGA 工程/时序/资源报告 | R2/R10/R11 |
| P2 | 四路独立流/四路 AI | R4，高阶 |
| P2 | 事件引擎/跟踪 | 创新点，非硬门槛 |

---

## 4. 诚实边界

- 当前是“软件 + 单板 + 单路 PCIe 视频 + HUD/HSST”原型，不是完整赛题闭环。
- 不能把 2×2 mosaic 写成四路独立采集。
- 不能把 HUD applied 写成 HDMI OSD 肉眼可见。
- 不能把 HSST 邮箱写成 MobileNet 已验证。
- 不能把 PC/BDD/Cityscapes 指标写成比赛域实板精度。
- 不能把文件回放写成 PCIe 四路实时。
- 不能把“策略测试通过”写成“现场场景通过”。

---

## 5. 证据索引

- `docs/audit/PROJECT_INVENTORY.md`
- `docs/architecture/SYSTEM_ARCHITECTURE.md`
- `docs/architecture/PCIE_DATA_FLOW.md`
- `docs/architecture/FPGA_RK3566_INTERFACE.md`
- `docs/deployment/RK3566_BENCHMARK.md`
- `docs/hmi/TESLA_ADAS_HMI.md`
- `docs/models/MODEL_COMPARISON.md`
- `docs/roadmap/ADAS_EVENT_ENGINE_PLAN.md`
