# Competition requirement matrix

来源：官方《FPGA 创新设计赛道选题指南》PDF（选题三），逐项按当前仓库证据核对，日期 2026-09-06。

| ID | 官方要求 | 现场验收方式 | FPGA职责 | RK3566职责 | 当前状态 | 缺失项 | 优先级 | 完成证据 |
|---|---|---|---|---|---|---|---|---|
| P0-1 | 一路或多路视频输入、处理、显示闭环 | HDMI/摄像头/给定视频现场播放并输出 HDMI | 采集、缓存、OSD、HDMI OUT | 接收帧并返回元数据 | PARTIAL | FPGA bitstream与实机闭环未提供 | P0 | `docs/architecture.md`; sources are PC/UDP contracts |
| P0-2 | MCU/FPGA高速通信（如 PCIe） | 发送帧、返回控制/检测结果 | PCIe endpoint/DMA | PCIe host、帧调度 | PARTIAL | Real endpoint/physical link; software mock passes | P0 | `competition_io/protocol.py`, `competition_io/mock_transport.py`; supplied Host BAR0 ABI documented |
| P0-3 | 识别行人、道路路障、斑马线等道路环境 | U盘道路视频现场观察 | 预处理/OSD/传统视觉可选 | AI检测与场景融合 | PARTIAL | COCO无专用 cone/road-obstacle/zebra 类，需场景验收 | P0 | `competition_ai`; COCO class limits documented |
| P0-4 | 红灯停、绿灯行、强制通行报警 | 十字路口视频+按键模拟车道/行为 | LED/箭头显示 | HSV traffic-light ROI + 决策 | PARTIAL | 真实视频/FPGA输出未联调 | P0 | `competition_ai/traffic_light.py`, `competition_ai/adas.py`, tests |
| P0-5 | 虚线变道、实线禁变道/报警 | 变道视频+按键 | 车道线/叠加 | 决策状态机 | PARTIAL | FPGA/真实车道输入未接 | P0 | decision engine tests |
| P0-6 | 行人闯灯刹车等额外道路场景 | 给定图片/视频 | OSD/报警 | danger-zone决策 | PARTIAL | 场景素材与实机 replay | P0 | pipeline decision tests |
| P1-1 | 至少四路视频并行低延迟传输 | 前后左右四路展示 | 四路采集/DDR/PCIe | 轮询推理与公平性 | PARTIAL | FPGA四路 bitstream、PCIe实测 | P1 | multistream replay reports |
| P1-2 | 夜间/复杂路口感知避障 | 夜间/路口视频 | 增强/融合 | 检测与状态机 | BLOCKED | 合法道路视频与实机验收 | P1 | `docs/competition_eval_video_plan.md` |
| P1-3 | 轻量AI实时多路识别 | 现场视频实时框与状态 | 帧预处理/OSD | RKNN NPU | PARTIAL | 最终道路数据、四路端到端实测 | P1 | RK3566 model build/smoke docs |
| P2-1 | 工程规范、时序约束、完整文档 | 检查源码/报告 | RTL规范/约束 | 软件日志/接口 | PARTIAL | FPGA工程源码与时序报告 | P2 | existing docs and tests |
| P2-2 | 提交1080P三分钟演示视频和源码 | 材料审查 | 录制完整链路 | 录制 AI/ADAS | BLOCKED | 最终实机演示 | P2 | no final recording in repo |
