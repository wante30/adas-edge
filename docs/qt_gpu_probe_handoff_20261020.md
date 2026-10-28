# RK3566 Qt Quick / Mali-G52 交接文档

更新时间：2026-10-20（本轮真实 PCIe 联调后）

先读本文件，再读 `deployment/lubancat/qt_gpu_probe/README.md`。本轮用户的目标是：把初版 QML 做得更好看，接入 **真实 PCIe 视频**，用 Mali 硬件加速尽量接近 60 fps，并接入真实模型识别。

## 当前判断

**PASS（有明确性能/AI 限制）**：

- Qt 5.12.8 + Qt Quick 在真实 LubanCat RK3566 HDMI/X11 会话运行，Mali-G52 / GLES 3.2 硬件加速证据完整。
- 板卡重启后 PCIe 链路恢复；已加载 `pango_video.ko`，真实 publisher `/dev/shm/pango-video/latest.pgv` 实测约 **58.8 fps**。
- Qt 已从真实 PGVF 输入渲染单路、四路和空间视图；不再用 replay 冒充 PCIe。
- 真实四宫格布局已确认：Q1 前视、Q2 导航 HDMI、Q3 车内、Q4 路口。
- **主视默认 Q2（右上）**；点击底部四张卡任意一张可切换主视，MAIN 标记跟随，AI 的 road quadrant 同步切换。
- 导航卡底部增加 `Q1/Q2/Q3/Q4` 导航源芯片；导航识别源和主视相互独立。
- 主视/导航选择通过 `MVS1` 控制包动态更新 AI sidecar，不需要重启 RKNN；已验证日志 `SELECT road=... navigation=...`。
- Q1 曾出现输入冻结（两帧原始像素差 0.0，而 Q2/Q3/Q4 正常）；板卡重启/恢复采集后 Q1 恢复实时。若复现，先查前视源/FPGA 通道。
- 真实 AI sidecar 已跑通：选择 Q1 时截图有 6 个真实检测框；坐标与 Q1 对齐。
- **Tesla-BevCam-Clone 已接入空间视图**：使用 clone 的 BEV 渲染序列做暗色适配，叠加真实 ADAS 计数；上游 torch/YOLO/depth 未在 RK3566 上重跑，所以标“视觉演示/非实时几何”。
- **右下角 ADAS 卡已改为实时 BEV 投影**：用当前主视的 `aiBoxes` 绘制，主视切换后跟随；不再是一张静态 clone 图。
- **PCIe HUD V2 已打通**：root helper 映射 BAR1 `resource1`，Qt 通过 UDP 38262 发感知快照；实测手动红灯 `ext_request=0x3`、绿灯+行人 `0xd`、自动行人 `0x9`，applied seq 递增、error=0。
- 上位机新增 **PCIe 信号测试面板**：红/绿/熄灭、行人、静止目标、超速、LKA、左右车道、HUD/ADAS 使能；AI 自动发送每 500 ms。
- 最终视觉比初版明显改进：1280×720 重新排版、切角玻璃卡、真实相机快照、顶部遥测、右侧语义卡、底部 dock、实时 ADAS BEV 和 Tesla 空间视图。
- 旧布局 Qt scene graph 无 AI 时 60.3 fps；AI 5 Hz 时约 51–57 fps；AI 3 Hz 时约 61 fps。新增 LiveBev/PCIe 测试面板后的截图档约 20–29 fps；视频唯一帧上传仍不足严格 60，必须如实汇报。
- AI 交通灯/导航 OCR 当前没有可用结果；UI 标“未识别/设计预览”，不编造。
- AI sidecar 在持续高频推理后可能变慢/停止返回；UI 1.5 秒后清空并显示离线。截图前建议重启 sidecar；降低频率有帮助但不保证。

## 关键路径

本地仓库：`C:\Users\JingyuWang\adas-edge`

Qt 独立工程：`deployment/lubancat/qt_gpu_probe/`

板端：`cat@<board-ip>`，工程 `/home/cat/qt_gpu_probe_ai_20261019`

主要文件：

- `main.qml`：新 1280×720 HUD，单路/四路/空间三模式。
- `components/HudPanel.qml`：静态 Canvas 切角玻璃面板。
- `video_item.cpp/.h`：PGVF reader、RGB565 GLES 纹理、共享显示队列、AI UDP、遥测。
- `main.cpp`：CLI、QML context、报告。
- `qml.qrc`：QML + PNG/GIF 资产。
- `ai_bridge.py`：本工程匹配的 sidecar（含 MVS1 动态选择补丁）。
- `assets/tesla_bev_demo.gif`：Tesla-BevCam-Clone 暗色适配后的空间视觉素材。
- `components/LiveBev.qml`：右下角实时 BEV 投影（当前主视 AI boxes）。
- `hud_bridge.py`：root BAR1 V2 helper（UDP 38262）。
- `hud_control.cpp/.h`：Qt UDP 客户端。
- `tools/hud_bar1_probe.py` / `hud_bar1_write_test.py` / `hud_udp_test.py`：BAR1 验收工具。
- `README.md`：完整启动、性能和限制。
- `tools/`：截图、模式点击、无 AI/AI 运行、PGVF 夹具工具。

最终截图：

- `results/qt_ui_default_q2.png`：默认 Q2 主视 + MAIN/NAV 标记 + AI。
- `results/qt_ui_selected_q1.png`：点击底部 Q1 后的主视 + 6 个真实检测框。
- `results/qt_ui_live_bev_q1.png`：右下角 ADAS 实时 BEV 跟随 Q1 目标。
- `results/hud_test_panel.png`：PCIe HUD V2 测试面板。
- `results/qt_ui_tesla_spatial.png`：Tesla-BevCam-Clone 空间视图。
- `results/qt_ui_optimized_quad.png`：真实四路 2×2。
- `results/pcie_quad_raw.png`：真实 PCIe 原始 2×2 帧。
- 旧的 `qt_ui_optimized_pcie/quad/spatial.png` 保留作为历史布局。
- 所有最终 PNG 都是从 Qt 窗口用 `xwd -id` 抓取；不是根窗口、不是 replay。

## 真实 PCIe 启动流程

1. 确认 FPGA 端点：`lspci -nn | grep 01:00.0`。
2. 若 `/dev/pango_video0` 不存在：
   ```sh
   sudo insmod /home/cat/pcie_test/pango_video_t05/pcie_video/driver/pango_video.ko
   ```
3. 开机可能已有 publisher 输出到 `/dev/shm/pango-video`。先 `pgrep -af pango_video_capture` 和 `cat /dev/shm/pango-video/status.json`；已有就不要启动第二个（会 `capture_io`）。若没有：
   ```sh
   sudo /home/cat/pcie_test/pango_video_t05/pcie_video/app/pango_video_capture \
     --continuous --fps 60 --publish-dir /dev/shm/pango-video
   ```
4. `cat /dev/shm/pango-video/status.json` 应为 `state=running`，本次约 58.8 fps。
5. 启动 AI sidecar 与 Qt，见 README。
6. 如果 `lspci` 为空 / `PCIe Link Fail`：本次通过重启板卡恢复。不要用 PGVF 夹具截图冒充真实 PCIe。

## PGVF 与渲染实现

- 40 字节头：`PGVF`、version=1、1280、720、frame_id、generation、CRC32、payload=1,843,200、timestamp_ns。
- Qt 端：
  - `stat` 文件签名（dev/ino/mtime）只读新帧；
  - 预分配 4 个 frame slot；
  - 读取 payload 直接进入 slot，zlib CRC 校验；
  - 共享 `DisplayQueue` 最多 4 帧，render 线程直接 `take()`；
  - 一个复用 RGB565 纹理，`glTexSubImage2D`；
  - 单次 `QQuickFramebufferObject`，不逐帧建 QML 对象；
  - 静态卡片 `layer.enabled` 缓存；目标框使用 3 个预渲染 BorderImage 角标，避免每框十几个 QML 图元；
  - 单路按 `roadQuadrant` 选 `sourceViewport`（Q1=0..Q4=3）；四路用 `(0,0,1,1)`；PGVF 使用 top→bottom V，replay 保留旧 1→0；
  - `sourceViewport` setter 会 `update()`，否则模式切换后 FBO 不重绘。
- 不修改正式 EGL 路径、FPGA 协议、驱动或模型。

## AI 对接

- 使用 `$RUN/ai_bridge.py`，不要使用旧的 formal 目录 20KB 版本。
- 真实参数：
  - `--source PGV_LIVE`
  - `--layout quad --display-view front --road-quadrant 1`（默认主视 Q2）
  - `--navigation-quadrant 1`（默认导航源 Q2）
  - 主模型 `yolo11n_rk3566_int8_split_512x288.rknn`
  - 专模 `tl_specialist_512x192_hybrid2.rknn`
  - 5 Hz 截图档：`--main-hz 5 --specialist-hz 1 --ocr-hz 0.5`
- `MFR1` 写完整 1280×720 原始帧；`MVR2` 请求；`MVA2` 返回框 + 状态 JSON。
- **MVS1 动态选择**：`"MVS1" + uint8 road + uint8 navigation`，发到 UDP 38260。sidecar 在 `_listen` 里直接更新 `args.road_quadrant` / `args.navigation_quadrant`，下一次推理生效，不重启 RKNN。已验证日志 `SELECT road=0 navigation=2` 等。
- Qt `VideoItem` 暴露 `aiRoadQuadrant` / `aiNavigationQuadrant`；底部卡点击改主视，导航卡 Q 芯片改导航源；切换时清空旧框，等待新结果。
- `aiBoxes` 增加 `className/classId/confidence`；UI 实时显示车辆/行人/非机动车计数、导航状态、危险、结果年龄。
- 无结果 1.5 秒清空；来源断开 500 ms 标 `sourceOnline=false`。

## Tesla-BevCam-Clone 空间视图

- 上游 clone 仍在 `.deps/Tesla-BevCam-Clone`；本工程没有在 RK3566 上重跑 Ultralytics/torch/depth。
- 从已有 clone 输出提取 BEV 右侧画面，裁掉文字、背景压暗、生成 8 fps 循环 GIF（`tesla_bev_demo.gif`）和底卡静帧（`tesla_bev_card.png`）。
- 空间视图 `AnimatedImage` 占左侧 720×460，右侧显示真实 ADAS 计数/危险卡和 clone 说明；底卡显示同一视觉。
- 明确标注“视觉演示 / 非实时几何”；真实数字来自 ADAS/AI，不声称是 clone 当前帧几何重建。
- 若要变成实时 clone 渲染，需要在板端运行上游 YOLO/depth/lane 管线或把其场景对象映射到 RKNN 检测，属于后续工作。

## PCIe HUD V2 回传与测试界面

- 协议：`PCIE_HUD_CONTROL_PROTOCOL_V2_20261018.json`，BAR1 HUD 窗口：
  - `0x638` ext_request（perception/traffic/pedestrian/static/overspeed/lane/LKA）
  - `0x63C/0x640` BCD time0/time1
  - `0x644` commit sequence
  - `0x648` status（pending/ready）
  - `0x64C` applied sequence
  - `0x608` legacy HUD request；`0x101` = HUD + ADAS 可见
- `hud_bridge.py` 以 root mmap `/sys/bus/pci/devices/0000:01:00.0/resource1`，只接受经过校验的 JSON 字段；绑定 `127.0.0.1:38262`。
- `HudControl`（QtNetwork UDP）发快照；Qt 打开测试面板时暂停 AI 自动发送，便于保持手动测试状态。
- 自动发送每 500 ms：`perception_enable=1`、traffic 取 `aiTraffic`（RED=1/GREEN=2/其它=0）、pedestrian 取 person 检测；legacy 0x101。
- 手动测试：红/绿灯、行人、静止目标、超速、LKA、左右车道、HUD/ADAS；时间默认不发（板端时钟未同步），黄灯无协议字段。
- 验收证据（本机实测）：
  - magic `0x31445548`，caps `0x000000ff`
  - 手动红 `ext_request=0x00000003`，applied seq 递增
  - 手动绿+行人 `0x0000000d`
  - 自动行人 `0x00000009`，applied seq 两次采样 `0xa0 → 0xa3`，error 0
- HUD overlay 在 FPGA HDMI 输出端合成，不会出现在上游 PGVF；不能靠截图 PGVF 判断 HUD 是否生效，必须读 BAR1 applied/status。
- 未验证：真实 HDMI overlay 的最终像素效果、长时间写压力、驱动 reset 后的自动恢复；helper 目前是手动启动。

## 性能记录

| 场景 | Qt 源 FPS | GL 接受唯一帧 | QSG | AI | 上传 p95 |
|---|---:|---:|---:|---:|---:|
| 真实 PCIe 无 AI | 60.3 | 51.4/s | 60.3/s | off | 1.89 ms |
| 真实 PCIe + AI 5 Hz | 51.5 | 46.6/s | 56.7/s | 4.86 Hz | 2.45 ms |
| 真实 PCIe + AI 3 Hz | 56.1 | 49.4/s | 61.2/s | 3.0 Hz（末窗 0.76） | 2.33 ms |
| 真实 PCIe + AI 5 Hz 单路截图 | 51.3 | 44.2/s | 51.3/s | 4.92 Hz | 3.13 ms |

补充：

- publisher 实测 59.2 fps；
- 无 AI 时 QSG 60.3、source 60.3、上传 51.4/s；
- AI 5 Hz 时 GPU 约 40–42%，CPU 约 24%，NPU 100%；AI 3 Hz 时 QSG 61.2；
- `physical_scanout_fps` 始终 `null`，不能把 QSG 当物理扫描输出；
- 唯一视频上传没有到 60/s，主要受 latest-only、AI sidecar 和渲染竞争影响；不要宣称“端到端 60 帧唯一视频”。

## 视觉

![默认 Q2](../deployment/lubancat/qt_gpu_probe/results/qt_ui_default_q2.png)

- 顶部：ADAS 品牌、三模式页签、PCIe 输入源、PCIE/视频/AI/NPU/GPU、板端时钟。
- 主画面：真实 Q1 前视；左上信号灯卡、底部危险/监控条、AI 目标框。
- 右侧：导航、危险事件、车道状态。
- 底部：真实四宫格快照 + ADAS 空间示意。
- 四路：真实 2×2，四角标 Q1–Q4；空间：BEV 示意，标非标定。
- 参考图：`results/adas_cockpit_reference_user.png`（1536×864）；本轮按 1280×720 重排，不直接缩图。

## 已知问题 / 下一步

1. **AI sidecar 持续运行变慢或停止**：5 Hz 约 15–20 秒后出现，3 Hz 也不能保证不发生；建议加 watchdog/重启脚本。不要用“AI 在线”覆盖旧结果；当前 UI 已清空。
2. 交通灯专模看 Q1 时没有可确认信号灯，`traffic=UNKNOWN`；若要真实信号灯，需要把 Q4 路口图也送专模，或选择正确相机。
3. 导航 OCR `UNREADABLE`；Q2 是真实地图 HDMI，但 OCR ROI/模型未得到有效文本。UI 显示设计预览。
4. 唯一视频上传 44–51/s；若必须严格 60，需要继续优化 producer/GPU 或降低 AI/UI 负载，不能虚报。
5. 底部相机卡是当前 PGVF 帧快照（SNAP），标 MAIN 的是主识别源、标 NAV 的是导航 OCR 源；四路大画面才是实时 2×2。右下角 ADAS 卡是实时 AI BEV 投影，不是相机快照。
6. Q1 曾出现输入冻结，重启/恢复采集后已恢复；若复现，检查前视相机/HDMI 源、FPGA 通道与输入时钟。Qt 侧只负责显示。
7. Tesla clone 空间主画面是暗色适配后的渲染序列 + 真实 ADAS 计数叠加；上游 torch/YOLO/depth 未在板端运行，不是实时几何。右下角 LiveBev 是另一套实时 2D→BEV 示意投影。
8. 板卡系统时钟未同步，截图日期为板端时间；PCIe HUD 时间字段默认不发。
9. `run_real_pgv_ai.sh` 的清理会结束 sidecar；若 sidecar 不响应 TERM，脚本可能等待，需 `kill -9`。
10. 不要 `git add .`；仓库有大量其他未提交改动。用户未要求提交。
11. PCIe HUD V2 helper 需要 root 手动启动；HUD overlay 在 FPGA HDMI 输出端，上游 PGVF 看不到，不能靠截图验收。自动发送目前只覆盖交通灯/行人；静止目标/超速/LKA/车道是手动测试。
12. LiveBev 是 2D box 底边/尺寸的示意投影，不是标定地面模型；测距验收不能用它。
13. 加入 LiveBev/测试面板后当前截图档源 FPS 约 20–29；旧的 51–61 fps 数据来自更轻的布局，不能直接当作新布局指标。

## 结论

本轮已完成真实 PCIe PGVF、Qt Mali 渲染、三模式 HUD、可自由选择主视/导航源、实时 ADAS BEV、Tesla clone 空间视觉，以及 **PCIe HUD V2 识别信号回传和上位机测试面板**。UI 截屏、V2 寄存器证据均已落盘。剩余主要问题是 AI sidecar 长时间运行稳定性、交通灯/导航语义未完全接入、LiveBev 仅为示意投影，以及新布局下源 FPS 约 20–29；Q1 冻结只是历史现象，重启后已恢复。
