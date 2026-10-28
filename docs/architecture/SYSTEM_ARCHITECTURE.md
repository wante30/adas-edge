# 系统架构

更新日期：2026-10-28  
范围：2026 全国大学生嵌入式芯片与系统设计竞赛 · 紫光同创 FPGA 创新设计赛道 · 选题三（ADAS）  
原则：把“已验证的实机链路”“软件契约”“计划/仿真”分开描述。

---

## 1. 一句话架构

```text
FPGA 负责确定性视频采集/缓存/图像处理/OSD；
RK3566 负责 YOLO 感知、场景事件、HMI 可视化与语义控制；
FPGA 最终执行车辆控制/比赛状态机与 HDMI/LED 输出。
```

当前实机是：

```text
FPGA → PCIe → RK3566（视频 + HSST 车道邮箱）
RK3566 → PCIe BAR1 → FPGA（HUD/SCENE/NAV/ISP/ALGO 控制）
RK3566 本地：Qt HMI + AI sidecar + HUD bridge
```

![当前实板 HMI](../hmi/assets/current_live_20261028.png)

---

## 2. 硬件与软件分层

```text
┌─────────────────────────────────────────────────────────────┐
│ 车端/台架                                                     │
│  OV5640 / HDMI IN / 四路视频 ──► FPGA PG2L100H-6FBG676       │
│                                   ├─ 采集/缓存/拼接/ISP       │
│                                   ├─ 传统图像处理/OSD         │
│                                   ├─ PCIe Endpoint 0755:0755  │
│                                   └─ HDMI OUT / LED / 控制    │
└───────────────────────────┬─────────────────────────────────┘
                            │ PCIe Gen2 x1（实测；IP 支持 x4）
┌───────────────────────────▼─────────────────────────────────┐
│ RK3566 LubanCat-1                                            │
│  pango_video.ko → /dev/pango_video0                           │
│  pango_video_capture → /dev/shm/pango-video/latest.pgv (PGVF) │
│  Qt qt_gpu_probe（Mali-G52 / RGB565 纹理 / 3D BEV / 框）      │
│  ai_bridge.py（RKNN / Raw-head / 危险 / OCR）                 │
│  hud_bridge.py（root BAR1 HUD V2）                            │
└─────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│ 开发/验证                                                    │
│  Windows PC：训练、ONNX/RKNN 转换、PC 测试台、浏览器调试台    │
│  macOS：审计、文档、Git 同步                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 三条数据平面

### 3.1 视频平面（FPGA → RK3566）

```text
FPGA 采集/拼接
  → PCIe DMA
  → pango_video.ko
  → /dev/pango_video0
  → pango_video_capture
  → PGVF 文件（40B header + 1280×720 RGB565 + CRC32）
  → Qt 读取/校验/上传纹理
```

- 当前是一路 1280×720 的 **2×2 mosaic**，不是四路独立 PCIe 流。
- PGVF 是文件/共享内存发布，不是标准 V4L2 流；`latest.pgv` 原子替换。
- 实测采集 49.49 FPS；PCIe 链路 5GT/s x1。

### 3.2 AI 平面（RK3566 本地）

```text
Qt 写 MFR1 共享帧（1280×720 RGB565 + 元数据）
  → UDP 38260 MVR1 请求（frame_id/timestamp/ROI）
  → ai_bridge.py 读取 MFR1
  → RGA/OpenCV 预处理
  → RKNN Raw-head 推理
  → DFL/NMS/坐标还原
  → UDP 38261 MVA2 结果（boxes + status JSON）
  → Qt 更新检测框/状态
```

- 当前主模型 10 Hz，结果率 9.98 Hz。
- `main-hz` 是 sidecar 内部调度；Qt `--ai-hz` 是请求频率。
- AI 结果与视频帧通过 `frame_id`/`timestamp` 关联；过期结果应丢弃。

### 3.3 控制平面（RK3566 → FPGA）

```text
Qt / AI 状态
  → hud_bridge.py（UDP 38262）
  → BAR1 HUD V2 寄存器（0x638/0x63c/0x640/0x644/0x648/0x64c）
  → FPGA OSD/LED/状态机
```

- HUD V2 是当前实际写回路径；`applied seq` 增长、`reject=0`、`error=0` 证明被接受/应用。
- HUD overlay 只出现在 FPGA HDMI 输出，不回到 PGVF 上游帧；因此不能从 Qt 截图验证。
- HSST 车道邮箱（0x680–0x68c）是 FPGA→RK 的只读状态，当前实时更新，但 Qt 尚未消费。

---

## 4. 软件模块边界

```text
competition_ai/
  detector.py / pc_detector.py     检测器适配
  traffic_light.py                HSV 灯色
  hazard_events.py                图像空间危险事件
  image_tracks.py                 轻量跟踪
  scenario_policy.py              路口/变道/导航/危险决策
  adas.py                         旧版确定性策略

competition_io/
  protocol.py                     ADSF/ADSR 软件契约（mock）
  adas_status_v1.py               ADST v1 60B CRC
  hud_v1.py / scenario_hud.py     HUD V1/场景
  multistream.py                  多流调度

deployment/lubancat/qt_gpu_probe/
  main.cpp / main.qml             Qt HMI
  video_item.cpp                  PGVF/MFR1/MVA2
  tesla_scene_item.cpp            3D BEV
  ai_bridge.py                    RKNN sidecar
  model_profiles.py               A/B/C/D
  b05p_competition_bridge.py      Raw-head adapter
  hud_bridge.py                   BAR1 HUD V2
  pcie_protocol.py / pcie_*       协议/服务

deployment/rknn/
  export_onnx.py / convert_rknn.py / decode_yolo_output.py
```

**关键原则**：检测器不拥有 PCIe/显示状态；HMI 不直接写 BAR1；HUD 写入经过 root bridge；协议层与业务层分离。

---

## 5. 实际 vs 契约 vs 计划

| 链路 | 状态 | 证据 |
|---|---|---|
| FPGA → PCIe → RK 视频 | **实机运行** | 49.49 FPS、PGVF、截图 |
| RK → PCIe BAR1 HUD | **实机运行** | HUD magic、applied seq、reject=0 |
| FPGA HSST 车道邮箱 | **实机运行** | 0x680 seq 递增、0x688 valid |
| Qt AI sidecar | **实机运行** | 10 Hz、summary |
| MFR1/MVR1/MVA2 | **实机运行** | 代码 + summary |
| ADSF/ADSR | 软件 mock | `competition_io/protocol.py`、单元测试 |
| HUD V1 / ADST v1 | 软件/历史 | 代码、测试 |
| 四路独立 PCIe DMA | 未验证 | 当前是 2×2 mosaic |
| FPGA OSD 可见性 | 未截图验证 | HUD applied seq 只能证明写入/锁存 |
| MobileNet 车道识别 | 未证实 | 只有 HSST 邮箱，无算法/模型证据 |
| 四路 AI 同时 10 Hz | 未验证 | 文件回放 27.9 FPS 聚合，不是实机 |

---

## 6. 职责分工（比赛语义）

| 能力 | FPGA | RK3566 |
|---|---|---|
| 视频采集/缓存/拼接 | ✅ 主责 | 接收 |
| ISP/去雾/图像处理 | ✅ 主责 | 可下发配置 |
| HUD/NAV/OSD/LED | ✅ 执行 | 生成语义 |
| 车道线/HSST | ✅ 现有邮箱 | 读取/融合（待接） |
| YOLO 检测 | ❌ | ✅ NPU |
| 交通灯颜色/归属 | 可选传统视觉 | ✅ 主责（HSV/模型） |
| 危险事件/跟踪 | ❌ | ✅ 主责 |
| 3D HMI | ❌ | ✅ Mali-G52 |
| 比赛状态机/最终控制 | ✅ 主责 | 提供建议/事件 |
| 导航 OCR | ❌ | ✅ CPU（当前慢） |

> 当前分工原则与赛题一致：RK3566 负责 YOLO 感知与车机可视化，FPGA 负责最终车辆控制和比赛状态机。但“最终控制”是否已由 FPGA 状态机实际执行，当前资料只能证明 HUD/NAV/ISP/ALGO 寄存器写入，不能证明车辆执行器/赛道状态机闭环。

---

## 7. 相关文档

- `docs/architecture/PCIE_DATA_FLOW.md`
- `docs/architecture/FPGA_RK3566_INTERFACE.md`
- `docs/deployment/DEPLOYMENT_GUIDE.md`
- `docs/hmi/TESLA_ADAS_HMI.md`
- `docs/competition/REQUIREMENT_MATRIX.md`
