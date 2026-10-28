# ADAS Edge — 2026 紫光同创 FPGA 创新设计赛道选题三

以紫光同创 FPGA 为核心的高级辅助驾驶系统（ADAS）：FPGA 负责确定性视频采集、缓存、图像处理与 OSD；RK3566 负责 YOLO 感知、场景事件、Tesla 风格 3D 车机与语义控制；FPGA 最终执行车辆控制/比赛状态机与 HDMI/LED 输出。

- 比赛：2026 全国大学生嵌入式芯片与系统设计竞赛
- 赛道：紫光同创 FPGA 创新设计赛道
- 选题：三（ADAS）
- 仓库：<https://github.com/wante30/adas-edge>
- 本文档对应：`live/stage2`，2026-10-28 审计与同步

---

## 1. 当前真实状态（先看这里）

### 1.1 已实测运行

| 项目 | 当前值 | 证据 |
|---|---|---|
| FPGA→RK PCIe 视频 | 1280×720 RGB565，**49.49 FPS** | `/dev/shm/pango-video/status.json` |
| PCIe 链路 | `0755:0755`，5 GT/s **x1**（IP 支持 x4） | `lspci -vv` |
| 当前检测模型 | **A：BDD-YOLOv8n Raw-head INT8** | 进程参数 + summary |
| 模型 SHA256 | `a86ab053...041afb` | `artifacts/manifests/model_manifest.json` |
| AI 结果率 | **9.98 Hz** | `adas_A_live_default_summary.json` |
| 主 RKNN 调用 p50 | **29.15 ms** | 同上 |
| 主模型总耗时 p50 | **38.77 ms** | 同上 |
| HMI | 1280×720 Qt Quick，Mali-G52，自报 **56 FPS** | 实板截图 |
| NPU / GPU / CPU | 100% / mean 33.9% / 69.5% 单核 | 同上 |
| HUD V2 回写 | applied seq 递增，reject=0，error=0 | BAR1 读回 |
| HSST 车道邮箱 | 实时更新（约 30 Hz） | 0x680–0x68c 读回 |

![当前实板 HMI](docs/hmi/assets/current_live_20261028.png)

### 1.2 已实现但未完成现场验收

- 四方案模型切换 A/B/C/D；当前实际运行 A，UI 标签曾显示 B（待修）。
- B05N P2、B05L Specialist、Raw-head INT8 的 PC/板端回放数据。
- 红绿灯 HSV、交通灯专模、实线/虚线决策、行人/障碍危险事件。
- PC 原生车机测试台、浏览器 PCIe 调试台。
- FPGA ISP/ALGO/HUD/NAV/HSST 寄存器接口。

### 1.3 明确未完成 / 不能宣称

- 四路独立 PCIe 采集/DMA：当前是**一路 1280×720 2×2 mosaic**。
- FPGA 完整工程/bitstream/时序/资源报告：本机不可访问。
- FPGA MobileNet 车道识别：只有 HSST 邮箱活数据，**算法/模型未证实**。
- 红绿灯“红停绿行/强制红行”现场闭环：当前 HMI 显示“信号灯未识别”。
- 导航跟随：OCR p50 约 1.2 s，当前“导航暂不可读”。
- 行人闯红灯/急刹确认：仅有实验性图像事件，不能确认。
- 1080P 三分钟最终演示视频：未录。
- 物理 HDMI scanout FPS：未测，56 FPS 是 Qt 自报。

> 完整的“已完成/未完成/证据/风险”见 [需求矩阵](docs/competition/REQUIREMENT_MATRIX.md) 与 [项目盘点](docs/audit/PROJECT_INVENTORY.md)。

---

## 1.4 LVT1 真实数据升级（独立分支）

分支 `feature/lvt1-ego-bev-20261028` 已完成：

- LVT1 左右车道线型/数量/来源/有效性接入 Qt；
- 新增本车控制卡（油门/刹车/左右变道请求 + FPGA 判定/原因）；
- Tesla 3D 道路边界按 LVT1 实线/虚线绘制；
- YOLO 跟踪目标驱动 3D 车辆/行人，900ms 过期，无检测不生成模拟目标；
- 默认道路象限改为 Q1（前视），Q2 只作为导航；
- 修复 3D chase camera 左右镜像。

实板离线截图：

- `docs/hmi/assets/lvt1/single_live_targets_20261029.png`
- `docs/hmi/assets/lvt1/quad_live_targets_20261029.png`
- `docs/hmi/assets/lvt1/spatial_live_targets_20261029.png`
- `docs/hmi/assets/lvt1/single_left_change_denied_20261029.png`

当前状态：**实板 Qt 5.12.8 编译通过、fixture + 本地视频 + 第二 AI sidecar 截图通过；生产二进制未替换，真实 LVT1 驱动/位流未验收。**

详见 [LVT1 真实数据 HMI](docs/hmi/LVT1_REAL_DATA_HMI.md)。

---

## 2. 系统架构

```text
OV5640 / HDMI IN / 四路视频
        │
        ▼
┌──────────────────────────────┐
│ FPGA PG2L100H-6FBG676        │
│ 采集/缓存/拼接/ISP/OSD        │
│ PCIe Endpoint 0755:0755      │
└──────────────┬───────────────┘
               │ PCIe Gen2 x1
┌──────────────▼───────────────┐
│ RK3566 LubanCat-1            │
│ pango_video → PGVF → Qt HMI  │
│ ai_bridge → RKNN Raw-head    │
│ hud_bridge → BAR1 HUD V2     │
│ Tesla 3D BEV / 四宫格         │
└──────────────┬───────────────┘
               │ HUD/SCENE/NAV/ISP/ALGO
               ▼
        FPGA OSD/LED/状态机
```

详细：

- [系统架构](docs/architecture/SYSTEM_ARCHITECTURE.md)
- [PCIe 数据流](docs/architecture/PCIE_DATA_FLOW.md)
- [FPGA↔RK3566 接口](docs/architecture/FPGA_RK3566_INTERFACE.md)

---

## 3. 代码结构

```text
adas-edge/
├── competition_ai/          检测、交通灯、危险事件、场景策略
├── competition_io/          ADSF/ADSR、ADST、HUD、mock transport
├── deployment/
│   ├── lubancat/
│   │   ├── qt_gpu_probe/    当前 Qt HMI + AI sidecar + HUD/PCIe
│   │   └── mali_video/      旧 Mali HMI/备选
│   └── rknn/                ONNX/ RKNN 工具与历史模型
├── docs/
│   ├── audit/               项目盘点、Git 同步报告
│   ├── architecture/        系统/PCIe/FPGA 接口
│   ├── models/              训练、对比、数据、manifest、证据
│   ├── deployment/          RKNN、板端 benchmark、部署指南
│   ├── hmi/                 Tesla HMI、性能
│   ├── benchmarks/          历史性能
│   ├── roadmap/             ADAS 事件引擎计划
│   └── competition/         需求矩阵
├── src/adas_edge/           早期 Python 包/CLI
├── tools/                   PC 测试台、仿真、训练/评估工具
├── tests/                   559 个测试
├── rtl/avm_640/             四路 AVM 640 模块级 RTL
├── artifacts/manifests/     模型 SHA256 清单
└── README.md
```

---

## 4. 快速开始

### 4.1 PC：安装与测试

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
```

本次实测：**559 passed**（78.01 s）。

### 4.2 PC 原生车机测试台

```powershell
.\.venv\Scripts\python.exe tools\run_competition_console.py `
  --video 'D:\BaiduNetdiskDownload\紫光同创杯赛题三参考素材\video_20260929_113539.mp4' `
  --paused
```

或双击 `run_competition_console.cmd`。  
这是 PC 测试台（视频文件 + ONNX Runtime + 软件 HUD），不是 FPGA PCIe 实时 HMI。

### 4.3 RK3566 板端

当前运行目录：

```text
/home/cat/qt_gpu_probe_ai_20261019/
```

服务：

```sh
systemctl status pango-video-capture adas-ai-bridge-live adas-hud-bridge
cat /dev/shm/pango-video/status.json
```

启动 Qt：

```sh
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
QT_QPA_PLATFORM=xcb QT_XCB_GL_INTEGRATION=xcb_egl \
./qt_gpu_probe --fps 60 --duration 0 --start-mode 0 \
  --pgv-file /dev/shm/pango-video/latest.pgv --source-label pgv-live \
  --pgv-layout quad --road-quadrant 0 \
  --ai-frame-file /dev/shm/qt_gpu_probe_ai.mfr1 --ai-hz 10
```

完整步骤/故障排查：[部署指南](docs/deployment/DEPLOYMENT_GUIDE.md)。

---

## 5. 模型与训练

当前默认：

```text
A · BDD-YOLOv8n Raw-head INT8 单模型
  /home/cat/cityscapes_arch_compare_20261027/models/bdd_yolov8n_512x288_raw_heads_int8.rknn
  SHA256 a86ab05338b49be54f95aa4d0cb937f1471924d917840d0f907480e29e041afb
```

备选：

```text
B · B05N P2 10 类
C · BDD + B05L Specialist
D · B05N P2 + B05L Specialist
```

关键结论：

- BDD 主模型在 BDD holdout-1000 mAP50 `0.3831`，高于 P2 的 `0.3711`。
- P2 在 car/person/TL/sign 更好，但 rider/motor/bike 退化。
- B05L Specialist BDD50 AP50 `0.6107`、8–16px Recall `0.5750`，但 Cityscapes 上 FP 高。
- Raw-head INT8 把 P2/Specialist 拉回 ONNX 附近；BDD 仍有小差距。

文档：

- [训练报告](docs/models/TRAINING_REPORT.md)
- [模型对比](docs/models/MODEL_COMPARISON.md)
- [数据与评估](docs/models/DATASET_AND_EVALUATION.md)
- [模型清单](docs/models/MODEL_MANIFEST.md)
- [RKNN 量化](docs/deployment/RKNN_QUANTIZATION_REPORT.md)
- [板端 benchmark](docs/deployment/RK3566_BENCHMARK.md)

模型权重/ONNX/RKNN 不进入普通 Git；按 manifest 的路径与 SHA256 从训练机/板端取回。

---

## 6. Tesla 风格 3D 车机

- Qt 5.12.8 + Qt Quick + Mali-G52 + OpenGL ES 3.2。
- 1280×720，单纹理 RGB565，四宫格 + Tesla BEV。
- 四条同向车道、自车 3D mesh、FSD 蓝色路径、车辆/行人 billboard。
- 检测框来自 AI MVA2，按 1280×720 源坐标叠加。
- `RAW/FIX4/TIME` 是显示端外推，不是跟踪器。
- 当前没有真实标定/自车姿态/车道 ID；BEV 是视觉投影，不是测距。

详见 [Tesla HMI](docs/hmi/TESLA_ADAS_HMI.md) 与 [HMI 性能](docs/hmi/HMI_PERFORMANCE.md)。

---

## 7. 已知问题（2026-10-28）

1. 当前 HMI 信号灯“未识别”，导航“暂不可读”，车道“车道视觉未接入”。
2. 截图模型标签“方案 B”与实际 sidecar A 不一致。
3. 板端 `main.qml`/`video_item.cpp` 与仓库有漂移；运行二进制 hash 已记录。
4. 板端时钟未同步（显示 2024）。
5. 四路独立 PCIe/DMA 未验证。
6. FPGA 工程/时序/资源/MobileNet 资料缺失。
7. 旧 YOLO11n 320/416 smoke 指标不能归到 B05N/P2。

---

## 8. 下一步（按比赛重要性）

1. **红绿灯闭环**：当前“未识别”是 R6 的直接缺口；先修检测/颜色/归属，再做红停绿行/强制红行录像。
2. **HSST 车道接入 HMI + 实线/虚线变道验收**：FPGA 已有实时数据，HMI 未消费。
3. **1080P 三分钟演示视频**：采集/PCIe/红停绿行/虚实线/导航/危险/回退。

事件引擎（跟踪/Cut-in/TTC/行人横穿）作为创新点，最小侵入方案见 [ADAS 事件引擎计划](docs/roadmap/ADAS_EVENT_ENGINE_PLAN.md)。

---

## 9. 文档索引

| 文档 | 用途 |
|---|---|
| [项目盘点](docs/audit/PROJECT_INVENTORY.md) | 机器/路径/Git/模型/服务/缺失 |
| [Git 同步报告](docs/audit/GIT_SYNC_REPORT.md) | 推送分支/commit/PR/默认分支 |
| [系统架构](docs/architecture/SYSTEM_ARCHITECTURE.md) | FPGA/RK/PC 分工 |
| [PCIe 数据流](docs/architecture/PCIE_DATA_FLOW.md) | PGVF/MFR1/MVA2/HUD/HSST |
| [FPGA 接口](docs/architecture/FPGA_RK3566_INTERFACE.md) | BAR/寄存器/事务/安全 |
| [训练报告](docs/models/TRAINING_REPORT.md) | B01→B05P 训练 |
| [模型对比](docs/models/MODEL_COMPARISON.md) | 指标/延迟/选型 |
| [数据与评估](docs/models/DATASET_AND_EVALUATION.md) | split/口径/限制 |
| [模型清单](docs/models/MODEL_MANIFEST.md) | 路径/SHA256 |
| [RKNN 量化](docs/deployment/RKNN_QUANTIZATION_REPORT.md) | FP16/INT8/Raw-head |
| [板端 benchmark](docs/deployment/RK3566_BENCHMARK.md) | 实板 FPS/延迟/资源 |
| [部署指南](docs/deployment/DEPLOYMENT_GUIDE.md) | 构建/服务/故障 |
| [Tesla HMI](docs/hmi/TESLA_ADAS_HMI.md) | 车机/3D/接口 |
| [HMI 性能](docs/hmi/HMI_PERFORMANCE.md) | 渲染/瓶颈/指标 |
| [事件引擎计划](docs/roadmap/ADAS_EVENT_ENGINE_PLAN.md) | 跟踪/事件/FPGA 协议 |
| [需求矩阵](docs/competition/REQUIREMENT_MATRIX.md) | 官方要求逐项状态 |

---

## 10. License

No project-wide license has been selected. Do not assume permission to reuse code or data beyond GitHub's normal viewing rights. Third-party datasets, model weights, and toolkits retain their own terms.
