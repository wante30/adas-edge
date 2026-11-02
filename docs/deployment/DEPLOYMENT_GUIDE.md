# 部署指南（RK3566 + Windows 开发机）

更新日期：2026-10-28  
适用：LubanCat-1 RK3566 实板、Windows 开发机、当前 `live/stage2` 源码。  
原则：**不要直接改动正在运行的生产服务**；先在独立目录/隔离进程验证，再替换。

---

## 1. 当前板端部署拓扑

```text
FPGA (PG2L100H, PCIe endpoint 0755:0755)
  │  PCIe Gen2 x1 (actual; IP supports x4)
  ▼
RK3566 pango_video.ko → /dev/pango_video0
  │  pango_video_capture --continuous --fps 60
  ▼
/dev/shm/pango-video/latest.pgv  (PGVF: 40B header + 1280x720 RGB565 + CRC32)
  │
  ├─ Qt qt_gpu_probe  ← /dev/shm/qt_gpu_probe_ai.mfr1 (MFR1 snapshot)
  │     │  MVR1 request / MVA2 result over UDP 38260/38261
  │     ▼
  │   ai_bridge.py (RKNN, Raw-head, hazard, OCR)
  │     │  HUD V2 UDP 38262
  │     ▼
  │   hud_bridge.py (root) → BAR1 0x638... → FPGA OSD/LED
  │
  └─ 四宫格显示 + Tesla BEV + 检测框
```

当前实际运行目录：

```text
/home/cat/qt_gpu_probe_ai_20261019/
  qt_gpu_probe
  ai_bridge.py
  b05p_ai_bridge_launcher.py
  b05p_competition_bridge.py
  model_profiles.py
  hud_bridge.py
  start-pcie-hmi-gui.sh (实际在 /home/cat/.local/bin/)
  results/
```

---

## 2. 板端服务

### 2.1 `pango-video-capture.service`

```ini
[Unit]
Description=FPGA PCIe video capture to PGVF
After=local-fs.target
Before=adas-ai-bridge-live.service

[Service]
Type=simple
ExecStartPre=/bin/sh -c 'test -e /dev/pango_video0 || /sbin/insmod /home/cat/pcie_test/pango_video_t05/pcie_video/driver/pango_video.ko'
ExecStart=/home/cat/pcie_test/pango_video_t05/pcie_video/app/pango_video_capture --continuous --fps 60 --publish-dir /dev/shm/pango-video
Restart=always
RestartSec=2
```

检查：

```sh
systemctl status pango-video-capture --no-pager
cat /dev/shm/pango-video/status.json
ls -l /dev/pango_video0
lspci -nn | grep 01:00.0
```

期望：

```json
{"state":"running","frames":...,"fps":49.5,"error":"","last_success_ns":...}
```

### 2.2 `adas-ai-bridge-live.service`

比赛默认选型确定为 **B：B05N P2 十类单模型**。`tools/zz-b05p-model-B-single.conf` 是对应的比赛启动配置；目前 RK3566 生产 service 仍运行 A，尚未安装该 drop-in。危险事件开发先用隔离测试进程，不改生产 unit 配置。

当前板端的 1280×720 PCIe 四路拼图中，道路画面位于 Q4（零基索引 3）；B 启动配置使用 Q4，Qt 的 MVS1 选择消息仍可在运行时改选其他象限。2026-10-29 首次 B 实测错误地使用索引 1（桌面画面），因此那次 0 框不是模型精度结论。修正后，对实时 MFR1 帧 445865 使用 B 模型及 RGA 预处理，正式 raw-head 解码在 conf=0.25、NMS IoU=0.45 下输出 9 框（车辆、自行车、行人、交通标志）；模型 SHA256 为 `65089ef0697bd7378071290baa218905cdcaed768452678da440b8eb116b2a5f`。这是单帧检测证据，不代表连续跟踪或危险事件已通过实测。

当前 drop-in：

```text
/etc/systemd/system/adas-ai-bridge-live.service.d/99-b05p-model-A.conf
/etc/systemd/system/adas-ai-bridge-live.service.d/zz-b05p-model-A-single.conf
```

最终生效的 `ExecStart`（后者覆盖前者）：

```sh
/usr/bin/python3 /home/cat/qt_gpu_probe_ai_20261019/b05p_ai_bridge_launcher.py \
  --live-frame-file /dev/shm/qt_gpu_probe_ai.mfr1 \
  --source PGV_LIVE --layout quad --display-view front \
  --road-quadrant 0 --navigation-quadrant 1 \
  --main-model /home/cat/cityscapes_arch_compare_20261027/models/bdd_yolov8n_512x288_raw_heads_int8.rknn \
  --main-hz 10 --hazard-hz 5 --ocr-hz 0.1 --navigation-ocr \
  --idle-timeout 90 \
  --timing-log /home/cat/qt_gpu_probe_ai_20261019/results/adas_A_live_default_timing.jsonl \
  --summary /home/cat/qt_gpu_probe_ai_20261019/results/adas_A_live_default_summary.json
```

启动器 `b05p_ai_bridge_launcher.py` 会安装 `b05p_competition_bridge.py` 的 Raw-head adapter，并强制 `"A"` 方案。  
检查：

```sh
systemctl status adas-ai-bridge-live --no-pager
ps -eo pid,args | grep b05p_ai_bridge_launcher | grep -v grep
cat /home/cat/qt_gpu_probe_ai_20261019/results/adas_A_live_default_summary.json
```

### 2.3 `adas-hud-bridge.service`

```ini
[Service]
ExecStart=/usr/bin/python3 /home/cat/qt_gpu_probe_ai_20261019/hud_bridge.py
Restart=always
RestartSec=2
User=root
```

检查：

```sh
systemctl status adas-hud-bridge --no-pager
sudo python3 /home/cat/qt_gpu_probe_ai_20261019/hud_bar1_probe.py
```

期望：

```text
HUD_MAGIC = 0x31445548
HUD_VERSION = 0x00010000
HUD_CAPS = 0x000001ff
HUD_REJECT_COUNT = 0
HUD_LAST_ERROR = 0
```

### 2.4 Qt HMI

不是 systemd 服务；由 XFCE autostart 启动：

```text
~/.config/autostart/adas-pcie-hmi.desktop
  Exec=/home/cat/.local/bin/start-pcie-hmi-gui.sh
```

脚本核心：

```sh
RUN=/home/cat/qt_gpu_probe_ai_20261019
PGV=/dev/shm/pango-video/latest.pgv
AI_FRAME=/dev/shm/qt_gpu_probe_ai.mfr1
"$RUN/qt_gpu_probe" \
  --fps 60 --duration 0 --start-mode 0 \
  --pgv-file "$PGV" --source-label pgv-live \
  --pgv-layout quad --road-quadrant 0 \
  --ai-frame-file "$AI_FRAME" --ai-hz 10
```

环境：

```sh
DISPLAY=:0
XAUTHORITY=/home/cat/.Xauthority
QT_QPA_PLATFORM=xcb
QT_XCB_GL_INTEGRATION=xcb_egl
QT_FRAME_METADATA_HZ=15
```

---

## 3. 模型文件与切换

### 3.1 模型目录

```text
/home/cat/cityscapes_arch_compare_20261027/models/
  bdd_yolov8n_512x288_raw_heads_int8.rknn          A（当前生产）
  b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn  B（比赛选型）
  b05l_v8n_p2_raw_heads_int8_500_20261026.rknn     C/D Specialist
```

### 3.2 方案

| ID | 主模型 | Specialist | 说明 |
|---|---|---|---|
| A | BDD Raw-head INT8 | 无 | 当前生产运行，保留回退 |
| B | B05N P2 Raw-head INT8 | 无 | 比赛选型与独立验证目标 |
| C | BDD | B05L | 交通灯专模 |
| D | B05N P2 | B05L | 双模型 |

注意：2026-10-27 的 B 合成 bridge benchmark 曾将道路帧写入零基索引 1，却请求索引 0。其吞吐率不能作为检测精度或正例覆盖证据；当前 benchmark 代码已让回放写入位置和推理象限共用同一常量，修正后的测试结果需单独记录。

切换代码：

- `model_profiles.py`：路径/输入/类别映射
- `ai_bridge.py`：`MVM1` 请求、`MMS1` 回复、active_id
- `video_item.cpp` / `main.qml`：模型选择面板

**已知问题**：2026-10-28 截图右上角显示“方案 B”，但进程参数/summary 为 A。切换/显示状态需要修，不要只看 UI 标签判断实际模型。

### 3.3 hash 校验

```sh
sha256sum /home/cat/cityscapes_arch_compare_20261027/models/*.rknn
```

期望：

```text
a86ab053...041afb  bdd_yolov8n_512x288_raw_heads_int8.rknn
65089ef0...b2a5f  b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn
c11b95b4...38b6f  b05l_v8n_p2_raw_heads_int8_500_20261026.rknn
```

---

## 4. 构建 Qt（板端）

```sh
cd /home/cat/qt_gpu_probe_ai_20261019
qmake qt_gpu_probe.pro
make -j2
```

板卡时钟未同步时，`make` 可能反复执行 qmake；先：

```sh
touch *.cpp *.h *.pro *.qml qml.qrc assets/* components/*
qmake qt_gpu_probe.pro && make -j2
```

运行：

```sh
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
QT_QPA_PLATFORM=xcb QT_XCB_GL_INTEGRATION=xcb_egl QSG_INFO=1 \
./qt_gpu_probe --fps 60 --duration 0 --start-mode 0 \
  --pgv-file /dev/shm/pango-video/latest.pgv --source-label pgv-live \
  --pgv-layout quad --road-quadrant 0 \
  --ai-frame-file /dev/shm/qt_gpu_probe_ai.mfr1 --ai-hz 10
```

> 仓库 `deployment/lubancat/qt_gpu_probe/` 是源码；板端目录中的 `main.qml`、`video_item.cpp` 与仓库有漂移。重新部署前用仓库版本覆盖，并保留板端 `.bak`。

---

## 5. Windows 开发机

### 5.1 Python 环境

```powershell
cd C:\Users\JingyuWang\adas-edge
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
```

本次实测：**559 passed**（78.01 s）。

### 5.2 PC 原生车机测试台

```powershell
cd C:\Users\JingyuWang\adas-edge
.\.venv\Scripts\python.exe tools\run_competition_console.py `
  --video 'D:\BaiduNetdiskDownload\紫光同创杯赛题三参考素材\video_20260929_113539.mp4' `
  --paused
```

或双击 `run_competition_console.cmd`。

这是 **PC 测试台**：视频文件 + ONNX Runtime + 软件 HUD 接收，不是 FPGA PCIe 实时 HMI。

### 5.3 PCIe 调试网页

板端启动：

```sh
python3 /home/cat/qt_gpu_probe_ai_20261019/pcie_protocol_service.py \
  --port 38264 --web-host 0.0.0.0 --web-port 8080 --token '<random>'
```

Windows Chrome：

```text
http://<board-ip>:8080/?token=<redacted>
```

安全要求：

- 必须设置 token；不要暴露无认证 MMIO API。
- 只允许文档化寄存器；DMA/PIO 写被阻止。
- BAR 读回全 `FF` / magic 不匹配时，禁止写。

---

## 6. 常见故障

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `status.json` state=error, error=capture_io | 第二个 capture 进程 / FPGA 断链 / 设备节点丢失 | `pgrep -af pango_video_capture`；只保留一个；检查 `lspci`、`/dev/pango_video0` |
| `/dev/pango_video0: No such device` | 驱动未加载/端点掉链 | `sudo insmod .../pango_video.ko`；检查 FPGA 位流/供电 |
| `lspci` 01:00.0 `rev ff` / BAR `ffffffff` | PCIe 链路异常 | 不要写 BAR1；重启/恢复端点后再试 |
| HUD magic 不是 `0x31445548` | 位流/窗口不匹配 | 停止 HUD 写入；核对 FPGA 工程版本 |
| AI 显示离线 | sidecar 崩溃/未启动/端口占用 | `systemctl status adas-ai-bridge-live`；查 `results/ai*.log` |
| UI 显示“方案 B”但实际 A | 模型状态显示/切换回读不一致 | 以进程参数/summary/hash 为准；修 `ai_model_id_` 状态同步 |
| 导航 OCR `valid=0` | 画面没有距离/路名 / OCR 超时 | 检查 `NAV_OCR_RESULT` 日志；不要伪造导航 |
| 红绿灯“未识别” | HSV 置信不足/无关联灯组/模型漏检 | 检查检测框、ROI、颜色阈值；不要强制给绿灯 |
| 板端时间 2024 | RTC 未同步 | 不影响相对时序；报告中注明 |

---

## 7. 安全与回滚

- 不运行第二个 `pango_video_capture`。
- 不在 BAR1 读回 `0xffffffff` 时写寄存器。
- 不直接改正在运行的 systemd 服务；用 drop-in + `daemon-reload`，并保留 `.bak`。
- 回滚模型：保留旧 `yolo11n_rk3566_int8_split_512x288.rknn` + `tl_specialist_512x192_hybrid2.rknn`。
- 回滚 Qt：保留板端 `qt_gpu_probe.bak`、`main.qml.bak`、`video_item.cpp.bak`。
- 不上传板端 token、SSH 私钥、个人配置。

---

## 8. 复现清单

- [ ] `lspci` 01:00.0 枚举，驱动 `pango_video`
- [ ] `/dev/pango_video0` 存在
- [ ] `status.json` fps > 0，error 为空
- [ ] `adas-ai-bridge-live` active，summary 有结果
- [ ] `hud_bridge` active，HUD magic/caps 正确
- [ ] Qt 启动，主视有实时画面
- [ ] 检测框与视频同帧（源帧号/时间戳）
- [ ] 模型 hash 与 manifest 一致
- [ ] 记录 CPU/NPU/GPU/温度/内存
- [ ] 记录“已验证/未验证”边界，不把 mock/回放写成实机
