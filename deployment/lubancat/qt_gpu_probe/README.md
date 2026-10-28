# RK3566 Qt Quick ADAS HUD / 真实 PCIe PGVF

Qt 5.12.8 + Qt Quick 的 1280×720 ADAS HMI 验证程序。它直接消费 PCIe 采集程序发布的 `PGVF` 快照，用 Mali-G52 / OpenGL ES 3.2 做单纹理 RGB565 硬件加速，并通过现有 UDP sidecar 接入 YOLO11n RKNN、交通灯专模与导航 OCR；同时把识别信号按 **PCIe HUD Control V2** 写回 FPGA BAR1，并提供上位机信号测试界面。

正式 EGL 程序、FPGA/PCIe 协议、RKNN 模型和驱动均未修改；本工程是独立探针，不替换正式车机入口。

![默认 Q2 主视 + AI HUD](results/qt_ui_default_q2.png)

## 已验证的硬件与数据链路

- LubanCat RK3566，Ubuntu 20.04.6 LTS arm64，Linux `4.19.232`，1280×720 HDMI/X11。
- Qt `5.12.8`，`QQuickFramebufferObject`，Mali-G52，OpenGL ES `3.2`，EGL vendor `ARM`。
- 实际加载 `/lib/aarch64-linux-gnu/libmali.so.1`、`libqxcb.so`、`libqxcb-egl-integration.so`；`physical_scanout_fps` 未测量，保持 `null`。
- PCIe 端点：`01:00.0`，驱动 `pango_video.ko`，设备 `/dev/pango_video0`。
- 采集程序：`/home/cat/pcie_test/pango_video_t05/pcie_video/app/pango_video_capture`。
- PGVF：40 字节头 + 1280×720 RGB565，payload `1,843,200` 字节，CRC32；Qt 读取与校验 CRC 后才发布到渲染线程。
- 当前真实画面/四宫格布局（以 `pcie_quad_raw.png` 为准）：
  - Q1 左上：前视道路；
  - Q2 右上：导航/地图 HDMI；
  - Q3 左下：车内摄像头；
  - Q4 右下：路口/后视道路。
- **默认主视改为 Q1（左上，前视道路）**；Q2 是导航/地图，不能作为道路 AI 输入。底部四张卡任意点击即可切换主视，四路模式显示完整 2×2。
- 导航识别源可独立选择：导航卡底部 `Q1/Q2/Q3/Q4` 四个芯片，不要求与主视相同。
- 主视和导航的选择通过 `MVS1` 控制包实时发给 AI sidecar，不需要重启 RKNN；Qt 立刻切换显示裁剪和检测框。
- 联调中曾观察到 Q1 前视源冻结（原始像素差 0.0，而 Q2/Q3/Q4 正常）；板卡重启/恢复采集后 Q1 已恢复为实时。若再次出现，先检查前视相机/HDMI/FPGA 通道，不要先怀疑 Qt。

![真实 PCIe 2×2 原始帧](results/pcie_quad_raw.png)

## 板端环境与构建

板端工程目录：`/home/cat/qt_gpu_probe_ai_20261019`。

```sh
cd /home/cat/qt_gpu_probe_ai_20261019
qmake qt_gpu_probe.pro
make -j2
```

板卡时钟未同步（系统显示 2024 年），从 Windows 拷贝文件后若 make 反复执行 qmake，
先 `touch` 源文件再构建：

```sh
touch *.cpp *.h *.pro *.qml qml.qrc assets/* components/*
qmake qt_gpu_probe.pro && make -j2
```

## 启动真实 PCIe 视频

先确认 FPGA 端点已供电且链路枚举成功：

```sh
lspci -nn | grep 01:00.0
sudo insmod /home/cat/pcie_test/pango_video_t05/pcie_video/driver/pango_video.ko
ls -l /dev/pango_video0
```

本次调试中板卡重启后链路恢复为 `PCIe Link up`。若仍是 `PCIe Link Fail`，先恢复 FPGA 位流/端点，不要用夹具冒充真实输入。

板端开机可能已经自动启动采集，输出到 `/dev/shm/pango-video`。先检查：

```sh
pgrep -af pango_video_capture
cat /dev/shm/pango-video/status.json
```

如果已有 publisher，不要再启动第二个（第二个会报 `capture_io`）。若没有，启动：

```sh
sudo /home/cat/pcie_test/pango_video_t05/pcie_video/app/pango_video_capture \
  --continuous --fps 60 --publish-dir /dev/shm/pango-video
cat /dev/shm/pango-video/status.json
```

本次实测 publisher 状态约 `fps=58.8`，`latest.pgv` 持续更新。

启动 Qt（真实 PCIe + AI）：

```sh
cd /home/cat/qt_gpu_probe_ai_20261019
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
QT_QPA_PLATFORM=xcb QT_XCB_GL_INTEGRATION=xcb_egl QSG_INFO=1 \
  ./qt_gpu_probe --fps 60 --duration 30 --start-mode 0 \
  --pgv-file /dev/shm/pango-video/latest.pgv --source-label pgv-live \
  --pgv-layout quad --road-quadrant 0 \
  --ai-frame-file /dev/shm/qt_gpu_probe_ai.mfr1 --ai-hz 5 \
  --output-json results/qt_real_pcie.json
```

### AI sidecar

必须使用本工程同目录的 `ai_bridge.py`，不要再用旧的
`/home/cat/competition_cockpit_port_20261018/deployment/lubancat/mali_video/ai_bridge.py`
（两者端口/状态字段不匹配）。

```sh
RUN=/home/cat/qt_gpu_probe_ai_20261019
ADAS_EDGE_ROOT=/home/cat/competition_cockpit_port_20261018 \
/usr/bin/python3 "$RUN/ai_bridge.py" \
  --live-frame-file /dev/shm/qt_gpu_probe_ai.mfr1 --source PGV_LIVE \
  --layout quad --display-view front --road-quadrant 0 \
  --navigation-quadrant 1 \
  --main-model /home/cat/adas-edge-rect/rect_models/yolo11n_rk3566_int8_split_512x288.rknn \
  --specialist-model /home/cat/adas-edge-rect/tl_check/tl_specialist_512x192_hybrid2.rknn \
  --main-hz 10 --hazard-hz 5 --specialist-hz 1 --ocr-hz 0.1 --idle-timeout 90 \
  --no-navigation-ocr \
  --summary "$RUN/results/ai_summary.json" \
  > "$RUN/results/ai.log" 2>&1 &
```

等日志出现 `READY` 后再启动 Qt。AI 结果通过 `MVA2` 返回；无结果超过 1.5 秒时 UI 自动清空检测框、交通灯、危险和导航状态并显示未连接。

### 自由选择主视 / 导航源

- **主视**：点击底部四张卡任意一张；被选卡显示 `MAIN`，主画面立即切到该象限，并给 sidecar 发 `MVS1` 更新检测裁剪。
- **导航源**：点击右侧导航卡底部的 `Q1/Q2/Q3/Q4`；选中芯片变琥珀色，并给 sidecar 发 `MVS1` 更新 OCR 裁剪。
- 默认 `主视=Q1`、`导航源=Q2`；主视和导航源可以不同。
- 四路页签仍显示完整 2×2；空间页签显示 Tesla-BevCam-Clone 视觉层。
- **右下角 ADAS 卡改为实时 BEV 投影**：用当前主视的 `aiBoxes` 画目标，主视切换后立即跟随，不再是一张静态 clone 卡。
- `MVS1` 包格式：`"MVS1" + uint8 road + uint8 navigation`，发到 UDP `38260`。

### PCIe HUD V2 回传与信号测试

协议使用 `PCIE_HUD_CONTROL_PROTOCOL_V2_20261018.json` 的 BAR1 HUD 窗口：`0x638` 感知字段、`0x63C/0x640` 时间、`0x644` commit、`0x648` 状态、`0x64C` applied seq；legacy `0x608` 保持 HUD/ADAS 可见。

Qt 进程不以 root 直接 mmap `resource1`，而是通过本机 root helper：

```sh
RUN=/home/cat/qt_gpu_probe_ai_20261019
cd "$RUN"
sudo nohup python3 hud_bridge.py \
  --bar1 /sys/bus/pci/devices/0000:01:00.0/resource1 --port 38262 \
  > results/hud_bridge.log 2>&1 &
cat results/hud_bridge.log
```

日志应出现 `READY ... magic=0x31445548 caps=0x000000ff`。Qt 里的 `HudControl` 监听 `127.0.0.1:38262`，每 500 ms 自动发送 AI 交通灯/行人快照；手动测试界面由主画面右上角 **“PCIe 信号测试”** 按钮或 `T` 打开。

- 自动字段：`perception_enable=1`、red/green 取 `aiTraffic`、pedestrian 取 person 检测数；其它字段置 0。
- 手动字段：红/绿灯、行人、静止目标、超速、LKA、左右车道、HUD/ADAS 使能；每个按钮发送一次。
- 时间字段默认不发：板端系统时钟未同步；黄灯协议没有字段，不伪造。
- legacy 使用 `state=0x101`（HUD + ADAS 可见）并递增 sequence。
- 实测：手动红灯 `ext_request=0x00000003`、绿灯+行人 `0x0000000d`、自动行人 `0x00000009`，`ext_applied` 持续递增、`error=0`。
- HUD overlay 在 FPGA 的 HDMI 输出端合成，**不会出现在上游 PGVF 视频里**；因此验收看 BAR1 applied seq / status，而不是截取 PGVF 画面。

![PCIe HUD V2 测试界面](results/hud_test_panel.png)

### 常用参数

| 参数 | 作用 |
|---|---|
| `--pgv-file PATH` | 开启真实 PGVF 输入（空则使用 replay / 生成源） |
| `--source-label pgv-live\|pgv-fixture\|replay` | UI 来源标签；夹具必须标 `pgv-fixture` |
| `--pgv-layout front\|quad` | 原始帧布局；本次真实源为 `quad` |
| `--road-quadrant 0..3` | Q1=0、Q2=1、Q3=2、Q4=3；默认 Q1=0（前视道路），可在 UI 自由切换 |
| `--start-mode 0\|1\|2` | 单路/四路/空间视图 |
| `--capture-cycle` | 仅测试用，每 5 秒自动切换模式，便于一次抓三张图 |
| `--ai-frame-file PATH` | `MFR1` 快照路径；必须与 sidecar 一致 |
| `--ai-hz N` | 向 sidecar 发请求的频率 |
| `--hazard-hz N` | 危险事件分析频率；`0` 跟随 YOLO，生产配置为 5 Hz |

## 当前性能

| 负载 | Qt 源 FPS（末窗） | 不同 PCIe 帧被 GL 接受 | QSG 回调 | AI | 上传 p95 |
|---|---:|---:|---:|---:|---:|
| 真实 PCIe，无 AI | 60.3 | 51.4/s | 60.3/s | off | 1.89 ms |
| 真实 PCIe + AI 5 Hz | 51.5 | 46.6/s | 56.7/s | 4.86 Hz | 2.45 ms |
| 真实 PCIe + AI 3 Hz | 56.1 | 49.4/s | 61.2/s | 3.0 Hz（末窗 0.76） | 2.33 ms |
| 真实 PCIe + AI 5 Hz，Q1 截图档（旧） | 51.3 | 44.2/s | 51.3/s | 4.92 Hz | 3.13 ms |

当前常驻桌面配置的专项复测（单路主视、真实 PCIe、`QT_FRAME_METADATA_HZ=15`）如下。这里的 `AI 请求` 是 Qt 向 sidecar 请求结果的频率，sidecar 内部 YOLO 主模型仍按 `--main-hz 10` 配置运行。

| 配置 | QSG 回调 | AI 结果 | 说明 |
|---|---:|---:|---|
| 无 AI | 约 45 FPS | 0 | Qt 基线 |
| AI 请求 5 Hz，sidecar `Nice=10` | 约 42–44 FPS | 约 4.7–5.0 Hz | 基本不影响 Qt，推荐稳定档 |
| AI 请求 10 Hz，sidecar 默认优先级 | 约 30–35 FPS | 约 9.3–9.8 Hz | YOLO 上限，Qt 明显降速 |
| AI 请求 10 Hz，sidecar leader `Nice=10` | 约 30–36 FPS | 约 9.5 Hz | 只改善调度竞争，仍未回到基线 |

同一配置的单次结果会随 PCIe publisher、NPU/GPU 频率和窗口负载波动；以上是当前板端多次复测范围，不是物理显示器的 scan-out FPS。

- 采集 publisher 本身约 **58.8 fps**；旧布局 Qt 单路 scene graph 约 **51–61 fps**。新增 LiveBev + PCIe 测试面板后的截图档实测约 **20–29 fps**，不能用旧表当作新布局指标。
- `GL 接受` 只表示 `glTexSubImage2D` 无错误，不等于物理 HDMI scanout；`physical_scanout_fps` 保持 `null`。
- Qt 使用一个复用 RGB565 纹理、一次上传；producer 读取直接写入预分配帧池，CRC 使用 zlib，避免每帧临时整帧 copy。
- 渲染线程直接拉取共享帧队列，QML 不再为每个 PCIe 帧排一次 GUI 事件；静态卡片用 `layer.enabled` 缓存，目标框使用轻量边框和标签图元。
- AI sidecar 在持续高频率推理时可能变慢或停止返回（约 15–20 秒后出现）；5 Hz 截图档可见，但持续运行也可能变慢，必要时重启 sidecar。UI 会在 1.5 秒后如实显示离线，不保留陈旧框。

## AI 结果与诚实边界

![默认 Q2 主视](results/qt_ui_default_q2.png)

![点击 Q1 后的主视与检测框](results/qt_ui_selected_q1.png)

![右下角实时 BEV 跟随主视](results/qt_ui_live_bev_q1.png)

![Tesla-BevCam-Clone 空间视图](results/qt_ui_tesla_spatial.png)

- 历史截图默认主视是 Q2（右上），当时 Q2 被配置为道路；当前布局 Q1=前视、Q2=导航，默认已改为 Q1。点击底部卡片可切换主视，AI 检测框跟随道路象限。
- 右下角“ADAS 空间识别”卡使用当前主视的 AI boxes 做实时 BEV 投影，选中 Q1 时能看到对应目标；不再使用一张静态图片。
- 底部卡状态：`MAIN` 是主识别源，`NAV` 是导航 OCR 源，`SNAP` 是其它快照。
- 交通灯仍可能 `UNKNOWN`；导航 OCR 仍可能 `UNREADABLE`；没有数据时显示“未识别/设计预览”，不编造。
- 危险事件为 0 时显示监控态；不伪造 TTC。
- 四路页签是实时的 2×2；单路主视也是实时的，取决于所选象限。
- **Tesla-BevCam-Clone 接入**：空间主视图使用 clone 的 BEV 渲染序列（已做暗色适配）。上游完整流程需要 Ultralytics/torch/depth，RK3566 上没有重跑整条上游；因此 clone 背景标“视觉演示/非实时几何”。右下角 ADAS 卡是**另一套实时 BEV 投影**，用当前主视 AI boxes 绘制，不是 clone 的静态图。
- 识别信号通过 BAR1 V2 写回 FPGA；自动发送的交通灯/行人字段来自当前主视 AI，切换主视后同步变化。
- Q1 曾经出现过一次输入冻结，重启后恢复；如果再次出现，按硬件输入源排查。

## 测试工具

- `tools/capture_one.sh OUTPUT`：抓当前 Qt 窗口（不是根窗口）。
- `tools/capture_three.sh`：在 `--capture-cycle` 运行时依次抓单路/四路/空间。
- `tools/verify_mode_switch.py OUTPUT_DIR`：XTest 点击三个页签并抓窗口；坐标为 1280×720 新版。
- `tools/click_mode.py X Y`：不退出程序，直接点击运行中 HUD 的某个坐标；用于选择主视/导航源或切页签。
- `tools/run_pgv_no_ai.sh`：真实 PGVF、无 AI 的 18 秒性能与截图。
- `tools/run_real_pgv_ai.sh [seconds]`：真实 PGVF + AI 的启动/截图。
- `tools/run_pgvf_probe.sh`：**PGVF 夹具**测试，UI 标 `pgv-fixture`，不能冒充真实 PCIe。
- `tools/pgv_to_png.py latest.pgv out.png`：检查当前原始帧/象限布局。
- `tools/pgvf_fixture_publisher.py`：写合法 PGVF 测试流，仅用于解析/性能回归。
- `tools/run_hud_bridge.sh`：以 root 启动 `hud_bridge.py`（BAR1 V2）。
- `tools/hud_bar1_probe.py`：只读 BAR1 magic/caps/status/applied seq，需 root。
- `tools/hud_bar1_write_test.py`：一次性 V2 + legacy 写测试，需 root。
- `tools/hud_udp_test.py`：不经过 Qt，向 bridge 发一条红/绿灯+行人测试快照。

## 本地素材回放（当前无 FPGA / 无输入）

参考素材在 `D:\BaiduNetdiskDownload\紫光同创杯赛题三参考素材`（1080p30 HEVC）。本地测试时把它们组成 1280×720 的 2×2：

- Q1：`video_20260929_113421.mp4`（前视道路）
- Q2：`video_20260929_111440.mp4`（导航屏，用于导航卡测试）
- Q3：`video_20260929_112739.mp4`（车内/仪表）
- Q4：`video_20260929_112454.mp4`（路口）

已生成板端文件：

```text
/home/cat/qt_gpu_probe_ai_20261019/local_quad.mp4      1280x720 30fps 合成视频
/home/cat/qt_gpu_probe_ai_20261019/local_quad.rgb      240 帧 RGB565LE（约 422 MB）
```

启动（不需要 FPGA、不需要 pango_video_capture）：

```sh
cd /home/cat/qt_gpu_probe_ai_20261019
python3 pgv_local_publisher.py --raw local_quad.rgb \
  --output /dev/shm/pgv_local/latest.pgv --fps 30 --source-fps 30 &

ADAS_EDGE_ROOT=/home/cat/competition_cockpit_port_20261018 \
/usr/bin/python3 ai_bridge.py \
  --live-frame-file /dev/shm/qt_gpu_probe_ai.mfr1 --source LOCAL_REPLAY \
  --layout quad --display-view front --road-quadrant 0 --navigation-quadrant 1 \
  --main-model /home/cat/adas-edge-rect/rect_models/yolo11n_rk3566_int8_split_320x192.rknn \
  --specialist-model /home/cat/adas-edge-rect/tl_check/tl_specialist_512x192_hybrid2.rknn \
  --main-hz 10 --specialist-hz 1 \
  > results/ai_local.log 2>&1 &

DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
QT_QPA_PLATFORM=xcb QT_XCB_GL_INTEGRATION=xcb_egl \
./qt_gpu_probe --fps 30 --duration 0 --start-mode 0 --road-quadrant 0 \
  --pgv-file /dev/shm/pgv_local/latest.pgv --source-label pgv-local \
  --pgv-layout quad --ai-frame-file /dev/shm/qt_gpu_probe_ai.mfr1 --ai-hz 10 \
  --output-json results/local_final3.json
```

注意：导航 OCR 默认就是**关闭**，只有加 `--navigation-ocr` 才启用；本地性能启动命令不要带这个参数。

本地性能（15 秒报告）：

| 场景 | 源 FPS（末窗） | QSG | 上传 p95 | AI 结果年龄 |
|---|---:|---:|---:|---:|
| 本地 PGVF，无 AI | 30.3 | 60.6 | 1.67 ms | — |
| 本地 PGVF + 512 模型 10 Hz | 30.9 | 56.7 | 2.20 ms | 2 ms |
| 本地 PGVF + 416 模型 10 Hz | 23.7 | 59.2 | 2.64 ms | 140 ms |
| 本地 PGVF + 320 模型 10 Hz（当前） | 24–29 | 约 59 | 约 2.6 ms | 约 118–140 ms |

- 参考素材本身是 30 fps，所以本地视频内容最多 30 fps；QSG 接近 60。不要宣称本地回放是 60 fps 内容。
- 320/416 模型降低延迟但交通灯/远小目标精度下降；512 模型精度更好但 AI 状态约 3–5 fps。
- **导航 OCR 默认关闭**：Q2 卡片 OCR 在 RK3566 上单次约 10 s，会抢占 CPU 并让源 FPS/框延迟变差。需要时加 `--navigation-ocr`，并接受掉帧；本地性能模式不启用。
- **生产 AI cadence**：YOLO 主模型保持 10 Hz，危险事件分析独立为 5 Hz，避免每个 YOLO 结果都重复执行全帧运动/跟踪计算；这不会降低检测框更新频率。
- 图像/时序证据：`results/local_noai.json`、`local_ai30.json`、`local_ai416.json`、`local_final3.json`。

## 已知限制

- FPGA/PCIe 链路在冷启动/端点未就绪时会失败；本次通过板卡重启恢复，未改驱动。
- 采集 publisher 是测试程序，不是生产服务；60 fps 状态会随输入/负载波动。
- Qt AI-on 的 `QSG` 为约 51–61 fps，PCIe 帧上传约 44–49/s，尚未做到严格 60 帧唯一视频流；无 AI 时 QSG 稳定 60.3。
- 交通灯、导航 OCR、车道/危险语义仍受模型和输入限制；没有数据时显示预览/未知。
- Q1 前视源曾出现一次冻结，板卡重启后恢复；若再次遇到，检查前视相机/HDMI 源、FPGA 前视通道和输入时钟/格式。Qt 侧无法让静态源产生运动；可切到四路确认其它输入是否刷新。
- Tesla clone 空间主画面是**渲染序列 + 真实 ADAS 计数叠加**，不是由当前帧 AI 几何实时重建；上游模型未在 RK3566 上运行。右下角 ADAS 卡才是实时 AI 投影。
- 右下角 LiveBev 是**示意性透视投影**（用 2D box 底边/尺寸估算），不是标定地面模型；不能用于测距验收。
- PCIe HUD V2 helper 需要 root；HUD overlay 在 FPGA HDMI 输出端，不在上游 PGVF 内。自动发送只覆盖交通灯/行人，静止目标/超速/LKA/车道需手动测试；时间字段因板端时钟未同步默认不发，黄灯协议无字段。
- 板卡系统时钟未同步，截图右上角日期为板端时间。
- 本轮没有修改正式 EGL、FPGA 协议、模型或驱动；只新增独立 BAR1 helper 和 V2 写路径。

## 结果文件

- `results/qt_ui_default_q2.png`：默认 Q2 主视 + AI 框 + MAIN/NAV 标记。
- `results/qt_ui_selected_q1.png`：点击底部 Q1 后的主视 + AI 框。
- `results/qt_ui_live_bev_q1.png`：右下角 ADAS 实时 BEV 跟随 Q1 的盒子。
- `results/hud_test_panel.png`：PCIe HUD V2 手动信号测试界面。
- `results/hud_pcie_v2_validation.json`：红/绿+行人/自动行人的 BAR1 写回证据。
- `results/qt_ui_tesla_spatial.png`：Tesla-BevCam-Clone 空间主视图。
- `results/qt_ui_optimized_pcie.png`：此前 Q1 单路 + AI 截图（旧布局，保留）。
- `results/qt_ui_optimized_quad.png`：四路截图。
- `results/qt_ui_optimized_spatial.png`：旧空间示意截图（已被 Tesla 版本替代）。
- `results/pcie_quad_raw.png`：真实 PCIe 2×2 原始帧。
- `results/qt_ui_optimized_pcie.json`：单路截图当次 Qt 报告。
- `results/qt_ui_optimized_5hz.json` / `qt_ui_optimized_3hz.json`：AI 5/3 Hz 性能。
- `results/qt_ui_no_ai.json`：真实 PCIe、无 AI 性能。
- `results/qt_ui_optimized_cycle.json`：一次自动切换三模式的 20 秒报告。

原始窗口抓取格式为 XWD，最终交付只保留 PNG；重新抓取使用 `capture_one.sh` 或 `verify_mode_switch.py`。
