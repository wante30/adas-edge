# PC ADAS 车载大屏感知可视化模拟器

这个子系统验证比赛产品的 **道路视频 → 检测元数据 → Scene Builder → Three.js 感知世界 → ADAS HMI** 链路。它只在 PC 上运行，不连接 FPGA、PCIe、RK3566/RK3568，也不修改现有冻结的比赛 AI pipeline。

## 1. 架构

```text
mock / pc-system(4ch) / road.mp4 + JSONL / road.mp4 + PC YOLO or RKNN
                         │
                         ▼
          backend.video_pipeline + competition_io mock
                         │ detections + video frame
                         ▼
                 backend.scene_builder
              pseudo world + lanes + tracking
                         │ SceneFrame JSON
                         ▼
                 FastAPI WebSocket /ws
                         │
                         ▼
                 Three.js browser renderer
```

浏览器永远只认识 `SceneFrame`，不认识 YOLO tensor、RKNN 输出或检测器内部类。前视视频以 JPEG 预览随 WebSocket 发送，最终 FPGA 架构仍然可以让 RK 返回 metadata、FPGA 做 OSD。

## 2. 启动

在仓库根目录执行：

```powershell
python tools/adas_hmi_sim/run.py --mode mock
```

默认监听 `http://127.0.0.1:8765/` 并打开浏览器。服务端若不能自动打开浏览器，可以加 `--no-browser` 后手动访问该地址。

四种输入模式：

```powershell
# A. 不依赖视频和模型的 20 秒演示
python tools/adas_hmi_sim/run.py --mode mock

# PC 整机模拟：四路同步相机、PCIe descriptor、RK3566 mock、HMI
python tools/adas_hmi_sim/run.py --mode pc-system

# B. 视频 + 已保存检测结果，调 UI 时不重新跑 YOLO
python tools/adas_hmi_sim/run.py --mode replay --video road.mp4 --detections detections.jsonl

# C. PC .pt 使用现有 UltralyticsDetector；.rknn 使用 CompetitionDetector
python tools/adas_hmi_sim/run.py --mode live --video road.mp4 --model yolo11n.pt
```

`pc-system` 是本项目面向整机演示的推荐入口。它生成 `front/left/right/rear`
四路同步相机预览，通过仓库已有的 `competition_io.FrameDescriptor` 和
`ResultDescriptor` 做内存中的 FPGA→PCIe→RK3566 模拟传输，再把 front 感知结果
交给同一个 Scene Builder 和浏览器 HMI。页面会显示 front 大图以及另外三路
channel 缩略图（默认折叠），并在 `SceneFrame.metadata` 中保留 `PC_FOUR_STREAM`、
`RK3566_SIM`、descriptor 顺序、队列深度和累计字节数。

这条路径验证的是 **软件边界、同步关系、序号回传和 HMI 接线**。它不代表真实
FPGA PCIe endpoint、DMA、RK3566 NPU 或 HDMI 已经连接；真实硬件仍需在板上单独
验收。

`live` 仅复用已有适配器，不训练、不下载权重。`.pt` 输入走 `src/adas_edge/perception/ultralytics_detector.py`；`.rknn` 输入走 `competition_ai/detector.py`。没有模型时不会假装 live 已经完成，改用 mock 或 replay。

## 3. SceneFrame

`backend/schemas.py` 定义稳定 JSON：

```json
{
  "frame_id": 1002,
  "timestamp_ms": 1234,
  "ego": {"speed": 42, "steering": 0, "action": "FORWARD"},
  "lanes": {"left": {"points": [{"x": -1.8, "z": 0}], "style": "solid"}},
  "objects": [{
    "id": 3, "class_name": "car", "confidence": 0.91,
    "bbox": [420, 170, 580, 360],
    "world": {"x": -1.2, "z": 11.5},
    "velocity": {"vx": 0, "vz": 0}, "alpha": 1,
    "pseudo_distance": true
  }],
  "traffic_light": {"state": "RED", "confidence": 0.8},
  "navigation": {"action": "STRAIGHT", "distance_m": 200},
  "warnings": ["RED_LIGHT"],
  "metrics": {"object_count": 1},
  "source": "mock"
}
```

这份协议可直接保存为 JSONL，也可以替换 WebSocket 为后续的二进制传输。建议未来的 RK→FPGA metadata 使用小端固定头：`magic u16`、`version u8`、`frame_id u32`、`timestamp_ms u32`、`object_count u8`、`traffic_light u8`、`lane_state u8`、`action u8`，之后每个目标使用 `track_id u16 + class u8 + confidence u8 + bbox 4×u16 + world_x/world_z 2×i16`。单位、缩放和最大目标数应在 FPGA/RK 联调前冻结，不能把这个模拟协议当成已经验收的 PCIe ABI。

## 4. Pseudo-world 算法

`SceneBuilder.bbox_to_world` 取检测框 bottom-center `(u, v)`。它用可配置 `fx/fy/cx/cy`、类别典型高度和相机高度估计深度：

```text
z_size   = fy × class_height / bbox_height
z_ground = fy × camera_height / (bottom_y - cy)
z        = height_weight × z_size + (1 - height_weight) × z_ground
x        = (u - cx) × z / fx
```

结果经过合理范围裁剪并带有 `PSEUDO_DISTANCE` 标记。它是 **HMI visualization approximation**，不是高精度真实测距，不能代替相机标定、IPM、深度网络或雷达。更换真实标定或雷达时只需要替换 Scene Builder，不需要修改 Three.js。

## 5. Tracking 与帧率

`ObjectTracker` 做同类别 IoU 贪心匹配，track 失配最多保留 4 次 detector update。world X/Z 采用 EMA，默认 `alpha=0.35`，并用 `alpha` 字段做淡出。后端可以保持最近检测结果，而 UI 以 30 Hz 发送；mock 的检测输入约 20 Hz，live 默认把检测节流到 10 Hz。因此渲染频率不等于 YOLO 频率。右侧 telemetry 显示后端 WebSocket rate、检测更新 rate、目标数和轨迹数。

## 6. Lane、Traffic Light、Navigation

- lane 支持 metadata：`left/right.points`、`style=solid|dashed|unknown`、`confidence`；没有 metadata 时由 builder 生成稳定的左实线、右虚线 mock lane，并将 `lane_mock=true` 放入 metadata。
- traffic light 支持 `UNKNOWN/RED/YELLOW/GREEN`。mock 时间线包含红灯、绿灯，右侧灯箱和 action 同步变化。本模拟器不重新训练 specialist。
- navigation 由 `configs/demo_route.json` 的时间区间产生 `STRAIGHT/TURN_LEFT/TURN_RIGHT/CHANGE_LEFT/CHANGE_RIGHT/ARRIVE` intent 和距离 overlay。它不是 GPS、地图或路径规划。红/黄灯约束为 `STOP`，行人/碰撞约束为 `WARNING`，实线变道约束为 `SOLID_LINE`。

## 7. Three.js 场景与 HMI

前端使用本地打包的 Three.js 0.180 依赖（`frontend/vendor/`，不依赖 CDN），场景全部由程序化几何体组成：自车、道路平面、实线/虚线车道、车辆、行人、悬浮交通灯和轻量警告状态。默认相机在自车后上方、朝道路前方看，保留固定驾驶辅助视角，不开启自由 OrbitControls。

页面左侧约 60% 是 3D perception world；右侧以 front sensor video 为主，下面是 traffic light/action/warning、navigation intent 和 telemetry。left/right/rear 三路 channel 默认折叠，点击 `AUX CAMS · S` 或按 `S` 才显示，用于调试和泊车/变道场景。按 `D` 打开 debug inspector，可查看 frame、track、bbox、world X/Z、lane、light 和 navigation。前视 bbox 是 PC 调试显示，不能解读为 FPGA 已经具备 OSD。

## 8. Demo timeline

`mock` 运行 20 秒并循环：0–4 秒双前车，4–8 秒红灯 STOP，8–11 秒绿灯 FORWARD，11–15 秒行人 WARNING，15–20 秒左转导航和左侧车道变化。它用于演示 UI 状态，不是道路安全验证。

## 9. 性能与限制

当前记录的是 PC 浏览器和 Python backend 的运行速率，不能替代 RK3566/RK3568 或 FPGA 测量。场景不加载大型 GLTF、不用数据库、不依赖云服务。已知限制包括：伪距离没有真实标定、mock lane 不是 FPGA 车道输出、mock traffic light 不是 specialist 精度、JPEG over WebSocket 不是最终视频通道。

## 10. RK 与 FPGA 迁移

目标迁移路径是：

```text
PC Python detector → RKNN detector on RK3566/RK3568
Browser Three.js/WebGL → RK-side Chromium/WebGL or native renderer
SceneFrame metadata → optional RK→FPGA metadata
HMI panel → RGB565 framebuffer (for example 640×360)
→ FPGA layer compositor → PG2L100H → 1280×720 HDMI
```

本次没有实现 PCIe、DMA、framebuffer 或 HDMI，也没有宣称它们已经连通；这份模拟器只把最终 panel 的内容与 metadata 边界在 PC 上提前验证。
