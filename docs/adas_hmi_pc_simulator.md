# PC ADAS 车载大屏感知可视化模拟器

这个子系统验证比赛产品的 **道路视频 → 检测元数据 → Scene Builder → 3D ADAS 驾驶场景 → 车载大屏 HMI** 链路。它只在 PC 上运行，不连接 FPGA、PCIe、RK3566/RK3568，也不修改现有冻结的比赛 AI pipeline。

## 1. 架构

```text
mock / pc-system(4ch) / surround(4ch BEV) / road.mp4 + JSONL / road.mp4 + PC YOLO or RKNN
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

五种输入模式：

```powershell
# A. 不依赖视频和模型的 20 秒演示
python tools/adas_hmi_sim/run.py --mode mock

# PC 整机模拟：四路同步相机、PCIe descriptor、RK3566 mock、HMI
python tools/adas_hmi_sim/run.py --mode pc-system

# 四路视频回放 + BEV 卡片（默认近似平面映射）
python tools/adas_hmi_sim/run.py --mode surround --surround-dir C:\path\to\four_cameras

# B. 视频 + 已保存检测结果，调 UI 时不重新跑 YOLO
python tools/adas_hmi_sim/run.py --mode replay --video road.mp4 --detections detections.jsonl

# C. PC .pt 使用现有 UltralyticsDetector；.rknn 使用 CompetitionDetector
python tools/adas_hmi_sim/run.py --mode live --video road.mp4 --model yolo11n.pt

# D. 前视主链路：真实前视视频 + 检测 + 跟踪 + 3D HMI
python tools/adas_hmi_sim/run.py --mode front `
  --video "D:\BaiduNetdiskDownload\紫光同创杯赛题三参考素材\video_20260929_112507.mp4" `
  --model .\yolov8n.pt --input-size 512x288 --detector-fps 12
```

`.rknn` 前视运行默认每 3 次检测更新一次 CPU 车道候选并复用中间结果；PC `.pt`
默认每次检测更新。可用 `--lane-stride` 显式冻结该取舍。

`pc-system` 是本项目面向整机演示的推荐入口。它生成 `front/left/right/rear`
四路同步相机预览，通过仓库已有的 `competition_io.FrameDescriptor` 和
`ResultDescriptor` 做内存中的 FPGA→PCIe→RK3566 模拟传输，再把 front 感知结果
交给同一个 Scene Builder 和浏览器 HMI。页面会显示 front 大图以及另外三路
channel 缩略图（默认折叠），并在 `SceneFrame.metadata` 中保留 `PC_FOUR_STREAM`、
`RK3566_SIM`、descriptor 顺序、队列深度和累计字节数。

这条路径验证的是 **软件边界、同步关系、序号回传和 HMI 接线**。它不代表真实
FPGA PCIe endpoint、DMA、RK3566 NPU 或 HDMI 已经连接；真实硬件仍需在板上单独
验收。

`surround` 模式读取四路视频，把 BEV JPEG 接入同一条 WebSocket。HMI 默认以视频素材
所示的 3D ADAS 驾驶场景为主画面：透视道路、车道、蓝色规划走廊、自车、目标车辆、
交通灯、限速/车速和转向提示都在同一个视图中；导航地图和环视 AVM 通过顶部按钮切换。
收到 BEV 后才会启用 `环视 AVM` 按钮。默认配置是 `APPROXIMATE_PLANAR`，用于快速验证画面和协议；有鱼眼内参/外参时，用
`tools/build_calibrated_surround_lut.py` 生成上游 Open-3D-Surround-View 风格 LUT，
再通过 `--bev-lut-dir` 运行标定版。RK3566 的实测、输入要求和性能边界见
[`docs/adas_hmi_surround_rk3566.md`](adas_hmi_surround_rk3566.md)。

导航地图使用开源 [MapLibre GL JS](https://github.com/maplibre/maplibre-gl-js) 适配器，作为 3D 感知驾驶页之外的可选地图页。
PC/RK 离线演示内置一小块哈尔滨 OSM 栅格瓦片，MapLibre 初始化失败时会回退到同一瓦片包
的 Canvas 渲染，因此不会因为外网、WebGL worker 或网络瓦片不可用而显示空白地图。
`?online-map=1` 可显式尝试公开 OSM 瓦片。当前路线和车辆位置是 SceneFrame 驱动的演示
overlay，不是 GPS 定位或路径规划；要做完整导航，需要再接入更大范围 PMTiles/MBTiles
地图包和 Valhalla/OSRM 路径服务。

`live` 仅复用已有适配器，不训练、不下载权重。`.pt` 输入走 `src/adas_edge/perception/ultralytics_detector.py`；`.rknn` 输入走 `competition_ai/detector.py`。`front` 是 live 的前视验收入口：默认 512×288、检测节流和前视专用 SceneFrame 元数据。没有模型时不会假装 live 已经完成，改用 mock 或 replay。

### Tesla-BevCam 真实前视验证

项目额外保留并接入了上游 [Tesla-BevCam-Clone](https://github.com/Ga0512/Tesla-BevCam-Clone)
作为真实前视视频验证和渲染入口。这个仓库是社区 MIT 项目，不是 Tesla 官方量产车机源码；
页面中的“Tesla-BevCam-Clone 原版”按钮直接调用上游
`render_tesla_ui`，对象和车道数据来自当前 `SceneFrame`；它不把车机屏幕录制当作相机输入。
先克隆上游仓库，再用比赛素材中的原始前视视频运行：

```powershell
git clone --depth 1 https://github.com/Ga0512/Tesla-BevCam-Clone.git .deps/Tesla-BevCam-Clone
python tools/run_tesla_bev_clone.py `
  --video "D:\BaiduNetdiskDownload\紫光同创杯赛题三参考素材\video_20260929_112507.mp4" `
  --model .\yolov8n.pt `
  --output .\outputs\tesla_bev_probe.mp4 `
  --frames 750
```

这个入口直接复用上游的 YOLOv8、单目深度、UFLDv2 车道和 BEV 跟踪/渲染代码；当前
Tesla checkout 的原生路径自动读取 `.deps/Tesla-BevCam-Clone/weights/ufldv2.onnx`。
没有该 ONNX 权重或导出头不匹配时，输出会明确显示车道检测关闭/丢失，不会把合成车道
或伪距离当作真实结果。生成的 MP4 是 PC 感知验证证据，不能替代
RK3566 NPU、FPGA 采集、PCIe DMA 或 HDMI 的板端验收。

当前浏览器“Tesla-BevCam-Clone 原版”视图使用 commit
`e99318bde342c665382950d46a3fc01b469b939a` 的 OpenCV BEV 渲染器，输出通过 WebSocket
JPEG 发送到页面。上游默认仍是灰色几何车体和蓝色路径，它本身不是官方 Tesla 资产，
但页面现在直接以这条上游链路为主视图；Apollo Carviz 只作为明确标注的对照实验。
该适配器目前只在 PC HMI 中启用，不改变 RK3566/FPGA 主链。

如果不需要网页，直接在上游仓库运行原生窗口：

```powershell
cd .deps\Tesla-BevCam-Clone
..\..\.venv\Scripts\python.exe main.py `
  --video "D:\BaiduNetdiskDownload\紫光同创杯赛题三参考素材\video_20260929_112507.mp4" `
  --model "..\..\yolov8n.pt" --native --loop --no-record --demo-motion
```

启动日志应出现 `UFLD: UFLDv2 ONNX ativo (800x320, CPU)`；原生 HUD 的
`LANE UFLD n/4` 是当前帧真正拟合出的车道曲线数量。该 ONNX 是 PC CPU 验证路径，
还没有把它宣称为 RK3566 的 RKNN/OM 性能结果。

这个窗口仍然使用原版 `render_tesla_ui()`，只在其 BEV 画布上增加前视小窗和键盘控制：
空格暂停、`R` 重播、`S` 截图、`Q/Esc` 退出。原版跟踪器现已按类别匹配，短暂漏检最多保留
5 帧，避免转弯时不同类别互相抢轨迹。自车在 BEV 中固定在底部是 ego-centric
坐标系的设计；没有速度/位姿输入时道路也不会流动。`--demo-motion` 会明确标记
`DEMO` 并演示一次平滑变道，不能当作视频估计出的真实轨迹；接入带有
`lateral_m`/`steering_deg` 的遥测 CSV 后才切换为 `TELEMETRY`。该改造已提交到上游
checkout 的 `841377d`，不依赖 `adas_hmi_sim` 网页服务。

### Apollo Dreamview 复用评估（第一轮）

Apollo 的 [Dreamview 官方说明](https://github.com/ApolloAuto/apollo/blob/master/modules/dreamview/README.md)
定位是依赖 Apollo 数据总线的 Web HMI：它接收
`LocalizationEstimate`、`Chassis`、`ADCTrajectory` 等 Protobuf 消息，前端使用
Three.js 绘制场景（[官方 FAQ](https://github.com/ApolloAuto/apollo/blob/master/docs/15_FAQS/Dreamview_FAQs.md)），并通过 Apollo 的构建和启动流程提供浏览器页面。它不是一个可以
直接链接进 RK3566 应用的独立 3D 渲染库。

因此当前结论是：PC 比赛演示可以借鉴或抽取 Dreamview 的 Three.js 场景组织方式，
但不能把完整 Dreamview/Cyber 运行时搬到 RK3566。先把 Tesla-BevCam 的检测、跟踪、
车道和路径输出收敛成项目自己的 `SceneFrame`，再评估只复用前端渲染层；FPGA 负责
采集、同步、缩放/格式预处理和 PCIe 传输，RK3566 负责推理与场景数据，渲染层保持
独立。这个边界要等 PC 输出稳定后再做板端移植验证。

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

`ObjectTracker` 做同类别 IoU 贪心匹配，track 失配最多保留 2 次 detector update。world X/Z 采用 EMA，默认 `alpha=0.35`，并用 `alpha` 字段做淡出。`front/live` 的 HMI 层还会按类别置信度、道路横向范围、伪距离和同类近邻做一次显示过滤，默认最多保留 24 个目标；它只影响 SceneFrame 显示，不修改原始 YOLO 结果或 RK/FPGA 协议。后端可以保持最近检测结果，而 UI 以 30 Hz 发送；mock 的检测输入约 20 Hz，live 默认把检测节流到 10 Hz。因此渲染频率不等于 YOLO 频率。右侧 telemetry 显示后端 WebSocket rate、检测更新 rate、目标数和轨迹数。

## 6. Lane、Traffic Light、Navigation

- lane 支持 metadata：`left/right.points`、`style=solid|dashed|unknown`、`confidence`；`front` 模式现在用 CPU OpenCV Hough/颜色候选生成这些点，并把 `lane_detector.source=OPENCV_HOUGH_EXPERIMENTAL` 和 `validation=EXPERIMENTAL` 写入 SceneFrame。没有 metadata 时才由 builder 生成稳定的左实线、右虚线 mock lane，并将 `lane_mock=true` 放入 metadata。
- traffic light 支持 `UNKNOWN/RED/YELLOW/GREEN`。mock 时间线包含红灯、绿灯，右侧灯箱和 action 同步变化。真实 `front` 模式会把 YOLO 的 traffic-light 框送入颜色分类和时域滤波，并同时写入 `traffic_light.state` 与 `metadata.traffic_light`；颜色状态只标为 `CANDIDATE`，因为仅凭前视框和颜色还不能证明它是控制本车车道的信号灯。运行时会把 `VEHICLE_OR_HORIZONTAL` 与 `PEDESTRIAN_OR_VERTICAL` 分开，后者写入 `pedestrian_state`，不会提升为车辆灯状态。`decision_state` 默认保持 `UNKNOWN`，所以不会把禁停牌、行人灯或横向信号误当作本车通行许可。PC 端可选的 `--signal-roi` 现在使用顶部 55%、横向 10%–90% 的 640×384 高分辨率支路，默认阈值 0.10；这是针对远处横向车辆灯的召回增强，不能把它的结果直接当作 RK3566 证据。另有 `--traffic-light-model` 专项支路（PC 用 `.pt`、RK3566 用 `.rknn`），按顶部 60% ROI、512×192、3 FPS 运行并把框映射回整帧；它是评测链路，不等于准确率已通过，本模拟器不重新训练 specialist。
- navigation 由 `configs/demo_route.json` 的时间区间产生 `STRAIGHT/TURN_LEFT/TURN_RIGHT/CHANGE_LEFT/CHANGE_RIGHT/ARRIVE` intent 和距离 overlay。它不是 GPS、地图或路径规划。红/黄灯约束为 `STOP`，行人/碰撞约束为 `WARNING`，实线变道约束为 `SOLID_LINE`。

## 7. Three.js 场景与 HMI

前端使用本地打包的 Three.js 0.180 依赖（`frontend/vendor/`，不依赖 CDN），车辆优先加载 Apollo Dreamview Carviz 的开源 `car.obj/car.mtl` 资源（`frontend/vendor/apollo/`），缺失时才回退到简单几何体。页面默认显示 `Tesla-BevCam-Clone 原版`：后端把当前真实前视检测和车道候选转成上游 `render_tesla_ui()` 的输入，直接返回其 OpenCV BEV 画面，而不是手写 Three.js 仿制场景。`Apollo Carviz 对照` 保留用于评估官方 Carviz 资源/坐标协议，`导航地图` 切换到 MapLibre/离线 OSM 栅格地图，不开启自由 OrbitControls。

驾驶感知视图左侧场景直接占满主工作区，叠加速度/限速、领航辅助状态、下一动作卡片、车道动作和底部车机 Dock；右下前视相机小窗来自当前 WebSocket 视频。自车采用车机常见的 ego-centric 视角，车体保持在画面下方，前进时车道线和导航带向车体流动，车道候选会提供显示用的转向角提示；这不是控制输出。`导航地图` 是可选地图页，`环视 AVM` 显示 BEV/鸟瞰画面。left/right/rear 三路 channel 默认折叠，点击 `AUX CAMS · S` 或按 `S` 才显示，用于调试和泊车/变道场景。按 `D` 打开 debug inspector，可查看 frame、track、bbox、world X/Z、lane、light 和 navigation。前视 bbox 是 PC 调试显示，不能解读为 FPGA 已经具备 OSD。

用户提供的参考素材 `video_20260929_111440.mp4` 已检查：1920×1080、30 FPS、约 100.5 秒，是车机屏幕录制的 3D ADAS 输出，不是四路原始相机视频。画面重点是“前向透视驾驶场景 + 感知目标和领航 UI”，因此不能把 OSM 地图当作它的主画面，也不能把该 MP4 当作 YOLO 输入直接拼接。

## 8. Demo timeline

`mock` 运行 20 秒并循环：0–4 秒双前车，4–8 秒红灯 STOP，8–11 秒绿灯 FORWARD，11–15 秒行人 WARNING，15–20 秒左转导航和左侧车道变化。它用于演示 UI 状态，不是道路安全验证。

真实前视专项支路可这样启动（PC）：

```powershell
python tools/adas_hmi_sim/run.py --mode front --video "D:\BaiduNetdiskDownload\紫光同创杯赛题三参考素材\video_20260929_112507.mp4" --model .\yolov8n.pt --signal-roi --signal-roi-confidence 0.05 --signal-roi-input-size 640x384 --signal-roi-top 0.55 --traffic-light-hold-frames 12
```

如果要验证已训练 specialist，把 `--signal-roi` 换成 `--traffic-light-model "D:\Datasets\adas_edge_training\tl_specialist\tl-512x192\weights\best.pt" --traffic-light-fps 3 --traffic-light-specialist-mode fuse`。默认是融合主模型候选；只有做消融时才使用 `--traffic-light-specialist-mode replace`。板端使用同一参数位置替换为 `tl_specialist_512x192_hybrid2.rknn`，不能把 PC 的 FPS 或颜色结果直接当作 RK3566 证据。

需要试验信号灯与自车车道的几何关联时，再追加 `--traffic-light-association geometry-experimental`。该模式要求当前帧同时有左右车道候选、信号框处在上方中央走廊，并连续确认后才填充 `decision_state`；它依赖未标定的 OpenCV 车道候选，默认不打开。当前窄片段对照中，关联决策 precision 为 `1.0000`、recall 为 `0.6341`（26/41），因此只能作为关联实验，不能替代标定/专项模型。

## 9. 性能与限制

当前记录的是 PC 浏览器和 Python backend 的运行速率，不能替代 RK3566/RK3568 或 FPGA 测量。使用补充审核的正负样本后，高分辨率 ROI/hold12 在这段窄片段上车辆绿灯为 `34/41`，precision `1.0000`、recall `0.8293`、F1 `0.9067`；行人红灯为 `10/13`，precision `1.0000`、recall `0.7692`、F1 `0.8696`。这些指标只覆盖人工审核的 169 帧，不是数据集级准确率，也不包含本车道归属。新支路的 169 帧墙钟约 7.72 FPS、感知均值约 99.27 ms，速度代价明显，当前作为 PC 评测/展示选项，不默认宣称已经是最终实时配置。场景不加载大型 GLTF、不用数据库、不依赖云服务。已知限制包括：伪距离没有真实标定、`front` 的 CPU lane 仍是实验性候选、`front` 的 traffic-light 颜色状态尚未完成本车车道归属、mock traffic light 不是 specialist 精度、JPEG over WebSocket 不是最终视频通道。

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
