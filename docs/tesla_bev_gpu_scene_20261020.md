# Tesla-BevCam-Clone GPU 空间视图恢复与升级

日期：2026-10-20。范围：RK3566 Qt `空间视图`，新增 Mali-G52 OpenGL ES 3.2 场景项；不改 FPGA RTL、PCIe 协议、YOLO/OCR 主链。

## 1. P0：原版视觉基线

原版审计对象：`.deps/Tesla-BevCam-Clone/main.py`

- 入口 `render_tesla_ui()`，相机 `BEV_W/BEV_H=800/720`、`BEV_CAM_Y=-10`、FOV 75°。
- 道路 `draw_clean_lanes_animated()`、`draw_fsd_path_gradient()`。
- 自车/车辆 `draw_solid_3d_vehicle()`，本质是 cuboid 加 roof plane；**原版没有可复用的精细自车 mesh**。
- 原版动画是程序化位移/镜头，不是真实自车里程。

实际运行基准截图：`results/p0_upstream_baseline.jpg`，来自已有真实上游渲染运行产物 `outputs/tesla_bev_probe_030.jpg`。它证明原版视觉基线是“道路 + 蓝色路径 + cuboid 车辆”，不是精模；因此本轮没有宣称“恢复原版自车模型”。

## 2. P1：GPU 场景

新增：

- `tesla_scene_item.h/.cpp`：`QQuickFramebufferObject` + OpenGL ES 3.2。
- `main.cpp` 注册 `TeslaSceneItem`。
- `main.qml` 空间视图改用 `TeslaSceneItem`，不再使用每帧 PNG + 双 Image 刷新。
- `assets/ego_car_rear.png`：项目已有的 Carviz 后视自车资产，QRC 内嵌。

实现：

- 一个动态 VBO 绘制道路/草地/路缘/车道线/城市剪影。
- 一个 billboard 纹理管线绘制周边车辆（车辆后视透明度纹理，带深度测试和 alpha）。
- 自车为 QML 后视纹理叠在 FBO 上，保证清晰可见；**这是后视 billboard，不是三维 mesh**。
- 不使用独立 EGL 程序句柄；GL 资源均在 `QQuickFramebufferObject::Renderer` 的 Qt 渲染线程创建/使用/释放。
- 不再通过 PNG 编码/解码刷新三维场景。

## 3. P2/P3：DEMO 与 LIVE

### DEMO

- 5 个合成交通目标沿车道连续移动；
- 道路虚线随 `elapsed` 移动；
- 自车固定，形成车后上方视角；
- 明确标记 `SCENE DEMO` / `DEMO · GPU 连续动画` / `演示目标 · 非真实检测`；
- 不把 DEMO 当成 LIVE。

### LIVE

- QML `boxes: video.aiBoxes`，`online: video.aiOnline`，`sourceKey: video.sourceMode`。
- C++ 使用 tracker 的 `trackId` 作为对象 key，source 改变时清空。
- 2D 框底边中心映射为**非标定相对位置**；不再用 bbox 高度作为精确深度。
- 目标 1.2 s 未更新删除；AI 离线清空。
- 明确标注 `LIVE TARGETS · GPU · 非实时几何` / `真实 Track ID · 坐标未标定`。
- 不输出米数、TTC、车速。

## 4. P5：Qt 车机不退化

保留：

- Qt Quick 主界面、顶部导航、四路缩略图；
- 真实 PCIe 接收路径（本次运行期间 PCIe 链路掉线，见第 6 节）；
- YOLO 框、时间戳跟踪、交通灯专项、导航 OCR、危险事件；
- 单路/四路/空间切换；
- HUD bridge。

## 5. 性能（DEMO，GPU）

15 秒运行 `results/gpu_scene_app.json`：

| 指标 | 值 |
|---|---:|
| QSG render callback FPS | **63.83** |
| GPU 场景项 | QQuickFramebufferObject / GLES 3.2 |
| Mali GPU 负载 | 55% |
| Qt 进程 CPU（全核百分比） | 26.60% |
| Qt 进程 RSS | 202.7 MB |
| 视频唯一帧率 | 0（PCIe 链路掉线，DEMO 不依赖视频） |
| YOLO FPS | 0（同上） |

DEMO 场景由 QSG 每帧驱动；GPU 场景不再受 Python 2–10 Hz 限制。

## 6. 真实 PCIe 状态

本轮开始前真实 PCIe 是通的：

- `01:00.0 [0755:0755]`
- `pango_video_capture` 约 53 FPS
- Qt `PCIe LIVE`，AI `PGV_LIVE`，HUD bridge `ext_applied` 递增。

随后 FPGA 端点变为：

```text
01:00.0 ... Device [0755:0755] (rev ff)
```

`/dev/pango_video0` 与 `/dev/shm/pango-video` 消失；`pango_video_capture` 报 `capture_io`。尝试卸载驱动、移除设备并 PCIe rescan 后端点没有重新出现。结论：**这是 PCIe 链路/FPGA 侧掉线，不是 Qt/场景代码问题；本轮 LIVE 无法在真实 PCIe 上完成验收。** 没有用本地 PGVF 冒充 PCIe LIVE。

## 7. 验收状态

实现：

1. 自车清晰可见：是，QML 后视资产。
2. 周围车辆有车身轮廓：是，纹理 billboard，非灰色立方体。
3. 道路透视与层次：有道路/草地/路缘/车道线/远景剪影。
4. DEMO 连续运动：是，GPU 每帧更新。
5. LIVE 根据真实感知更新：代码路径已接通；**真实 PCIe 硬件掉线，未完成实机验证**。
6. 不伪造距离/运动：LIVE 标记未标定；DEMO 标记演示。
7. 不再用低清 PNG 静态底图：是，空间视图改为 FBO。
8. Qt 主界面/AI 不退化：单路/四路界面未改；空间模式新渲染项独立。

限制：

- 自车是后视 billboard，不是三维 mesh；周边车辆同样是 billboard，尚未达到真正三维模型。
- 没有相机标定，LIVE 仍是相对示意；不输出准确米数。
- 当前 FPGA PCIe 链路掉线，LIVE 实机证据缺失。
- 没有做长时间压力测试。

## 8. 启动

```sh
cd /home/cat/qt_gpu_probe_ai_20261019
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
QT_QPA_PLATFORM=xcb QT_XCB_GL_INTEGRATION=xcb_egl \
./qt_gpu_probe --fps 30 --duration 0 --start-mode 2 \
  --road-quadrant 0 --pgv-file /dev/shm/pango-video/latest.pgv \
  --source-label pgv-live --pgv-layout quad \
  --ai-frame-file /dev/shm/qt_gpu_probe_ai.mfr1 --ai-hz 10
```

空间视图右上 `SCENE LIVE / SCENE DEMO` 切换；DEMO 用于视觉验收，LIVE 等待 PCIe 链路恢复。

## 9. 文件

- `deployment/lubancat/qt_gpu_probe/tesla_scene_item.h`
- `deployment/lubancat/qt_gpu_probe/tesla_scene_item.cpp`
- `deployment/lubancat/qt_gpu_probe/main.cpp`
- `deployment/lubancat/qt_gpu_probe/main.qml`
- `deployment/lubancat/qt_gpu_probe/qt_gpu_probe.pro`
- `deployment/lubancat/qt_gpu_probe/qml.qrc`
- `deployment/lubancat/qt_gpu_probe/assets/ego_car_rear.png`
- 结果：`results/p0_upstream_baseline.jpg`、`results/gpu_scene_demo.png`、`results/gpu_scene_app.json`
