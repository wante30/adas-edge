# Tesla-BevCam-Clone 空间视图：实时接入与三维观感优化

日期：2026-10-20。范围：Qt `空间视图`、原 `.deps/Tesla-BevCam-Clone` 渲染函数和 Qt 适配层。没有修改 YOLO、OCR、FPGA RTL 或 PCIe 协议。

## 1. 审计结论：原 Qt 空间视图并没有调用原渲染器

审计路径：

- 原实现：`.deps/Tesla-BevCam-Clone/main.py`
  - 投影/相机：`BEV_W, BEV_H = 800, 720`、`BEV_CAM_Y=-10`、`FOV_BEV=75°`、`project_bev()`。
  - 道路：`draw_clean_lanes_animated()`、`draw_fsd_path_gradient()`。
  - 车辆：`draw_solid_3d_vehicle()`，本质上仍是 cuboid 加一个深色 roof plane。
  - 入口：`render_tesla_ui(tracked, ego_state, offset_y, lane_polys)`。
- PC 端适配：`tools/adas_hmi_sim/backend/tesla_bevcam_renderer.py`，确实调用 `render_tesla_ui`，但 Qt 板端没有走这个适配。
- Qt 端此前：
  - `main.qml` 的空间视图中央是 **`AnimatedImage assets/tesla_bev_demo.gif`**；
  - `components/LiveBev.qml` 只是底部 `ADAS 空间识别` 小 Canvas；
  - 因此旧截图里的灰色方块是**预先录制的演示 GIF**，不随当前 Track ID / 当前检测更新。

这是“自车不明显、车辆乱窜”的首要原因：旧的中央画面根本不是实时场景。

## 2. 本轮改动

### 2.1 实时场景桥

新增：

- `deployment/lubancat/qt_gpu_probe/tools/tesla_scene_renderer.py`
- Qt `VideoItem`：
  - `spatialViewActive`；
  - 进入空间视图时，把当前 tracker 的稳定 `trackId / class / bbox / confidence / detect timestamp / state` 写入 `/dev/shm/tesla_scene_tracks.json`；
  - 其他视图不写、不启动场景。
- 渲染器独立进程读取最新 JSON，调用上游 `render_tesla_ui` 的投影/道路/相机逻辑，输出 `960×540` 场景。
- Qt 用双缓冲 `Image`（A/B）异步刷新；每次新帧 Ready 后才切换，**避免单文件重载造成的黑帧**。
- 场景输出约 5 Hz（缓存静态道路基底、目标 LOD、720×405 + 双缓冲；实测 3 秒 15 次写帧）。

### 2.2 自车 P0

- 保留上游 ego 投影位置 `(0,0,0)`，相机在车后上方，自车固定在画面下方中央。
- 自车尺寸单独放大 1.24×，增加：
  - 下车身、倾斜前后轮廓、车顶、后窗、侧窗；
  - 四个车轮、尾灯条、深色阴影；
  - 中性灰蓝车漆和青色轮廓。
- 修复了原上游 ego 为深色 cuboid、几乎与背景融合的问题。
- 场景标题/未标定说明由 Qt 覆盖层提供，避免重复 chrome。

### 2.3 周边车辆 P1

在保留上游低多边形投影的前提下，重写车辆绘制：

- `car`：侧面轮廓挤出（前舱/车顶/后窗/车尾），玻璃、车轮、尾灯。
- `truck`：驾驶室 + 货箱两段，带车轮和驾驶室窗。
- `bus`：长车身 + 连续侧窗带。
- `person`：头部/躯干/腿部简化轮廓，不是方块。
- 类别尺寸继续使用上游 `DIMS`；普通目标灰白/浅蓝灰，没有随机红色风险色。

### 2.4 目标乱窜 P2

- 场景对象以 Qt `trackId` 为唯一 key；同一 ID 持续复用同一个场景对象。
- `source` 改变时清空场景目标。
- 使用带时间戳的轨迹状态，按 `dt` 做：
  - 指数平滑；
  - 横向 ±5 m/s、纵向 ±14 m/s 的位置限幅；
  - 新增目标从当前检测位置进入；
  - 1.6 s 未再出现才删除。
- AI 超过 1.5 s 没有新 tracks：
  - 清空所有目标；
  - 仍然输出一帧“仅自车”场景，避免 QML 冻结在旧画面。
- 没有把二维检测框当作精确世界坐标；Qt 和渲染器均明确标记“空间示意 / 未标定”。
- 不显示上游的 `FRENTE: xx m`、`HDG/ACC/GPS` 等数字；不输出 TTC、真实距离或速度。

### 2.5 道路/镜头 P3

- 保留上游透视、车道线和 FSD 蓝色路径，背景改为深蓝黑，蓝色饱和度降低。
- 增加路缘/侧边线、低对比地面网格、暗角和远处尺度衰减。
- 相机保持车后上方，自车不随视频帧移动。
- 没有自车里程计/车道标定，因此道路是**静态示意背景**，不是重建车道。

### 2.6 动态视觉 P4

- 自车固定，周边目标相对平滑变化。
- 场景更新与 Qt 视频纹理解耦（独立进程 + 2 Hz）。
- 没有可靠自车运动输入，不使用道路滚动或假里程。
- 单路/四路/空间切换保留；空间模式进入/退出由 `spatialViewActive` 控制。

## 3. 视觉对比

- 原版空间视图（静态 GIF）：`results/mode_spatial.png`
- 动态空间视图：`results/final_opt3.png`、`results/final_opt10.png`
- 前后并排：`results/scene_before_after_opt.png`
- 10 帧、约 1 秒间隔：`results/scene_opt_sequence10.png`
- AI 暂停后仅自车、目标清空：`results/pause2.png`

原版：没有自车，灰色方块悬浮，只有蓝色道路和静态演示目标。

优化后：自车固定在下方中央，有车顶/后窗/尾灯/车轮；周边车辆为不同轮廓的 car/truck/bus/person；目标随真实 `trackId` 更新；地面网格和路缘提供空间感。

同一段本地 PGVF；不是 FPGA PCIe LIVE。

## 4. 场景与 Qt 性能

15 秒空间视图实测：

| 指标 | 值 |
|---|---:|
| QSG 回调 FPS | **51.85** |
| 场景写帧率（3 秒统计） | **5.0 Hz**（15 次/3 s） |
| 唯一源帧上传 FPS 平均 | 6.44 |
| YOLO FPS | 7.30 |
| Mali GPU 负载 | 72% |
| NPU 负载 | 100% |
| Qt 进程 CPU（全核百分比） | 31.5% |
| 场景渲染进程 CPU | 64.34% 单核 |
| 场景渲染进程 RSS | 53.9 MB |
| 四核平均负载 | 80.1 / 93.8 / 84.3 / 91.9% |

结论：

- **QSG 仍为 58.8**，空间场景没有拖垮 Qt 合成。
- 场景实际约 **5 Hz**；每帧基础道路/地面已缓存，只重画车辆，并按距离使用 LOD。CPU 单核约 64%。仍不是 60 Hz 的流畅动画，但相比最初的 2 Hz 已提升约 2.5 倍。
- 空间模式下唯一视频上传平均只有 6.44 FPS；单路/四路不受场景渲染影响，但因为避免 AI 断流，空间模式下 VideoItem 仍保持运行，源视频上传会被场景进程抢占。
- 没有做 20 分钟长测。

## 5. 稳定性验证

- 10 帧序列显示目标持续存在，自车固定，没有整帧黑屏。
- AI 停止 3 秒后：`pause2.png` 只剩自车，所有周边目标清空，说明不会在 AI 暂停时无限外推。
- 双缓冲 QML：旧帧保留到新帧 Ready，不再出现单文件重载黑帧。
- 场景进程通过 `/dev/shm/tesla_scene_tracks.json` 的 mtime 判断 AI 停止；超过 1.5 s 清空。
- 同一 Track ID 在 JSON 和渲染器场景对象中复用；source 改变时清空。
- 仍存在的限制：
  - 场景约 5 Hz，目标运动仍有低频跳步；
  - 深度由检测框高度近似，未标定；
  - 目标没有真正的透明度淡出（1.6 s 后删除，不是 alpha 渐隐）；
  - 车辆仍是程序化低多边形，不是高精度车模；
  - 场景道路是静态示意，不是车道重建；
  - 没有真实 PCIe / FPGA 输入。

## 6. 启动

```sh
# 场景渲染器（独立进程）
ADAS_EDGE_ROOT=/home/cat/competition_cockpit_port_20261018 \
python3 tesla_scene_renderer.py --rate 5 &

# Qt；进入空间视图后会自动置 active
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
QT_QPA_PLATFORM=xcb QT_XCB_GL_INTEGRATION=xcb_egl \
./qt_gpu_probe --fps 30 --duration 0 --start-mode 2 --road-quadrant 0 \
  --pgv-file /dev/shm/pgv_local/latest.pgv --source-label pgv-local \
  --pgv-layout quad --ai-frame-file /dev/shm/qt_gpu_probe_ai.mfr1 --ai-hz 10
```

## 7. 修改文件

- `deployment/lubancat/qt_gpu_probe/video_item.h`
- `deployment/lubancat/qt_gpu_probe/video_item.cpp`
- `deployment/lubancat/qt_gpu_probe/main.qml`
- `deployment/lubancat/qt_gpu_probe/tools/tesla_scene_renderer.py`
- `deployment/lubancat/qt_gpu_probe/tools/perf_sampler.py`（增加 scene 角色）
- 结果：`results/scene_before_after_opt.png`、`results/scene_opt_sequence10.png`、`results/live_final.png`、`results/pause2.png`、`results/scene_perf*.json`
