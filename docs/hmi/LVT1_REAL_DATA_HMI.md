# LVT1 真实数据驱动 HMI 升级

日期：2026-10-29  
分支：`feature/lvt1-ego-bev-20261028`  
范围：RK3566 Qt 车机、LVT1 车道/本车控制、YOLO 目标驱动 Tesla BEV。  
状态：**代码完成；Qt 5.12.8 在实板编译通过；离线 fixture + YOLO 目标截图通过；未替换生产二进制，未做真实 LVT1 硬件验收。**

---

## 1. 本次完成的内容

| 功能 | 状态 | 代码位置 |
|---|---|---|
| LVT1 只读解码/一致性读取 | 完成（离线） | `vehicle_telemetry_v1.py` |
| LVT1 原子 JSON → Qt 状态 | 完成 | `vehicle_telemetry_status.h/.cpp` |
| 车道感知卡（左右线型/数量/来源/有效性） | 完成 | `main.qml` `laneCard` |
| 动态车道线型图 | 完成 | `laneCanvas` |
| 本车控制卡（油门/刹车/左右变道/允许/拒绝/警告/原因） | 完成 | `main.qml` `egoCard` |
| Tesla 3D 道路边界按 LVT1 实线/虚线绘制 | 完成 | `tesla_scene_item.cpp` `laneEdge` |
| 真实 YOLO 目标驱动 3D 车辆/行人 | 完成 | `tesla_scene_item.cpp` `addLive` |
| 目标 ID 稳定、过期移除 | 完成 | Qt `video_item.cpp` tracker + `TeslaSceneItem` live map |
| 非道路象限隔离 | 完成 | `displayView`/`roadQuadrant` 映射 + 默认 Q1 |
| 危险卡中性态、导航紧凑态 | 完成 | `main.qml` |
| 离线 fixture 发布器 | 完成 | `vehicle_telemetry_fixture.py` |
| 单元/契约测试 | 完成 | `tests/test_vehicle_telemetry_v1.py`、`tests/test_lvt1_hmi_contract.py` |

---

## 2. 数据路径

```text
真实：FPGA BAR1 0x690..0x6bc
  → vehicle_telemetry_v1.py（只读 ioctl + generation 一致性）
  → /dev/shm/adas_vehicle_status.json（原子替换，schema pango.rk3566.vehicle_status.v1）
  → VehicleTelemetryStatus（Qt，150ms 轮询，400ms 本地新鲜度）
  → laneState / egoState
  → QML 车道卡 / 本车控制卡
  → TeslaSceneItem laneState → 3D 道路边界线型

离线：vehicle_telemetry_fixture.py（同 schema，simulated=true）
  → 同一 JSON 路径
  → 同一 Qt 代码路径
```

真实数据优先：Qt 只认新鲜文件；文件停止更新即离线，不保留旧许可。fixture 明确标记 `simulated=true`，UI 显示 `SIM`。

---

## 3. 车道感知卡

字段：

- 来源：`HSST / RK / NONE`
- 左最近边界：实线/虚线/未知
- 右最近边界：实线/虚线/未知
- 左右边界数量：`null` → `—`；有效 0 → `0`
- 有效性：在线/过期/离线
- 明确提示：**边界数量 ≠ 车道数量**

动态 `Canvas`：

- 左线按 `left_nearest_type` 绘制连续/虚线；
- 右线按 `right_nearest_type` 绘制连续/虚线；
- unknown 用稀疏点线，不冒充实线/虚线；
- 不生成 LVT1 未提供的像素坐标。

---

## 4. 本车控制卡

显示：

- 油门请求
- 刹车请求
- 左变道请求
- 右变道请求
- FPGA 允许/拒绝/警告/空闲
- 原因码及中文解释
- 在线/过期/离线

规则：

- 当前动作只取 `current_action`，不把 `last_action` 当当前操作；
- `brake_allow` 只表示“允许”，不表示物理制动已执行；
- 离线/过期时所有 allow 归零，显示未知；
- 允许绿色、拒绝红色、警告橙色、空闲灰蓝。

---

## 5. Tesla BEV 真实目标

- 复用 Qt 侧 `video_item.cpp` 已有跟踪输出（`trackId`、`missed`、`predicted`、`vx/vy`），不新增第三套跟踪器。
- `TeslaSceneItem` 只接收 `!predicted && missed==0 && confidence>=0.20` 的目标；预测目标不画，避免幽灵车。
- 目标丢失后跟踪器 900ms 内移除；3D live map 900ms 过期，和跟踪器一致。
- 目标 ID 稳定时更新同一 `LiveCar`，不反复创建/销毁。
- 检测类别映射：car/truck/bus/person/motorcycle/bicycle。
- 无检测时 `live_` 清空，不生成模拟周围车辆。
- `demoMode: false` 显式关闭 demo 车辆；只有显式 demo 模式才有固定车流。
- 3D 场景固定道路/建筑仍是**示意几何**，不冒充真实车道/标定。

坐标：

```text
检测框局部坐标（display_view=front）→ 1280×720
cx = (x + w/2) / 1280
bottom = (y + h) / 720
tx = clamp(-1.75 + (cx-0.5)*14, -6.8, 6.8)
tz = clamp(0.72*widthDepth + 0.28*bottomDepth, 4, 62)
```

- 没有相机标定/单应矩阵，因此只输出 `RELATIVE_ONLY`；
- 不显示米级距离；
- 若未来提供标定，可替换这段映射。

非道路象限：

- AI status 携带 `display_view`、`road_quadrant`；
- `display_view=quad` 时，TeslaSceneItem 减去象限偏移，导航/车内象限目标不能进入道路 BEV；
- 默认道路象限改为 **Q1（前视）**，Q2 是导航，不再默认把导航当道路。

---

## 6. 3D 车道线型

- 左道路边缘按 `left_nearest_type`：solid → 连续线；dashed → 虚线段；unknown → 稀疏点线。
- 右道路边缘同理。
- 只改已有示意道路的边缘样式，不新增车道、不虚构坐标。
- 修复了 chase camera 的 X 镜像：`projection.scale(-1,-1,1)`，保证屏幕左=车辆左，LVT1 左右不互换。
- 3D 页脚标注 `SCHEMATIC ROAD / LVT1 STYLE` 和 `RELATIVE_ONLY`。

---

## 7. 离线验证

### 7.1 编译

```text
LubanCat RK3566 / Qt 5.12.8 / g++ 9
qmake qt_gpu_probe.pro && make -j2
结果：成功生成 qt_gpu_probe
```

### 7.2 测试

```text
pytest tests/test_vehicle_telemetry_v1.py tests/test_lvt1_hmi_contract.py
25 passed
```

覆盖：

- 官方 LVT1 handoff 示例解码；
- stale/offline/零数量/未知类型；
- `last_action` 不当当前动作；
- brake warning 不是物理制动；
- fixture 场景；
- QML/C++ 契约（注册、绑定、RELATIVE_ONLY、默认 Q1）。

### 7.3 实板 UI 截图

使用：

- 本地 60 帧 `local_quad_60.rgb` → `pgv_local_publisher.py`；
- 第二个 AI sidecar（端口 38280/38281，A 模型 5Hz）；
- `vehicle_telemetry_fixture.py`；
- 独立测试目录，不替换生产二进制。

截图：

| 模式 | 文件 |
|---|---|
| 单路 + 真实 YOLO 目标 | `docs/hmi/assets/lvt1/single_live_targets_20261029.png` |
| 四路 + 真实 YOLO 目标 | `docs/hmi/assets/lvt1/quad_live_targets_20261029.png` |
| 空间 + 真实 YOLO 目标 | `docs/hmi/assets/lvt1/spatial_live_targets_20261029.png` |
| 左变道请求 + 左实线拒绝 | `docs/hmi/assets/lvt1/single_left_change_denied_20261029.png` |

截图中的 `SIM` 表示 LVT1 是 fixture；场景是本地视频而非 PCIe。  
截图中的 AI 目标是真实 RKNN 推理结果（本地视频），但不是当前生产 PCIe 输入。  
测试时生产 `pango-video-capture` 已处于 `capture_io`（`/dev/shm/pango-video/status.json` state=error），因此没有可用的真实 PCIe 帧；本次严格使用本地 PGVF fixture，没有把 fixture 写成实板 PCIe。

---

## 8. 性能（测试环境，不是生产）

测试环境同时运行：

- 生产 Qt / 生产 AI（未停止）；
- 测试 Qt；
- 测试 AI sidecar；
- 本地 PGVF publisher。

因此测试 Qt 的 FPS 低于生产，不能作为生产性能。

单路测试 JSON（`docs/deployment/evidence/lvt1_fixture_mode_0_20261029.json`）：

```text
source_fps ≈ 20.5
Qt scene-graph callback ≈ 10.2 FPS
AI result ≈ 4.36 FPS
NPU 100%
GPU 81%
process CPU ≈ 15.4% of all cores
RSS ≈ 191.8 MB
duration ≈ 16.4 s
```

测试 AI summary（`docs/deployment/evidence/lvt1_test_ai_summary_20261029.json`）：主模型结果率约 4.07 Hz。  
原因：两个 AI sidecar 共用 NPU；本地 publisher + 双 Qt；不是生产 10Hz/56FPS 配置。  
生产基线仍是 2026-10-28 的 49.49 FPS 视频 / 9.98 Hz AI / 56 FPS HMI；**本次未部署，未重测生产**。

---

## 9. 部署说明（未执行）

- 生产目录 `/home/cat/qt_gpu_probe_ai_20261019` 未被覆盖；
- 测试目录 `/home/cat/qt_gpu_probe_lvt1_dev_20261028` 独立；
- 未修改 FPGA RTL/内核模块/模型；
- 未停止生产服务；
- 未替换生产二进制。

部署前必须：

1. 获取 FPGA 队友的 LVT1 驱动/vehiclemon 源码和位流；
2. 备份生产二进制/QML/服务 drop-in；
3. 在测试目录验证真实 LVT1 ioctl；
4. 逐步替换 `adas-ai-bridge-live.service` 和 Qt autostart；
5. 保留回滚：旧二进制、旧 QML、旧 drop-in。

---

## 10. 真实 LVT1 验收清单

- [ ] `/dev/pango_pci_driver` 存在，ioctl 0xC008530B 可读 0x690–0x6bc
- [ ] MAGIC/VERSION/CAPS 匹配
- [ ] `vehicle_telemetry_v1.py --once` 输出 schema
- [ ] generation 连续读取 g0==g1
- [ ] 左右实线/虚线变化 → Qt 卡 + 3D 边缘同步
- [ ] 数量变化 → 卡片数字同步，不增车道
- [ ] 左变道请求 + 左实线 → 拒绝 + 原因 2
- [ ] 刹车请求 + reason 8 → 警告，不声称物理制动
- [ ] 断链/停更 >400ms → Qt 离线，allow 归零
- [ ] 确认 HUD auto-send 不会与生产 bridge 冲突

---

## 11. 未来 FPGA 车道几何字段

当前 LVT1 只有“左右数量 + 最近类型”，没有像素坐标/曲率/车道 ID。若后续要画真实车道几何，建议新增：

```text
lane_geometry_version u16
lane_count u8
per lane:
  lane_id u8
  role u8 (ego_left/ego_right/adjacent/...)
  type u8 (solid/dashed/double/unknown)
  confidence_q8 u8
  point_count u8
  points: N × (x_q8, y_q8)  # 归一化图像坐标，或
  polynomial: c0/c1/c2/c3 (Q格式) + y_range
  timestamp_ns u64 / frame_id u64
```

要求：

- 明确坐标系（源图像 1280×720 / ROI / BEV）；
- 明确时间戳/帧号；
- 明确有效性/过期；
- 不要用数量推导车道 ID；
- 不要在没有标定时输出米级距离。

---

## 12. 相关文件

- `deployment/lubancat/qt_gpu_probe/vehicle_telemetry_status.h/.cpp`
- `deployment/lubancat/qt_gpu_probe/vehicle_telemetry_v1.py`
- `deployment/lubancat/qt_gpu_probe/vehicle_telemetry_fixture.py`
- `deployment/lubancat/qt_gpu_probe/main.qml`
- `deployment/lubancat/qt_gpu_probe/tesla_scene_item.h/.cpp`
- `deployment/lubancat/qt_gpu_probe/video_item.h/.cpp`
- `deployment/lubancat/qt_gpu_probe/ai_bridge.py`
- `tests/test_vehicle_telemetry_v1.py`
- `tests/test_lvt1_hmi_contract.py`
- `docs/deployment/LVT1_VEHICLE_TELEMETRY.md`
---

## 13. PCIe / 本地回放切换修复（2026-10-29）

问题：`source_controller.cpp/.h` 仍存在，但 `main.cpp` 在后续重写中丢失了 `sourceSwitch` 注入和 `setVideo(video)`，顶部“输入源”卡片只剩状态显示，切换失效。

修复：

- `main.cpp` 恢复 `SourceSwitchController` 注册、context property、`setVideo(video)`。
- `main.qml` 输入源卡片增加“切换”按钮，`z: 950` 防止被遮挡；键盘 `O` 同样可切换。
- 增加 `SourceSwitchController::message`：本地 PGV 不存在时显示“本地回放未准备”，不会黑屏冒充。
- 新增 `tools/run_local_replay.sh`，用于启动本地 PGVF publisher。

实测日志：

```text
SourceSwitch toggle mode "pgv-local"
SourceSwitch apply "/dev/shm/pango-video/latest.pgv" "pgv-live" video true
SourceSwitch toggle mode "pgv-live"
SourceSwitch apply "/dev/shm/pgv_local/latest.pgv" "pgv-local" video true
```

截图：

- PCIe 路径（当前采集不可用，显示测试图）：`docs/hmi/assets/lvt1/source_switch_pcie_20261029.png`
- 切回本地回放：`docs/hmi/assets/lvt1/source_switch_local_20261029.png`

这证明按钮/O 键、`sourceSwitch.toggle()`、`VideoItem::startPgvProducer()` 实际工作；不是只改 UI。
