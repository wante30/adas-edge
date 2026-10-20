# Tesla-BevCam-Clone LIVE 第二阶段：连续目标、三维车型与性能回归

日期：2026-10-20。分支：`live/stage2`。不改 YOLO/RKNN、FPGA RTL、PCIe 协议和整体 HUD。

## 1. P0：主视频性能 A/B

同一真实 PCIe Q1、同一 AI（主模型 10 Hz、专项 1 Hz、OCR 0.1 Hz），25 s：

| 指标 | 场景关闭 | 场景开启 |
|---|---:|---:|
| 唯一视频帧上传 FPS | **14.96** | **13.72** |
| source FPS（窗口） | 25.16 | 25.64 |
| QSG callback FPS | 50.31 | **57.69** |
| YOLO FPS | 8.99 | 8.97 |
| Mali GPU | 25% | 36% |
| Qt CPU（全核百分比） | 31.45% | 28.85% |

结论：

- Tesla 场景对唯一视频上传的影响约 **-8%**，不是 11–15 FPS 的主因。
- 场景关闭时也只有 **14.96 FPS**，低帧率主要来自 AI/OCR 竞争和视频消费链路。
- 没有为了性能删除车型、道路或目标。

数据：`results/ab_scene_off.json`、`results/ab_scene_on.json`。

## 2. P1：真实 Track 闭环

- 单一 `TeslaSceneItem` 实例。
- 单路时挂在右下 `cardSceneHost`；空间页把同一实例重挂到 `spatialSceneHost`。
- 不创建第二个 Tesla 世界。
- 目标来自 `video.aiBoxes`，只画当前 confirmed、非 predicted、未超时目标。
- 同一 `trackId` 更新同一个 C++ `LiveCar`。
- `sourceKey` 变化清空；AI 离线清空；1.2 s 无更新删除。
- 当前对象日志：`/dev/shm/tesla_live_objects.json`，快照：`results/stage2_live_objects.json`。

字段：

```text
source_id, source_frame_id, source_timestamp_ns,
track_id, class, confidence, bbox,
predicted, missed, track_state, spatial_validity
```

## 3. P2：三维车型

- 自车改为场景内程序化三维车身，不再使用 QML 贴图覆盖。
- car：下车身 + 梯形座舱 + 后窗/侧窗 + 车顶 + 车轮 + 尾灯。
- truck：驾驶室 + 货箱。
- bus：长车身 + 连续侧窗带。
- person：头/躯干简化模型。
- 车辆模型按 YOLO class 选择；没有类别时不会随机补车。
- 自车与道路共用同一相机和投影。

仍是低面数程序化模型，不是高精车模；这是本阶段的明确限制。

## 4. P3/P4：道路、镜头与相对空间

- 四车道 + 路肩，蓝路径只铺自车车道。
- 扩展地面、城市剪影、近处街区体块，减少黑色空场。
- 单路小窗使用紧凑相机：更低视点、更紧 FOV，突出自车附近目标。
- 空间页使用完整相机；两者共享同一个场景对象和相机状态。
- 没有 Q1 标定，所有目标标记 `RELATIVE_ONLY`；不输出米数、车道测量、TTC。
- 目标横向按图像方位映射，纵深使用非标定相对尺度；不称为真实距离。
- 不再使用 bbox 高度作为“精确纵深”。

## 5. P5：连续更新

- 保留 Track ID。
- 使用检测时间戳和 `dt` 做有界平滑；
- AI 更新约 8–9 Hz，GPU/QSG 每帧绘制；
- AI 过期不继续外推；
- 删除超时目标；
- 没有真实 ego speed/yaw，自车与道路保持固定，不制造道路滚动。

## 6. P6：交付证据

- 单路 + 右下 LIVE：`results/stage2_live_single.png`
- 空间页同一实例：`results/stage2_live_spatial.png`
- 10 秒同步录屏：`results/stage2_live_sync10.mp4`
- Track/场景日志：`results/stage2_live_objects.json`
- A/B：`results/ab_scene_off.json`、`results/ab_scene_on.json`

![单路 LIVE](<C:/Users/JingyuWang/adas-edge/deployment/lubancat/qt_gpu_probe/results/stage2_live_single.png>)

![空间页 LIVE](<C:/Users/JingyuWang/adas-edge/deployment/lubancat/qt_gpu_probe/results/stage2_live_spatial.png>)

## 7. 明确限制

- 场景实际独立 FPS 未单独计数；QSG 57.7 是合成帧率，不等于 57.7 次感知更新。
- 没有 Q1 标定，空间位置全部是 `RELATIVE_ONLY`。
- 车辆是程序化低面数模型；小窗 88px 高，细节仍有限。
- 没有真实 ego speed/yaw，所以不滚动道路。
- 有一次 `RGB565 texture upload failed with GL error 1280` 日志；主视频仍正常显示，后续需清理 FBO 渲染后的 GL 状态。
- 未做 20 分钟长测。

## 8. 结论

本阶段完成了“真实 Q1 目标 → 单一 Track → 同一 Tesla 场景实例 → 右下小窗/空间页复用”的闭环，并改善了车型、相机和道路。  
但视觉仍是低面数示意，且目标空间位置不是标定测量。下一步应先确认这版真实联动与观感，再决定是否继续 GPU 细化。
