# Tesla-BevCam-Clone LIVE 第一阶段：真实 Track → 场景对象

日期：2026-10-20。分支：`restore/tesla-original`。本轮只完成真实感知到场景对象的闭环，不继续做完整三维重建。

## 1. 数据来源

- 真实 PCIe Q1：`pgv-live`
- `/dev/shm/pango-video/latest.pgv`
- Qt `VideoItem` 的现有 YOLO + tracker，不修改模型和 RKNN 调度
- 场景输入：`video.aiBoxes`
- 每个对象保留：
  - `trackId`
  - `className`
  - `bbox`
  - `confidence`
  - `detectTimestampNs`
  - `missed/predicted/state`
- 场景对象增加：
  - `source_id = pgv-live`
  - `source_frame_id`
  - `source_timestamp_ns`
  - `spatial_validity = RELATIVE_ONLY`
  - `track_state = CONFIRMED`

映射日志：`/dev/shm/tesla_live_objects.json`，本轮快照：`results/stage1_live_objects.json`。

## 2. 单实例、双显示位置

新增/修改：

- `TeslaSceneItem` 只实例化一次，声明在右下角 `cardSceneHost`。
- 单路模式：它显示在右下角 `Tesla BEV 空间` 卡片内。
- 空间视图：同一个 `TeslaSceneItem` 通过 `parent` 切换到 `spatialSceneHost` 并放大。
- 不存在两套场景对象或两套空间世界。
- 右侧卡片、视频、YOLO 框、AI、PCIe 和 HUD 布局不变。

## 3. 自车

- 自车已改为场景内的程序化三维成员，不再使用 205px QML 贴图覆盖。
- 位于自车车道中心，和道路共用透视相机。
- 当前是阶段一的简洁程序化车身，不是最终精细模型。
- 其外形是演示模型，不代表感知测量。

## 4. 目标位置与可信度

没有可验证的 Q1 相机内参/安装高度/俯仰角，因此：

- 不输出米制距离；
- 不使用 bbox 高度冒充真实纵深；
- 使用横向图像方位 + 非标定相对纵深；
- 所有对象标记 `RELATIVE_ONLY`；
- 不伪造车道归属、车速或 TTC；
- 只在场景中显示有当前检测证据的目标。

## 5. 更新和失效

- 同一 `trackId` 更新同一个 C++ `LiveCar`；
- `sourceKey` 改变时清空；
- `predicted/missed` 目标不进入场景；
- AI 离线清空；
- 1.2 s 没有更新的目标删除；
- 不做无限外推；
- 场景按 QSG 持续绘制，但目标状态只在真实检测更新时变化。

## 6. 实机证据

![单路：真实 Q1 + 右下角 LIVE 场景](<C:/Users/JingyuWang/adas-edge/deployment/lubancat/qt_gpu_probe/results/stage1_live_single.png>)

![空间视图：同一个场景实例放大](<C:/Users/JingyuWang/adas-edge/deployment/lubancat/qt_gpu_probe/results/stage1_live_spatial.png>)

映射日志实际值示例：

```json
{
  "source_id": "pgv-live",
  "source_frame_id": 150766,
  "source_timestamp_ns": 2514580704923,
  "objects": [
    {"track_id": 12, "class": "car", "spatial_validity": "RELATIVE_ONLY", "track_state": "CONFIRMED"},
    {"track_id": 45, "class": "car", "spatial_validity": "RELATIVE_ONLY", "track_state": "CONFIRMED"}
  ]
}
```

## 7. 限制

- 周边车辆仍使用轻量 billboard/程序化绘制，尚未完成 car/bus/truck 的独立精细三维模型；这是第二阶段。
- 自车是程序化演示模型，不是真实车辆外观测量。
- 没有 Q1 标定，所有空间位置是 RELATIVE_ONLY。
- 尚未完成 10–20 秒同步录屏；本轮先交付真实帧截图、帧号和映射日志。
- 当前右下卡片只有 88px 高，场景构图仍偏小；先验证数据闭环，再优化构图。

## 8. 验收问题回答

1. 目标来自：真实 PCIe Q1，`pgv-live`。
2. Track ID：见 `stage1_live_objects.json`，同 ID 持续更新同一对象。
3. 位置计算：图像横向方位 + 非标定相对纵深，不输出米制距离。
4. 标定：没有；全部 `RELATIVE_ONLY`。
5. 更新：真实 `aiBoxes` → C++ LiveCar；同 track 平滑更新。
6. 视频停止/AI 离线：`online=false` 清空目标；超时 1.2 s 删除。
7. 预录/随机/模拟目标：正式 LIVE 中不使用 GIF，不生成随机目标；场景对象只来自 `video.aiBoxes`。

## 9. 下一步

等待用户确认第一阶段真实数据联动后，再做：

- car/bus/truck/person 独立模型；
- 右下小窗构图优化；
- 10–20 秒同步录制；
- 更可靠的空间映射和连续插值。
