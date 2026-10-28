# Tesla LIVE 真实动态联动验收（LOCAL_REPLAY）

日期：2026-10-20。FPGA PCIe 断开，本轮全部为本地真实录制回放。

## 1. 回放素材审计

`local_quad.rgb`：

- 文件大小：442,368,000 字节
- 单帧：1280×720 RGB565 = 1,843,200 字节
- 帧数：**240 帧**
- 抽样 0/1/2/30/60/120/180/239 帧：
  - 整帧 MD5 全部不同；
  - Q1 区域 MD5 全部不同。
- 结论：这是真实多帧动态素材，不是静态图。

Publisher 30 FPS 循环 240 帧（约 8 秒一轮），`transport frame_id` 持续递增；内容循环时 generation 递增，C++ tracker epoch 重置，防止上一轮尾部与下一轮开头错误连接。

## 2. 动态联动实现

数据流：

```text
local_quad.rgb
  → pgv_local_publisher.py (30 FPS, atomic rename)
  → /dev/shm/pgv_local/latest.pgv
  → Qt VideoItem (Q1, local open/pread)
  → MFR1 → ai_bridge.py
  → YOLO11n 512 + Specialist + Q2 OCR
  → trackId + timestamp
  → TeslaSceneItem
  → 右下小窗 / 空间页
```

- 单一 `TeslaSceneItem`；
- 真实 YOLO Track ID；
- 无 GIF、无随机车辆、无伪检测；
- Tesla 空间位置仍是 `RELATIVE_ONLY`；
- 无 Q1 标定，不输出真实米数。

## 3. 循环边界

`pgv_local_publisher.py` 在内容索引回绕时递增 PGVF `generation`；C++ `VideoItem` 检测 generation 变化，递增 tracker epoch；AI 结果到达新 epoch 时清空旧 tracks，避免：

- 上一轮结束的车辆连接到下一轮开头；
- Track ID 跨循环复用造成虚假连续。

## 4. 10 秒同步录制

- 录屏：`results/dynamic_live_sync.mp4`
- 单路 + 右下 Tesla：`results/dynamic_live_single.png`
- 空间页：`results/dynamic_live_spatial.png`
- Track 生命周期：`results/track_lifecycle.jsonl`

采样 22 次（每 0.5 s），source frame 从约 1970 到 2273，覆盖一次 240 帧内容循环。日志包含每帧的 track_id 和 class，当前 YOLO 检测到 car/truck/traffic light 等。

## 5. 验收对应

1. 车辆进入画面创建目标：日志中 track_id 新增；
2. 同一 Track ID 持续更新：track_lifecycle 中同一 id 多次出现；
3. 类别模型：car/truck/person 由真实 class 选择；
4. 目标离开/超时：C++ 1.2 s 未更新删除，AI 离线清空；
5. 无重复创建虚假车辆：对象只来自 AI `aiBoxes`；
6. 循环边界：generation + epoch 重置；
7. 无 GIF / 随机目标 / 假 ego 速度。

## 6. 当前限制

- Q1 无相机标定；空间位置是相对示意，不输出真实距离；
- 周边车辆是低面数程序化模型；
- 右下 88px 小窗细节有限；
- 全功能唯一视频上传约 14.8 FPS（AI/OCR 资源限制）；
- 本地 240 帧循环，不是无限新道路场景；
- 无 20 分钟长测。

## 7. 文件

- `deployment/lubancat/qt_gpu_probe/tools/pgv_local_publisher.py`
- `deployment/lubancat/qt_gpu_probe/video_item.cpp/.h`
- `deployment/lubancat/qt_gpu_probe/ai_bridge.py`
- `deployment/lubancat/qt_gpu_probe/tesla_scene_item.cpp`
- `results/dynamic_live_sync.mp4`
- `results/dynamic_live_single.png`
- `results/dynamic_live_spatial.png`
- `results/track_lifecycle.jsonl`
