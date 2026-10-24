# B-05G — RK3566 交通灯实时链路接入及 A/B 验收

日期：2026-10-24  
分支：`research/rk3566-adas-models`  
基线：`6f1fe5af235ebfaeac981e495e87b2f17ad8e0af`

## 0. 结论

**BLOCKED / 未达到 B05G PASS。**

已经完成：

- 在真实板端 `AiReplayBridge` 中接入 `AsyncTrafficLightTracker`；
- `TL_MODE=tracked` 实际启动；
- 真实 Specialist 自动检测框进入 Tracker；
- 板端实际产生 604 行 Tracker 日志，包含 69 次 detection 事件和 535 次 frame 事件。

但没有完成：

- 没有取得一次 `color_valid=True` 的有效颜色输出；
- 没有完成 LK 跨帧颜色更新验收；
- 没有完成 legacy/tracked A/B；
- 没有完成性能测量。

因此不能标记 PASS。

## 1. 实际修改的板端文件

实际包根：

```text
ADAS_EDGE_ROOT=/home/cat/competition_cockpit_port_20261018
```

修改：

```text
/home/cat/qt_gpu_probe_ai_20261019/ai_bridge.py
/home/cat/competition_cockpit_port_20261018/tools/competition_console/pipeline.py
```

新增到实际包：

```text
/home/cat/competition_cockpit_port_20261018/competition_ai/traffic_light_async.py
/home/cat/competition_cockpit_port_20261018/competition_ai/traffic_light_tracking.py
```

备份：

```text
ai_bridge.py.bak-b05g
pipeline.py.bak-b05g
```

可审查补丁：

```text
research/rk3566_models/patches/B05G_board.patch
```

## 2. Tracker 真正被调用的位置

`ai_bridge.py::AiReplayBridge._analyze()`：

```python
lane = next(iter(Lane))
snapshot = self.pipeline.snapshot(lane, timestamp_ns)
if self.tl_tracker is not None:
    if snapshot.specialist_frame_id > self.tl_last_specialist_frame:
        self.tl_tracker.push_frame(...)
        self.tl_tracker.update_detections(...)
    else:
        self.tl_tracker.update_frame(...)
```

这不是独立研究脚本；是实际 AI Bridge 的 `_analyze` 路径。

## 3. 视频怎么进入 Tracker

实际测试：

```text
video_20260929_113421.mp4
t=2..5s
90 帧
1280×720 front
LOCAL_REPLAY PGVF
```

Publisher：

```text
pgv_local_publisher.py --raw /home/cat/b05g_113421_3s.rgb
--output /dev/shm/b05g_tl/latest.pgv --fps 30
```

Qt 用同一 PGVF 请求 AI，AI Bridge 对每帧调用 `push_frame` / `update_frame`。  
测试结束后已停止测试 Publisher，服务恢复到 legacy。

## 4. Specialist 帧号匹配

`pipeline.py` 新增 `FrontSnapshot` 字段：

```text
specialist_frame_id
specialist_timestamp_ns
```

在 Specialist 实际推理时记录：

```python
self.specialist_frame_id = int(frame_id)
self.specialist_timestamp_ns = int(timestamp_ns)
```

AI Bridge 用 `snapshot.specialist_frame_id` 与当前 frame_id 调用异步 Tracker。  
不是用推理完成时间顶替采集时间。

## 5. tracked 是否在 RK3566 运行成功

**部分成功：AI Bridge 确实运行了 tracked 模式，但颜色没有有效输出。**

实际日志摘要：

```text
log lines: 604
events:
  detection: 69
  frame: 535
frame_id range: 0..1756
source: LOCAL_REPLAY
track_ids: [1]
records: 6
effective_colors: {UNKNOWN: 6}
color_valid records: 0
```

说明：

- Specialist 自动框确实出现过；
- AI Bridge 确实调用了 Tracker；
- 但被 Tracker 接纳并持续输出的 Track 只有 1 个；
- 该 Track 全程 `effective_color=UNKNOWN`；
- 没有一次 `color_valid=True`。

日志样本：

```json
{"mode":"tracked","frame_id":464,"event":"detection",
 "specialist_frame_id":464,
 "records":[{"track_id":1,"effective_color":"UNKNOWN",
             "color_valid":false,"track_valid":true}]}
```

这不是“颜色已输出”，不能当成 PASS。

## 6. 颜色实际更新频率

**没有有效的颜色观察频率。**

虽然有 535 次 frame 事件，但：

- 没有颜色有效记录；
- `color_valid=True` 为 0；
- 不能把 frame 事件次数当作有效颜色 Hz。

因此不能报告有效颜色输出频率。

## 7. legacy vs tracked

**A/B 未完成。**

原因：

- tracked 没有产生有效颜色，无法比较首次颜色延迟；
- 没有运行完整 legacy 10Hz A/B；
- 没有真实 Specialist 自动检测下的稳定 Track 序列。

已确认的只有：

```text
legacy：默认未启用 Tracker
tracked：AI Bridge 实际执行，但有效颜色为 0
```

## 8. 主 YOLO FPS/CPU/RSS

**未完成可信测量。**

没有在同一时间窗口记录：

- 主 YOLO 实际 FPS；
- Qt 唯一帧 FPS；
- CPU/NPU；
- RSS；
- 视频端到端延迟。

不能声称无退化。

## 9. 为什么没有有效颜色

目前证据指向：

1. Specialist 自动框质量/置信度不足，很多框低于 Tracker 的 0.25 门槛；
2. 接纳的 Track bbox 是低质量候选（例如 x318..347, y194..226），ROI 不是实际亮灯；
3. 该类 ROI 的 HSV 没有足够亮/饱和的灯色像素，返回 UNKNOWN；
4. 没有足够连续成功 Track 来验证 LK 颜色更新。

这不是“Tracker 没接入”；是接入后没有产生可信颜色。

## 10. 强制 UNKNOWN 的情况

实现中包括：

- detection 超过 1200ms；
- 检测帧不能推进到当前帧；
- LK 特征不足；
- 位移/边界异常；
- track 丢失；
- 颜色证据超过 500ms；
- 红绿冲突未确认；
- `color_valid=False`。

测试通过（29 tests），但板端样本没有产生有效颜色。

## 11. 最终运行配置和回滚

测试配置：

```text
TL_MODE=tracked
source=LOCAL_REPLAY
video=113421 3s front
specialist-hz=1
main-hz=10
```

已回滚：

- 停止测试 Publisher；
- 删除 `adas-ai-bridge.service` 中的 `TL_MODE=tracked`；
- 服务恢复 legacy；
- 当前 `systemctl is-active adas-ai-bridge` = active；
- 未改 A 线 Qt、FPGA/PCIe、正式权重。

验证：

```bash
grep TL_MODE /etc/systemd/system/adas-ai-bridge.service  # no output
```

## 12. 回答交付问题

1. **实际修改哪个板端 AI Bridge 文件**：`ai_bridge.py` 和实际包 `pipeline.py`。
2. **Tracker 在哪里被调用**：`AiReplayBridge._analyze()` 的 snapshot 后。
3. **视频如何进入**：Local Replay PGVF → Qt 请求 AI → `_analyze` 的 `road` 帧 → `push_frame/update_frame`。
4. **Specialist 帧号如何匹配**：`pipeline.py` 记录 `specialist_frame_id/timestamp_ns`，AI Bridge 用原帧号关联。
5. **tracked 是否运行成功**：部分成功；AI Bridge 运行，但没有有效颜色输出。
6. **颜色实际更新频率**：0 个有效颜色记录，不能报 Hz。
7. **legacy 对比**：未完成，不能报延迟改善。
8. **主 YOLO/CPU/RSS**：未完成可信测量。
9. **真实日志/调试视频**：有日志 `B05G_tracker_summary.json`；没有可声称的调试颜色视频。
10. **最终配置/回滚**：已恢复 legacy；不推荐把 tracked 交给 A 线。

## 13. 结论

B05G **未 PASS**。

已达到：AI Bridge 实际调用 Tracker，Specialist 自动框进入，真实帧缓存/日志存在。  
未达到：至少一次真实 Specialist 自动检测后的有效 LK 颜色输出，以及 A/B/性能验收。

下一步不应继续写独立算法；应先分析为什么板端自动 Track 的 ROI 无法产生可信颜色（检测框质量、0.25 门槛、ROI 坐标/相机模型），再决定是否允许 Tracker 进入 A 线评审。

## 14. 交付

```text
research/rk3566_models/patches/B05G_board.patch
research/rk3566_models/reports/B05G_board_integration.md
research/rk3566_models/reports/assets/B05G_tracker_summary.json
competition_ai/traffic_light_async.py
tests/test_b05f_async.py
```

没有提交模型权重或大型视频。
