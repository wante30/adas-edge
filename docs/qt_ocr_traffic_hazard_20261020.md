# RK3566 Qt 车机 OCR 性能、导航一致性、交通灯与危险事件联动

日期：2026-10-20。基于 `d20c3ee`。没有修改 FPGA RTL、PCIe 协议、RKNN 模型或车机视觉布局。

## 1. 结论摘要

- 做了**同输入、同 publisher、同 512 模型、同 Qt 二进制**的 30 秒 OCR 关闭/开启 A/B。
- OCR 关闭：源视频平均 **14.53 FPS**，QSG **48.78**，YOLO **9.60 FPS**。
- OCR 开启（0.5 Hz 调度，实际受 6–9 s 推理限制）：源视频 **9.49 FPS**，QSG **51.55**，YOLO **7.22 FPS**；OCR 子进程约 **88.6%** 单核。
- 进一步把 OCR 降到 0.1 Hz 并缓存“距离/路名”行框后：源视频 **11.63 FPS**，QSG **40.23**，YOLO **8.16 FPS**；OCR **44.8%** 单核。
- 因此：**OCR 仍是明确的 CPU/吞吐瓶颈，不能宣称全面达到 50–60 QSG 和 14+ 源 FPS。** 0.5 Hz A/B 的 QSG 达到 51.55，但 0.1 Hz 长窗口波动到 40.23；不能只拿一次好窗口下结论。
- OCR 不再造成 YOLO 长时间停止：A/B 开启时结果间隔 p95 253.9 ms，>1000 ms 0 次；低频率 60 秒 p95 243.4 ms，>1000 ms 0 次。
- Traffic Light Specialist 确认真实运行：specialist age p50 **454 ms**、p95 **945 ms**，接近 1 Hz；本地 487 个 AI 帧中 **93 帧为 GREEN**，394 帧无可靠信号为 UNKNOWN。截图中有真实“绿灯”状态。
- 真实本地片段没有触发危险事件；危险事件卡保持“未检测到危险事件”。红灯/行人事件的**数据结构与状态转换**用明确标记的 fixture 验证，未计入真实识别通过。
- 车道接口没有可用的 FPGA 车道数据；界面改为“未接入 / —”，不再显示直行、可变道、禁止变道等假值。
- 没有 FPGA/PCIe 输入，全部为本地 PGVF；真实 PCIe 未验收。

## 2. A/B 方法与原始数据

同一次 publisher 进程、同一 `local_quad.rgb`、同一 512 模型、同一 Qt 二进制；每次 30 秒。OCR 关闭与开启分别重启 AI sidecar，app 重启，publisher 不换内容。

| 指标 | OCR 关闭 | OCR 开启 (0.5 Hz) | OCR 0.1 Hz |
|---|---:|---:|---:|
| 唯一源帧 FPS 平均 | 14.53 | 9.49 | 11.63 |
| Qt 唯一上传 FPS 平均 | 14.53 | 9.49 | 11.63 |
| QSG 回调 FPS | 48.78 | 51.55 | 40.23 |
| YOLO FPS | 9.60 | 7.22 | 8.16 |
| result age p50/p95 | 121/180 ms | 156/231 ms | 143/214 ms |
| result interval p50/p95 | 117/218 ms | 141/254 ms | 127/243 ms |
| >500 ms 间隙 | 2 | 1 | 2 |
| >1000 ms 间隙 | 1 | 0 | 0 |
| OCR CPU（单核百分比） | 0 | 88.6 | 44.8 |
| AI 进程 CPU | 65.6% | 67.5% | 77.0% |
| Qt 进程 CPU | 86.7% | 72.2% | 76.8% |
| publisher CPU | 19.6% | 17.6% | 18.4% |
| 四核平均负载 | 73.6/80.1/76.9/77.3 | 84.2/87.8/91.9/90.6 | 83.7/90.0/87.1/86.2 |

文件：

- `results/ab_off_app.json`、`results/ab_off_perf.json`、`results/ab_off_latency.json`
- `results/ab_on_app.json`、`results/ab_on_perf.json`、`results/ab_on_latency.json`
- `results/low_app.json`、`results/low_perf.json`、`results/low_latency.json`

说明：OCR 关闭时也出现一次 >1000 ms 的 AI 结果间隙，说明间隙不完全由 OCR 产生；开启 OCR 的 30 秒窗口反而没有 >1000 ms。不能把单次窗口的差异当成因果结论。

## 3. OCR 资源定位

审计结果：

1. **OCR 推理本身是主要成本**：空闲时 Q2 卡片检测约 **3.6 s**；与 YOLO/Qt 同时运行时 **6–9 s**。瓶颈是 CPU/ONNX 推理，不是图像缩放。
2. 旧代码把卡片再放大 2 倍，而 RapidOCR 检测器又把长边限制到 320；这是重复 resize。已删除 2 倍放大，卡片的文字框和阈值改为相对高度。
3. OCR 子进程没有反复初始化模型：`warmup()` 在子进程启动时一次，之后复用。
4. 子进程原来绑定最后一个 CPU 核；该核同时承受 Qt/GPU/发布唤醒，实测最忙。改为默认 **CPU1**，可用 `ADAS_OCR_CPU` 覆盖。
5. 独立进程避免了 Python GIL；`nice(19)` + `SCHED_IDLE`，队列 `maxsize=1`。
6. 导航请求只保留最新一张；OCR 忙时丢弃新请求，没有增长队列。
7. 32×18 全图变化检测改为 **卡片 ROI 96×45** 的差异检测，距离数字变化不会被 32×18 缩略图直接抹平；同时每 10 秒强制复核。
8. OCR 结果发布 QML 只在 AI 状态帧（10 Hz）更新，不按 OCR 结果高频重绘。
9. **快速路径**：首次检测缓存“距离行/路名行”两个文字框；之后只对这两个框做 recognition-only，约 **0.65–1.8 s**。检测每 15 秒或连续识别失败时重跑。识别失败不覆盖仍新鲜的已验证结果。
10. 导航有效期从 20 s 改为 **12 s**：因为 0.5 Hz 调度下实际 OCR 仍要 6–9 s，3–5 s 会在新结果到达前必然过期。这是基于实测的折中，不是达到 3–5 s 目标。

## 4. 导航结果与过期控制

`NavigationObservation` 增加：

- `preprocess_ms`、`engine_ms`、`ocr_timestamp_ns`。

状态结构包含：

- `source`、`source_frame_id`、`source_timestamp`（Qt 端映射）、`ocr_timestamp_ns`、`direction`、`distance_m`、`road_name`、`confidence`、`valid`、`reason`。

状态规则：

- 源视频停止：`STOPPED`。
- OCR 结果超过 TTL 或源帧时间过旧：`STALE`。
- OCR 失败：`UNREADABLE`，但**不覆盖仍新鲜的最近有效结果**；直到它自身 TTL 过期。
- 距离在 3 秒内跳变 >500 m：拒绝该次跳变并计数；允许后续一致的重新识别。
- 导航方向缺失时显示“导航暂不可读”，不编造方向。
- 导航 UI 只显示真实 `road_name/distance`；`Q2 OCR` 是实际来源。

当前本地结果：`315/328/341/352/386/397 m + 海城路` 等；最终 60 秒低频率窗口 `ai_navigation_state_last=VALID`、`distance=231 m`。

## 5. Traffic Light Specialist

- 模型：`tl_specialist_512x192_hybrid2.rknn`，1 Hz，运行在 sidecar 的 same pipeline。
- specialist age：p50 **454 ms**、p95 **945 ms**、max **997 ms**（近 1 Hz）。
- 颜色分类：`competition_ai/traffic_light.py` 的 HSV ROI 分类 + `require_lamp_evidence`；多个候选必须一致，冲突返回 UNKNOWN；低置信度/无灯 ROI 返回 UNKNOWN。
- 主模型和 specialist 的 traffic light 框都进入同一 `_detections_to_boxes` 坐标映射，绑定 `source/frame_id`。
- specialist age >3 s 时不再使用其交通灯证据；视频停止时 traffic=UNKNOWN。
- 本地 60 秒：487 个 AI 帧中 **GREEN 93 帧、UNKNOWN 394 帧**，没有把“没检测到”解释成绿灯。
- 截图 `results/full40.png` 显示：真实 **绿灯**、导航 **301 m/海城路**、车道“未接入”、危险事件“未检测到危险事件”。

UI：

- 红/黄/绿分别使用红/黄/绿；UNKNOWN 用“未识别”中性态。
- 没有新增大面板；沿用左上信号卡片。

## 6. 危险事件

已接入/保留的真实数据路径：

- `PEDESTRIAN_IN_PATH`：行人检测框落在假设图像路径且底边 >0.60，来源 `IMAGE_SPACE_PATH_PRESENCE`，不代表碰撞/TTC。
- `PEDESTRIAN_CROSSING_PATH`：原有轨迹横向运动确认。
- `RED_SIGNAL_CONFIRMED`：traffic=RED 且信号证据 <2.5 s，来源 `TRAFFIC_LIGHT_CLASSIFIER`；只做提醒，不判定闯红灯。
- 原有 `LEAD_RAPID_CLOSING`、`LEAD_BRAKE_LAMP_RISE` 等保持不变。

状态结构：

- `event_type/source_id/source_frame_id/timestamp/severity/valid/ttl_ms/evidence`。

UI：

- 现有右侧危险事件卡；按红灯、行人、前车接近、制动灯变化排序，最多 3 条。
- 离线时只显示“AI 源失联”，不再用“前车距离较近/右侧行人注意/无效变道风险”假预览。
- 本地片段没有触发真实危险事件，因此截图保持“未检测到危险事件”；没有用 QML 动画伪造事件。

**测试夹具**（明确不是真实 AI 识别通过）：`results/status_fixture.json` 用 RED + PEDESTRIAN_IN_PATH 输入验证 `hazards` 和 `hazard_details` 的结构、source/frame/time/TTL/evidence。未将其计入真实交通灯或危险事件通过。

## 7. 车道状态

审计 `tools/competition_console/fpga_perception.py`：`HudPerception.lane_left/lane_right` 目前为 `NONE`，本地/当前 PCIe 数据没有可用的左右实线/虚线/偏移字段。

因此保留车道卡片结构，但：

- 当前车道、左侧、右侧均显示 `—`；
- 角标“未接入”；
- 底部“车道信息未接入”；
- 不再显示“直行/可变道/禁止变道”的假值。

没有修改 FPGA RTL 或扩展 PCIe 协议。

## 8. 未达到的目标与真实瓶颈

1. **OCR 开启时源视频 FPS 明显下降**：14.53 → 9.49（0.5 Hz）或 11.63（0.1 Hz）。OCR 进程占一个核心；CPU 总负载接近饱和。
2. **QSG 没有稳定达到 50–60**：A/B 0.5 Hz 为 51.55，低频率 60 秒为 40.23；本地 publisher/Qt/AI 的负载波动大，不能只报一次好窗口。
3. **导航 3–5 s 有效期未达到**：实际 OCR 6–9 s，最终取 12 s；这是当前 OCR 推理能力的限制。
4. **真实 PCIe 未测试**：没有 FPGA 输入；本地 publisher 的 9–15 FPS 不是 FPGA 指标。
5. **真实危险事件未出现**：本地片段没有触发；只验证了数据结构和 fixture。
6. **交通灯在无信号/多灯冲突时大量 UNKNOWN**：这是保守策略的预期行为，但也意味着当前本地片段不能连续显示绿灯。

## 9. 板端启动

```sh
# publisher（本地回放）
python3 pgv_local_publisher.py --raw local_quad.rgb \
  --output /dev/shm/pgv_local/latest.pgv --fps 30 --source-fps 30 &

# 512 YOLO + specialist + Q2 OCR
ADAS_EDGE_ROOT=/home/cat/competition_cockpit_port_20261018 \
/usr/bin/python3 ai_bridge.py --source LOCAL_REPLAY \
  --live-frame-file /dev/shm/qt_gpu_probe_ai.mfr1 \
  --layout quad --display-view front --road-quadrant 0 --navigation-quadrant 1 \
  --main-model /home/cat/adas-edge-rect/rect_models/yolo11n_rk3566_int8_split_512x288.rknn \
  --specialist-model /home/cat/adas-edge-rect/tl_check/tl_specialist_512x192_hybrid2.rknn \
  --main-hz 10 --specialist-hz 1 --ocr-hz 0.1 --idle-timeout 90 &

# Qt
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
QT_QPA_PLATFORM=xcb QT_XCB_GL_INTEGRATION=xcb_egl \
./qt_gpu_probe --fps 30 --duration 0 --start-mode 0 --road-quadrant 0 \
  --pgv-file /dev/shm/pgv_local/latest.pgv --source-label pgv-local \
  --pgv-layout quad --ai-frame-file /dev/shm/qt_gpu_probe_ai.mfr1 --ai-hz 10
```

真实 FPGA 接回后只把 `--source` 改为 `PGV_LIVE`，Qt 去掉 `--pgv-file`；不要沿用本地 FPS 作为 PCIe 验收。

## 10. 文件清单

- `deployment/lubancat/qt_gpu_probe/ai_bridge.py`
- `deployment/lubancat/qt_gpu_probe/q2_navigation_reader.py`
- `deployment/lubancat/qt_gpu_probe/main.qml`
- `tools/competition_console/navigation_reader.py`
- `tools/competition_console/fpga_perception.py`
- `competition_ai/hazard_events.py`
- `deployment/lubancat/qt_gpu_probe/tools/perf_sampler.py`
- `deployment/lubancat/qt_gpu_probe/tools/status_fixture.py`
- 结果：`results/ab_*`、`results/low_*`、`results/status_fixture.json`、`results/full40.png`、`results/low20.png`、`results/final20b.png`
