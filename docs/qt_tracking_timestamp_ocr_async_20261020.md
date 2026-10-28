# RK3566 Qt 检测框时间戳补偿与导航 OCR 异步调度

日期：2026-10-20。范围：现有 Qt Quick / QML、C++ 跟踪器、AI sidecar 和 Q2 导航 OCR；没有修改 FPGA/PCIe 协议、RKNN 模型或车机视觉布局。

## 1. 结论

- 固定 4 帧预测已替换为 **CLOCK_MONOTONIC 时间戳补偿**：轨迹保存 `detect_timestamp_ns`，QML 用渲染帧的单调时间戳计算 `dt` 秒，速度改为 **px/s**，默认最大外推 **250 ms**。
- 旧检测到达时不再直接覆盖当前轨迹：每个 track 保留最近 8 个检测状态，匹配时回到该检测的采集时刻，在其历史状态上校正，再推进到当前渲染时刻。
- 新目标首次高置信度检测立即以 `TENTATIVE` 显示；第二次匹配升级 `CONFIRMED`；低置信度仍等第二次；未确认失配立即删除。
- 导航 OCR 恢复为**独立进程**，不再与 YOLO 共享 Python GIL；请求队列 maxsize=1、只保留最新导航帧，静态画面跳过，OCR 进程绑定最后一个 CPU 核并设为低优先级。
- 正式 512 模型（30 秒，无 OCR）result age p95 **191 ms**；YOLO+OCR 60 秒 result age p95 **191 ms**，AI 输出 p95 间隔 **228 ms**，没有 >1000 ms 停顿。
- 本机没有 FPGA/PCIe 输入，真实 PCIe 验收未完成；全部数据来自本地 PGVF 参考素材。

## 2. 时间基准与数据流

数据流：

```
FPGA/本地 PGVF -> Qt Frame::producedNs (CLOCK_MONOTONIC)
  -> MFR1 header timestamp
  -> ai_bridge time.monotonic_ns()
  -> MVA2 timestampNs (源帧采集时间)
  -> VideoItem tracker (检测时刻匹配)
  -> QML，用最近渲染帧的 displayTimestampNs 推进到当前显示时刻
```

- Qt `Frame::producedNs` 使用 `std::chrono::steady_clock`，Linux 上为 `CLOCK_MONOTONIC`。
- Python 使用 `time.monotonic_ns()`；板端已验证两者都在同一单调时钟域。
- Qt 端不只使用“收到结果的时间”；MVA2 的 `timestampNs` 被保留为检测采集时间。
- `displayTimestampNs` 在 `Renderer::synchronize()` 取到实际渲染帧时更新，不是使用 Unix 时间，也不是用接收时间冒充源时间。
- 旧结果/旧 frame_id 仍被拒绝，不会把过期结果当新帧。

## 3. 时间戳跟踪器

文件：`deployment/lubancat/qt_gpu_probe/video_item.cpp/.h`。

每个 track 保存：

- `trackId`、`lastDetectFrame`、`lastDetectNs`；
- `detectTimestampNs`（源采集时间）；
- 位置、宽高、`vx/vy`（px/s）、`vw/vh`（px/s）；
- `hits`、`missed`、`state`（tentative/confirmed/predicted）；
- 最近 8 个 `{ns,x,y,w,h}` 历史状态。

匹配与更新：

1. 对每个 track，找 `history.ns <= detectNs` 的最近状态。
2. 从该状态按 `vx/vy` 推进到 `detectNs`，只用于 IoU 匹配，最大外推不超过 `maxTrackPredictionMs`。
3. 匹配成功后，位置/尺寸直接采用新检测；速度用 `0.55*old + 0.45*measured` 更新并限幅 ±2000 px/s；尺寸速度限幅 ±800 px/s。
4. 把新检测状态压入历史，超过 8 个丢弃最旧。
5. 未匹配轨迹 `missed++`；confirmed 轨迹最多保留 2 次结果，tentative 轨迹一失配即删除；超过 900 ms 的轨迹也删除。
6. 输出给 QML 的是“最近真实检测状态 + 速度”，不是把预测框当成新检测。

QML 时间推进：

```
predictSeconds = min(maxTrackPredictionMs/1000,
                     max(0, (displayTimestampNs - detectTimestampNs)/1e9))
x = base_x + vx * predictSeconds
y = base_y + vy * predictSeconds
w = base_w + vw * predictSeconds
h = base_h + vh * predictSeconds
```

`maxTrackPredictionMs` 默认 250 ms，可配置；超过后不再外推。AI 失联时现有 1.5 s 清空逻辑生效；主视/导航切换清空 tracks。

## 4. 新目标首次显示

- 首次检测 confidence >= 0.45：立即输出 `TENTATIVE`，QML 以 0.62 不透明度显示，不额外加调试文字。
- 首次检测 confidence < 0.45：不显示，等第二次匹配。
- 第二次匹配：升级 `CONFIRMED`，不透明度 1.0。
- 未确认目标失配：直接删除，不会靠预测“造出”新目标。
- `PREDICTED` 目标只来自已有轨迹的短期外推，以 0.42 不透明度显示；`missed` 2 次后消失。

## 5. A/B/C 对比

QML 保留一个检测框模式按钮（`TRK TIME` / `TRK FIX4` / `TRK RAW`），按 `L` 循环：

- A `RAW`：不推进，显示最近检测位置。
- B `FIX4`：旧的固定 4 帧逻辑（按当前 source FPS 折算秒）。
- C `TIME`：单调时间戳推进。

对比图：`deployment/lubancat/qt_gpu_probe/results/tracking_modes_abc.png`。

三张图是在同一段本地回放中切换模式后采集，切换间隔约 1 秒，**不是同一源帧逐像素对比**；因此只能作为视觉参考，不作为误差统计。图中 TIME 模式的车辆框没有 FIX4 的明显前冲，RAW 在运动目标上更靠后。

## 6. 正式 512 模型延迟

模型：`yolo11n_rk3566_int8_split_512x288.rknn`；本地 PGVF；30 秒；无 OCR；279 个配对样本。

| 指标 | p50 | p95 |
|---|---:|---:|
| capture → AI infer start | 45.3 ms | 113.5 ms |
| AI receive → infer start | 0.38 ms | 37.0 ms |
| RKNN infer | 85.2 ms | 132.1 ms |
| publish → Qt receive | 1.22 ms | 5.26 ms |
| result age at Qt apply | 128.1 ms | 191.0 ms |
| display frame − AI source frame | 3 帧 | 5 帧 |
| result interval | 104.6 ms | 220.6 ms |

- AI 平均约 **9.4 FPS**；QSG **57.7**；本地源视频约 25.6 FPS。
- 结果间隔 >500 ms 1 次，>1000 ms 0 次。

文件：`results/ai_navoff_latency.json`、`results/navoff_app.json`。

## 7. YOLO + OCR 联合

60 秒、512 模型、Q2 导航 OCR 开启、570 个配对样本：

| 指标 | p50 | p95 |
|---|---:|---:|
| capture → AI infer start | 65.4 ms | 120.9 ms |
| AI receive → infer start | 0.43 ms | 36.1 ms |
| RKNN infer | 84.2 ms | 137.6 ms |
| publish → Qt receive | 1.26 ms | 7.82 ms |
| result age at Qt apply | 130.3 ms | 190.9 ms |
| display frame − AI source frame | 2 帧 | 5 帧 |
| result interval | 109.1 ms | 227.6 ms |

- AI 约 **9.5 FPS**；60 秒 570 条结果，无停止；>500 ms 1 次，>1000 ms 0 次。
- Qt QSG 约 **47.3**（同次最后一窗）；本地源视频最后一窗 23.7 FPS，平均受 OCR/回放影响约 14.2 FPS。
- OCR 实际结果：**386 m / 海城路**、**397 m / 海城路**、**273 m / 海城路**（不同时刻），导航卡显示 `OCR LIVE`。
- 与无 OCR 对比，YOLO result age 基本不变（p95 191 ms），但 QSG 从 57.7 降到 47.3，说明 OCR 仍消耗 CPU；它没有再阻塞 YOLO，但显示性能有可测代价。

文件：`results/ai_joint3_latency.json`、`results/joint3.json`、`results/final2.png`。

## 8. OCR 异步调度实现

文件：`ai_bridge.py`、`q2_navigation_reader.py`。

- OCR 在 `multiprocessing` **spawn 子进程**中运行，不在 YOLO 进程的线程中；不再共享 Python GIL。
- 子进程：最后一个 CPU 核、`nice(19)`、可用时 `SCHED_IDLE`；`prctl(PR_SET_PDEATHSIG, SIGKILL)` 保证父进程被杀后不会留下孤儿 OCR 进程（本轮实际遇到过旧孤儿进程，已修复）。
- 请求队列 `maxsize=1`：只保留最新待处理导航帧；OCR 忙时直接丢弃新请求，不建立增长队列。
- 静态画面通过 32×18 灰度缩略图差异跳过；导航结果 TTL 20 秒，不用无限期旧结果。
- 导航 ROI 由 `SourceRole`/MVS1 的 navigation quadrant 决定，默认 Q2；不再假设 Q4。
- OCR 结果回到父进程时刷新时间戳，UI 记录源 frame_id 和单调时间；无效时显示 `等待 Q2 OCR`/`UNREADABLE`，不编造指令。
- Q2 卡片 OCR 增加了“裸数字 + 单位丢失”兼容：卡片里 `315` 被识别为 315 m，避免因 `米` 字被检测漏掉而整条结果无效。当前本地回放可得到 `315/328/386/397 m + 海城路`。

## 9. 验证与未解决问题

已验证：

- 时间戳推进不依赖固定 FPS；A/B/C 模式可切换；
- 新目标首次高置信度可见，低置信度等确认；
- 旧 frame_id 不覆盖当前；
- OCR 进程运行 60 秒，YOLO 持续 9.5 Hz，无 >1000 ms 间隔；
- 导航 UI 能显示真实 OCR 距离/道路名；
- 切换主视会清空轨迹；
- UI 只显示类别/置信度和框，不增加调试文字。

未解决/限制：

1. **没有 FPGA/真实 PCIe**：不能宣称真实 PCIe 验收通过。本地 publisher 的 13–26 FPS 也不是 FPGA 指标。
2. OCR 仍使 QSG 从 57.7 降到 47.3；若正式验收要求更高，需要进一步降低 OCR 频率或把 OCR 放到独立 CPU 核/独立 RK3566 进程并做更严格 CPU 配额。
3. 512 模型 result age p95 191 ms，接近 250 ms 上限；250 ms 外推在急停/急转时仍可能过冲。
4. 摄像头整体运动、遮挡、两车交叉仅用 alpha-beta + IoU + 运动门控，未加入光流/全局运动估计；快速横向运动和交叉场景仍需真实视频观察。
5. A/B/C 截图不是同一源帧，缺少人工标注的中心误差/IoU；不能把截图当作定量精度结论。
6. Q2 OCR 识别仍依赖导航卡片版式，裸数字启发式可能把其他数字误判为距离；正式导航需用实际 FPGA Q2 画面复测。
7. 本地 30 fps 参考素材不能验证 50/60 fps 时间模型；时间公式不依赖帧率，但真实高帧率需板端复测。

## 10. 板端启动

无 FPGA（当前本地回放）：

```sh
python3 pgv_local_publisher.py --raw local_quad.rgb \
  --output /dev/shm/pgv_local/latest.pgv --fps 30 --source-fps 30 &

ADAS_EDGE_ROOT=/home/cat/competition_cockpit_port_20261018 \
/usr/bin/python3 ai_bridge.py --source LOCAL_REPLAY \
  --live-frame-file /dev/shm/qt_gpu_probe_ai.mfr1 \
  --layout quad --display-view front --road-quadrant 0 --navigation-quadrant 1 \
  --main-model /home/cat/adas-edge-rect/rect_models/yolo11n_rk3566_int8_split_512x288.rknn \
  --specialist-model /home/cat/adas-edge-rect/tl_check/tl_specialist_512x192_hybrid2.rknn \
  --main-hz 10 --specialist-hz 1 --ocr-hz 0.5 --idle-timeout 90 &

DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
QT_QPA_PLATFORM=xcb QT_XCB_GL_INTEGRATION=xcb_egl \
./qt_gpu_probe --fps 30 --duration 0 --start-mode 0 --road-quadrant 0 \
  --pgv-file /dev/shm/pgv_local/latest.pgv --source-label pgv-local \
  --pgv-layout quad --ai-frame-file /dev/shm/qt_gpu_probe_ai.mfr1 --ai-hz 10 \
  --output-json results/final2_app.json
```

真实 FPGA 接回后只改输入：

```sh
# sidecar: --source PGV_LIVE；Qt: 去掉 --pgv-file，使用真实 /dev/shm/pgv_qt_hud/latest.pgv
```

## 11. 修改文件

- `deployment/lubancat/qt_gpu_probe/video_item.h`
- `deployment/lubancat/qt_gpu_probe/video_item.cpp`
- `deployment/lubancat/qt_gpu_probe/main.qml`
- `deployment/lubancat/qt_gpu_probe/ai_bridge.py`
- `deployment/lubancat/qt_gpu_probe/q2_navigation_reader.py`
- `deployment/lubancat/qt_gpu_probe/tools/analyze_ai_latency.py`（沿用）
- 结果：`results/ai_navoff_latency.json`、`results/ai_joint3_latency.json`、`results/tracking_modes_abc.png`、`results/final2.png`
