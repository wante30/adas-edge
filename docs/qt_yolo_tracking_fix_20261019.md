# RK3566 Qt / YOLO 检测框同步与轻量跟踪修复

日期：2026-10-19。范围：只修复检测框延迟、抖动、跟踪连续性和 sidecar 负载；未修改 FPGA、PCIe 协议、RKNN 模型或车机 UI 结构。

## 1. 结论

- 原问题不是单纯 Qt 渲染慢。实测 AI 结果本身在 Qt 接收时的年龄为 **87 ms / 138 ms（p50/p95）**，窗口 frame lag 为 **1 / 3 帧（p50/p95）**。旧界面把这份结果直接画在当前视频帧上，因此车辆移动时框会落后并停留。
- sidecar 的 30 秒测试没有停止；286 次请求对应 286 次结果，结果间隔 p50 **98.5 ms**、p95 **202.8 ms**，>500 ms 间隙 1 次、>1000 ms 0 次。
- 之前的严重掉帧主要来自 Q2 导航 OCR 的 CPU 抢占；本次性能测试关闭 OCR。
- 新增 C++ 轻量跟踪：IoU 关联、稳定 track id、速度估计、帧间预测、预测/检测状态区分、快速淡出；不改模型，不伪造新目标。
- 本机没有 FPGA/真实 PCIe 输入，全部验证使用参考素材本地回放；真实 PCIe 的最终验收仍需接回 FPGA 后复测。

## 2. 实测延迟（30 秒，286 个配对样本）

时钟：同一 RK3566 上 Qt `steady_clock`、Python `time.monotonic_ns()`，均为系统单调时钟。

| 指标 | p50 | p95 |
|---|---:|---:|
| capture → AI infer start | 29.4 ms | 62.7 ms |
| AI receive → infer start | 0.32 ms | 2.23 ms |
| RKNN infer（主模型 + 打包） | 58.1 ms | 104.3 ms |
| result publish → Qt receive | 1.01 ms | 2.48 ms |
| result age at Qt apply | 88.1 ms | 138.5 ms |
| display frame id − AI source frame id | 1 帧 | 3 帧 |
| sidecar result interval | 98.5 ms | 202.8 ms |

- 结果年龄 p95 138 ms，低于 Qt 现有的 250 ms 过期丢弃门限；没有把超龄结果当新结果。
- sidecar 输入是 latest-only：`LatestFrameSlot` 单槽覆盖；286 请求/286 结果，`AI receive → infer start` p95 2.23 ms，没有 FIFO 积压。
- 数据文件：`deployment/lubancat/qt_gpu_probe/results/ai_latency_tracker.json`。

## 3. Tracker 设计

代码位置：`deployment/lubancat/qt_gpu_probe/video_item.cpp/.h`。

- 检测与 track 按 class + IoU 贪心关联；稳定 `trackId`。
- 匹配上的 track **位置和框尺寸直接采用新检测**，不把平滑后的旧位置当作新检测；速度用带限幅的 alpha-beta（常速 Kalman 稳态形式）更新。
- 未匹配 track 只做有限帧预测，`predicted=true`；超过 2 次结果丢失即移除，界面同时淡出。
- 新 track 需要 **第二次检测确认** 才显示，避免把 100–200 ms 前的旧框直接初始化到当前帧。
- 预测上限默认 4 帧，可通过 `VideoItem.maxTrackPredictionFrames` 调整；速度限幅 ±30 px/帧。
- 预测框不创建新目标；置信度仍来自原模型；`PRED` 标签和较低不透明度表示预测状态。
- 主视/导航切换、AI 超时清空时清空 tracks 和 `next_track_id`，避免跨路串框。
- QML 在每次 `frameAdvanced`（每个生产帧）用 `displayFrameId - payload.frameId` 推进框位置，因此两次 YOLO 结果之间框以视频帧率连续移动。

可点击主画面上的 `TRACK ON/OFF` 或按 `L` 做视觉对比。

## 4. 视觉对比与运行数据

- 对比图：`deployment/lubancat/qt_gpu_probe/results/tracker_compare_final.png`
- ON 截图：`results/final_on.png`
- OFF 截图：`results/final_off.png`

本地回放（30 fps 参考素材，512 主模型）：

- Qt scene graph：**63.2 FPS**（跟踪开启，30 秒报告）；
- AI 状态：约 **8.5–9.4 FPS**；
- 本机 PGVF publisher 是额外瓶颈，源视频在 17–30 FPS 波动；它不属于跟踪开销，真实 PCIe 输入下需复测；
- 上传 p95：2.8 ms；
- 跟踪开启时 QSG 未低于无 AI 的 60.6 FPS 基线。

## 5. sidecar 稳定性

- 30 秒内 286 次结果，无停止、无异常退出；`inference_errors` 未增加。
- 结果间隔 p95 202.8 ms；>500 ms 一次，>1000 ms 无。
- 之前出现的“越跑越慢/停止”与 Q2 OCR 检测线程的 CPU 抢占强相关；OCR 默认关闭后 AI 输出稳定在约 8–10 Hz。
- 未通过重启掩盖问题；本次只关闭了非必要的 OCR 负载，latest-only 与结果发布路径保留。
- 若后续在真实 PCIe 上出现 >500 ms 间隙，应继续查 RKNN 调用阻塞/共享帧读取，而不是让 UI 保留旧框。

## 6. 坐标/ROI 检查

- AI box 已由 sidecar 映射到 1280×720 display space；Qt 再按主视区域缩放到 QML 966×544。
- 单路 Q1 的 ROI、四宫格裁剪、QML 显示缩放已在本地 Q1 画面核对。
- 主视区域与 1280×720 同为 16:9，没有额外 letterbox；若后续切换非 16:9 显示区域，必须重新做 letterbox 还原。
- 切换主视时清空跟踪状态，避免上一个象限的 track id/预测串到新象限。

## 7. 启动与复测

本地无 FPGA：

```sh
python3 pgv_local_publisher.py --raw local_quad.rgb \
  --output /dev/shm/pgv_local/latest.pgv --fps 30 --source-fps 30 &

ADAS_EDGE_ROOT=/home/cat/competition_cockpit_port_20261018 \
/usr/bin/python3 ai_bridge.py --source LOCAL_REPLAY \
  --live-frame-file /dev/shm/qt_gpu_probe_ai.mfr1 \
  --layout quad --display-view front --road-quadrant 0 --navigation-quadrant 1 \
  --main-model /home/cat/adas-edge-rect/rect_models/yolo11n_rk3566_int8_split_512x288.rknn \
  --specialist-model /home/cat/adas-edge-rect/tl_check/tl_specialist_512x192_hybrid2.rknn \
  --main-hz 10 --specialist-hz 1 --idle-timeout 90 \
  --timing-log results/ai_timing.jsonl &

QT_AI_TIMING_LOG=$PWD/results/qt_timing.jsonl \
DISPLAY=:0 XAUTHORITY=/home/cat/.Xauthority \
./qt_gpu_probe --fps 30 --duration 30 --start-mode 0 --road-quadrant 0 \
  --pgv-file /dev/shm/pgv_local/latest.pgv --source-label pgv-local \
  --pgv-layout quad --ai-frame-file /dev/shm/qt_gpu_probe_ai.mfr1 --ai-hz 10 \
  --output-json results/local_tracker.json

python3 tools/analyze_ai_latency.py \
  --ai-log results/ai_timing.jsonl --qt-log results/qt_timing.jsonl \
  --output results/ai_latency_tracker.json
```

真实 FPGA 接回后，把 `--source LOCAL_REPLAY` 改回 `PGV_LIVE`，`--pgv-file` 改回真实 `/dev/shm/pgv_qt_hud/latest.pgv`，其余跟踪参数不需要改。

## 8. 限制

- 本次没有 FPGA、没有真实 PCIe 输入，不能宣称完成真实上板 PCIe 验收；本地 publisher 的 17–30 FPS 也不是 FPGA 指标。
- 参考素材为 30 fps；本地回放不能提供真实 60 fps 内容。
- 跟踪是 alpha-beta 常速模型，不是完整多模型 Kalman；大幅机动、遮挡和摄像头整体运动仍可能漂移。
- 预测框只用于可视化，不用于测距、TTC 或危险事件；危险事件仍来自原 AI 状态。
- 目前使用 512 主模型；320/416 模型可降低延迟但交通灯/远小目标精度下降，需按实车精度要求选择。
