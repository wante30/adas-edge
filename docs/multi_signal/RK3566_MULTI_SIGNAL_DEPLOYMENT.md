# RK3566 多交通灯识别部署与验收

## 当前状态

本轮代码已经把 INT8 Raw-head 检测器、主模型交通灯输出、原始解码帧 ROI、传统 HSV/轮廓箭头分类、多灯去重/跟踪/TTL 和 JSONL 回放串成一个独立验收入口。当前会话对 `ssh rog` 可用，但对 `opi5`/`lubancat` 的 SSH 在密钥交换阶段被目标端重置，因此本轮没有板端实测数据；下面的 MP4、JSONL 和性能数字只有在脚本实际运行于 RK3566 后才可标记为板端结果。

## 正式模型和输入契约

默认主模型是 `b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn`，来源于 ROG 的 `D:\Datasets\adas_edge_training\rknn_b05o\`，SHA256 为 `65089ef0697bd7378071290baa218905cdcaed768452678da440b8eb116b2a5f`。输入是 `512x288`、`uint8` NHWC；OpenCV BGR 帧先按现有 letterbox 规则转为 RGB，RKNN 输出由 ARM 侧 DFL 和 NMS 解码。BDD 十类 Raw-head 解码器现在按模型类别数解码，交通灯类别为 BDD class 8；不会把十类头误当成 COCO 80 类头。

可选 Specialist 是 `b05l_v8n_p2_raw_heads_int8_500_20261026.rknn`，SHA256 为 `c11b95b481b25ee34a093390001e821af8f2503f923a52159b8576c6a6f38b6f`。旧 BDD 模型的 SHA256 为 `a86ab05338b49be54f95aa4d0cb937f1471924d917840d0f907480e29e041afb`。大模型文件不进入普通 Git 提交，使用上述路径、哈希和部署脚本获取。

检测正式阈值保持 `0.25`；NMS 主模型保持 `0.45`，Specialist 保持 `0.70`。回放工具只在显式 `--conf-threshold`/`--nms-threshold` 时改变这些值，默认不会为了召回率改正式参数。

## 一键回放

把真实比赛视频和两个 RKNN 文件放到板端后运行：

```sh
cd /path/to/adas-edge-multi-signals
INPUT=/data/video/competition.mp4 \
MAIN_MODEL=/data/models/b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn \
SPECIALIST_MODEL=/data/models/b05l_v8n_p2_raw_heads_int8_500_20261026.rknn \
OUTPUT_DIR=/data/outputs/multi_signal_rk3566 \
./deployment/rk3566/run_multi_signal_replay.sh
```

输出包括：

- `rk3566_multi_signal.mp4`：保留输入帧率与顺序；每个灯独立绘制框、`track_id`、颜色、形状、置信度和 `VALID/STALE`。
- `rk3566_multi_signal.jsonl`：每个显示帧一行，含 `track_id`、`bbox`、`color`、`shape`、`confidence`、`timestamp_ns`、`expires_at_ns`、`failure_reason/reason`、`observed_directions`、`directions` 和阶段耗时。
- `rk3566_performance.csv`：逐帧耗时。
- `rk3566_performance.json`：模型哈希、更新频率、NPU/DFL-NMS/颜色/箭头/跟踪/流水线统计和 Linux 负载采样。
- `failures/`：最多 64 个 UNKNOWN/失效 ROI 及其 JSON。

脚本从 OpenCV 解码的原始 BGR 视频帧裁剪 ROI，再交给颜色和箭头分类器；没有把 512×288 缩放图插值放大来冒充原始细节。如果现场视频输入只提供 1280×720 RGB565 的 PGVF 帧，这个帧就是二阶段可获得的最高分辨率，ROI 坐标仍以 `q4_native` 表示。

主模型和 Specialist 默认分别按 10 Hz 和 2 Hz 更新，显示帧按输入 FPS 输出；JSONL 同时记录 `source_fps`、`display_fps`、`model_update_fps` 以及每次更新来自哪个模型。模型漏检时不刷新旧检测，轨迹在 TTL 内仅保留可见性，过期后颜色/形状回到 `UNKNOWN`。

## 正式桥接路径

`deployment/lubancat/afternoon_mosaic/ai/ai_bridge.py --multi-signals` 仍使用 `LatestSignalWorker`、`ContourTemplateBackend` 和原始 `road` 帧。为保持车辆/行人路径不变，`RknnConsoleDetector` 默认继续过滤主模型交通灯；仅在 `--multi-signals` 时打开 `include_traffic_lights=True`，从 BDD/P2 主模型与 Specialist 同时收集交通灯候选，再交给已有去重和独立 track_id 逻辑。`directions` 继续由关联层输出 UNKNOWN，视觉箭头只在 `observed_directions` 中保留。

本轮不部署 Autoware MobileNetV2，不修改 FPGA、PCIe 或视频传输协议，也不改变车辆、行人等目标的默认过滤条件。Qt/HUD 的正式显示需要在板端回放稳定后开启 `--multi-signals` 做一次联调；当前因 SSH 不可达，不能宣称已经完成 Qt/HUD 实测。

## 验收记录模板

板端运行后把 `rk3566_performance.json` 中的 `board_measured`、平台信息、各阶段 p50/p95、`actual_result_update_fps` 和 telemetry 复制到测试报告。PC 浮点 ONNX 与 RK3566 INT8 只做最终框/颜色/形状/有效状态统计比较，不要求逐位输出相同；任何差异先按 `RGB/BGR`、letterbox 映射、NMS、时间戳、TTL 和小目标尺寸分类。
