# HMI 性能报告

更新日期：2026-10-28  
对象：RK3566 Qt Quick ADAS HMI（`/home/cat/qt_gpu_probe_ai_20261019/qt_gpu_probe`）  
原则：区分“视频采集 FPS”“Qt 渲染回调 FPS”“AI 结果 FPS”“物理 HDMI scanout FPS”。

---

## 1. 当前实测（2026-10-28）

| 指标 | 数值 | 来源 |
|---|---:|---|
| 视频采集 | **49.49 FPS** | `/dev/shm/pango-video/status.json` |
| HMI 自报 | **56 FPS** | 截图顶部 |
| AI 结果率 | **9.98 Hz** | `adas_A_live_default_summary.json` |
| 主 RKNN 调用 p50 | 29.15 ms | 同上 |
| 主模型总耗时 p50 | 38.77 ms | 同上 |
| 全 AI 工作 p50 | 49.79 ms | 同上 |
| NPU | 100% @ 600 MHz | 同上 |
| GPU | mean 33.9%，p50 35% | 同上 |
| CPU | 69.48% of one core / 17.37% machine | 同上 |
| RSS | 144.06 MB（峰值 174.02 MB） | 同上 |
| SoC 温度 | 35.56–40.0 °C | 同上 |
| 物理 scanout | 未测（`null`） | Qt 报告 |

截图：`docs/hmi/assets/current_live_20261028.png`。

> 56 FPS 是 Qt/QML 自报的渲染/回调指标，不是 HDMI 物理扫描率。`physical_scanout_fps` 为空，不能宣称“屏幕 56 Hz 实扫”。

---

## 2. 历史配置对比（同一板端）

README/历史报告中的实测范围：

| 配置 | QSG 回调 | AI 结果 | 说明 |
|---|---:|---:|---|
| 真实 PCIe，无 AI | 约 60.3 FPS | 0 | 采集 60.3 FPS；上传 p95 1.89 ms |
| 真实 PCIe + AI 5 Hz | 约 51.5–56.7 FPS | 4.86 Hz | 上传 p95 2.45 ms |
| 真实 PCIe + AI 3 Hz | 约 56.1–61.2 FPS | 3.0 Hz | 上传 p95 2.33 ms |
| 无 AI（新布局） | 约 45 FPS | 0 | 新 LiveBev/测试面板 |
| AI 5 Hz，sidecar Nice=10 | 约 42–44 FPS | 4.7–5.0 Hz | 推荐稳定档 |
| AI 10 Hz，默认优先级 | 约 30–35 FPS | 9.3–9.8 Hz | Qt 明显降速 |
| AI 10 Hz，Nice=10 | 约 30–36 FPS | 约 9.5 Hz | 只改善调度竞争 |

当前截图 56 FPS 高于历史“10 Hz → 30–36 FPS”，可能原因：

- 当前主视/场景复杂度与历史测试不同；
- 当前没有开启 Specialist/Navigation OCR 高频路径；
- `QT_FRAME_METADATA_HZ=15` 与 `Nice=10` 生效；
- 56 FPS 是短时窗口，不代表长时间稳定。

**不要用单一截图承诺长时间 56 FPS。**

---

## 3. 渲染管线

```text
pango_video_capture
  → PGVF（40B + RGB565 + CRC）
  → Qt Producer 线程 stat/CRC/拷贝到帧池
  → 渲染线程 glTexSubImage2D（单纹理）
  → QQuickFramebufferObject / TeslaSceneItem
  → Mali-G52 / EGL / X11
  → HDMI 1280×720
```

优化点：

1. 单纹理 RGB565，一次上传；不做每帧格式转换。
2. 预分配帧池，避免每帧整帧 `QByteArray` 分配。
3. 渲染线程直接拉取最新帧；QML 不为每帧排 GUI 事件。
4. 静态卡片 `layer.enabled` 缓存。
5. 目标框用轻量边框/标签，不用重型 QML 组件。
6. AI 独立 latest-only worker，不阻塞 EGL 提交。
7. `QT_FRAME_METADATA_HZ=15` 限制元数据通知。
8. `Nice=10` 降低 AI sidecar 与 Qt 抢核。
9. `QT_PGVF_SKIP_CRC=1` 仅本地调试可跳过 CRC；真实 PCIe 默认校验。

---

## 4. 资源与瓶颈

### 4.1 NPU

- 当前 10 Hz 主模型已让 NPU 100%。
- 再叠加 Specialist、四路、跟踪/事件会争 NPU。
- 不能承诺“四路 10 Hz + 双模型 + HMI”同时稳定。

### 4.2 CPU

- AI sidecar 单核 69.5%；4×A55 整机 17.4%。
- DFL/NMS/状态构建/导航 OCR 都是 CPU 重活。
- Qt 渲染线程与 AI 抢核时，HMI FPS 会下降（历史 10 Hz → 30–36 FPS）。

### 4.3 GPU

- 当前 mean 33.9%，p50 35%，800 MHz。
- 3D BEV + 视频纹理 + 框叠加仍有剩余，但不是无限余量。
- 空间视图（Tesla BEV 全屏）历史上上传只有 7.22 FPS；当前布局为右下卡片，负载较低。

### 4.4 PCIe

- Gen2 x1，视频 91 MB/s @ 49.5 FPS，链路可承载。
- 但采集 publisher 本身约 49.5 FPS，不是 60；HMI 不能凭空显示 60 个新帧。

### 4.5 导航 OCR

- p50 约 1.2 s，0.1 Hz；CPU 占用高，结果经常 `valid=0`。
- 不适合实时导航；建议降频/异步/只在导航卡需要时运行。

---

## 5. 时延分解（当前 A）

```text
源帧 → PGVF 发布：约 20.2 ms 周期（49.5 FPS）
PGVF → Qt 读/CRC/上传：未单独测；历史上传 p95 1.89–3.13 ms
Qt → MFR1 → AI 请求：10 Hz
AI 主 RKNN：p50 29.15 ms
AI 预处理：p50 2.01 ms
AI DFL/NMS：p50 7.97 ms
AI 主模型总：p50 38.77 ms
AI 全工作：p50 49.79 ms
AI 结果 → Qt 框：未单独测
HUD 自动发送：约 500 ms 一次
HUD applied：seq 增长；像素延迟未测
```

注意：AI 结果 10 Hz，显示 56 FPS；检测框在两帧 AI 结果之间保持/外推，不是每帧新检测。

---

## 6. 测试方法

### 6.1 视频/系统

```sh
cat /dev/shm/pango-video/status.json
systemctl status pango-video-capture adas-ai-bridge-live adas-hud-bridge
```

### 6.2 AI

```sh
cat /home/cat/qt_gpu_probe_ai_20261019/results/adas_A_live_default_summary.json
tail -5 /home/cat/qt_gpu_probe_ai_20261019/results/adas_A_live_default_timing.jsonl
```

### 6.3 HMI

- 截图：`docs/hmi/assets/current_live_20261028.png`
- 历史 JSON：`docs/deployment/evidence/real_pcie_rga_2_20261025_summary.json`
- Qt 报告字段：`qt_scene_graph_render_callback_fps_last_window`、`source_fps`、`successfully_uploaded_frames`、`physical_scanout_fps`

### 6.4 资源

- `/sys/class/devfreq/fde40000.npu/load`（注意：可能恒为 100，不能单独作为利用率证据）
- `/sys/class/devfreq/fb000000.gpu/load`
- `/sys/class/thermal/thermal_zone*/temp`
- `perf_sampler.py` / `board_resource_sampler.py`

---

## 7. 能说什么 / 不能说什么

**能说：**

- 当前实板视频 49.49 FPS，AI 9.98 Hz，HMI 自报 56 FPS。
- NPU 100%，GPU mean 33.9%，CPU 69.5% 单核，RSS 144 MB。
- 历史 5 Hz AI 比 10 Hz 对 HMI 更友好。
- HUD V2 写入被接受/应用。

**不能说：**

- 不能说 56 FPS 是物理 HDMI scanout。
- 不能说四路 10 Hz + 双模型 + HMI 已稳定。
- 不能说 AI 10 Hz 等于显示 10 Hz。
- 不能说当前长时间运行不会降速。
- 不能说旧 YOLO11n 的 FPS 属于 B05N/P2。

---

## 8. 证据

- `docs/hmi/assets/current_live_20261028.png`
- `docs/deployment/evidence/adas_A_live_default_summary_20261028.json`
- `docs/deployment/evidence/real_pcie_rga_2_20261025_summary.json`
- `deployment/lubancat/qt_gpu_probe/README.md`
- `docs/display_pipeline_p1_20261021.md`、`docs/display_queue_latest_only_20261021.md`、`docs/qt_render_scheduling_p2_20261021.md`
