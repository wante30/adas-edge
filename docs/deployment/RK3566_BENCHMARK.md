# RK3566 实板性能基准

> 本文件是 2026-10-28 的 A 方案实板快照；2026-10-31 默认已切为 **B（B05N P2）**，B-CAPI/MTL 的回放数据见 `docs/deployment/MTL_CAPI_DEPLOYMENT.md`。


更新日期：2026-10-28  
板卡：EmbedFire LubanCat-1，RK3566，4×Cortex-A55，3.8 GiB，Mali-G52，RKNPU  
系统：Ubuntu 20.04.6 LTS，Linux 4.19.232，Qt 5.12.8，RKNN Lite2 2.3.2，驱动 0.8.2  
重要：板端系统时钟未同步（显示 2024-06-18）；以下“2026-10-28”是审计日期，不是板端 `date`。

---

## 0. 一句话结论

当前实板是**单路 1280×720 2×2 mosaic + BDD Raw-head INT8 主模型 10 Hz + Qt HMI**：

- 视频采集：**49.49 FPS**
- AI 结果率：**9.98 Hz**
- 主 RKNN 调用 p50：**29.15 ms**
- 主模型总耗时 p50：**38.77 ms**
- 全 AI 工作 p50：**49.79 ms**
- HMI 自报：**56 FPS**
- NPU：**100% @ 600 MHz**
- GPU：mean **33.9%**
- CPU：**69.5% of one core**
- RSS：**144 MB**

这不是四路独立 PCIe 流，也不是四路 AI 同时 10 Hz。

---

## 1. 当前 live 负载（A 方案）

数据源：`/home/cat/qt_gpu_probe_ai_20261019/results/adas_A_live_default_summary.json`  
采集时间：2026-10-28 审计时，已累计 10,637 次主模型请求。

| 指标 | 数值 |
|---|---:|
| 视频发布状态 | `state=running`, frames=445,008, fps=49.492, error="" |
| AI 请求率 | 9.9915 Hz |
| 主模型结果率 | 9.9756 Hz |
| 主 RKNN 调用 p50 / p95 | 29.15 / 34.41 ms |
| 主模型预处理 p50 | 2.01 ms |
| 主模型 DFL 解码 + NMS p50 | 7.97 ms |
| 主模型总耗时 p50 / p95 | 38.77 / 47.84 ms |
| 全 AI 工作 p50 / p95 | 49.79 / 67.34 ms |
| 共享帧读取 p50 | 3.22 ms |
| RGB565 解码 p50 | 0.96 ms |
| 状态构建 p50 | 2.77 ms |
| 结果编码 p50 | 1.19 ms |
| 结果发布 p50 | 0.24 ms |
| 进程 CPU（单核） | 69.48% |
| 进程 CPU（整机） | 17.37% |
| RSS 结束 / 峰值 | 144.06 / 174.02 MB |
| NPU load / freq | 100% / 600 MHz |
| GPU load mean / p50 | 33.89% / 35% |
| GPU freq mean / p50 | 721.33 / 800 MHz |
| SoC 温度 min/max | 35.56 / 40.0 °C |
| GPU 温度 min/max | 34.38 / 36.67 °C |

### 1.1 HMI

- 截图：`docs/hmi/assets/current_live_20261028.png`
- HMI 自报 56 FPS，AI 10.0 FPS，NPU 100%，GPU 25%，PCIe 在线。
- 视频源：`/dev/shm/pango-video/latest.pgv`，1280×720 RGB565。
- 主视：Q2（右上，导航/路口），四宫格底部可见。
- 检测框：TRUCK 53%、CAR 39%、PERSON 31%、BUS 95% 等。
- 信号灯：未识别；导航：暂不可读；车道：车道视觉未接入。

> HMI 的 56 FPS 是 Qt 自报的渲染/回调指标，不是物理 HDMI scanout；`physical_scanout_fps` 未测量。

### 1.2 当前模型

- A：`bdd_yolov8n_512x288_raw_heads_int8.rknn`
- SHA256：`a86ab053...041afb`
- 输入：`[1,288,512,3]` uint8 NHWC
- 主模型 10 Hz，危险 5 Hz，导航 OCR 0.1 Hz

> 进程参数与 summary 为 A；截图右上角显示“方案 B”。以进程参数/模型 hash/summary 为准，UI 标签待修。

---

## 2. 历史单流基准（YOLO11n COCO，非 B05N/P2）

这些数字来自早期 YOLO11n COCO 模型，**不能归到 B05N P2**。

| 输入 | 端到端吞吐 | NPU-only mean | 备注 |
|---|---:|---:|---|
| 320 | 27.715 FPS（36.081 s / 1000 帧） | 21.804 ms | 历史 pipeline |
| 416 | 21.731 FPS（46.016 s / 1000 帧） | 32.161 ms | 历史 pipeline |
| 640 | — | 70.692 ms | 历史 raw-head 6 输出 |
| 640 split | 10.394 FPS | 96.206 ms | 独立 30-run smoke |

旧生产 YOLO11n 512×288 split 的 5,787 次 timing：

- 主模型流水线 p50 `51.1 ms`
- RKNN 调用 p50 `39.2 ms`
- 解码/NMS p50 `9.5 ms`
- 预处理 p50 `1.8 ms`

当前 A 的 RKNN p50 29.15 ms，比旧 512×288 split 更快；但这是不同模型/不同后处理，不能直接得出“新模型更快”的结论。

---

## 3. B05P 四方案板端回放（10 Hz）

测试方式：32 张 Cityscapes 图打包成 1280×720 BGR565，走正式 bridge 的 `MVR1` 请求入口；warmup 10 次，120 次请求，12 秒。**不是 live PCIe 输入。**

| 方案 | AI 结果率 | latest-only 替换 | 主模型流水线 p50/p95 | 请求到结果 p50/p95 |
|---|---:|---:|---:|---:|
| A BDD | 9.67 Hz（116/120） | 4/130 | 41.8 / 99.6 ms | 46.6 / 135.0 ms |
| B P2 | 9.58 Hz（115/120） | 5/130 | 65.1 / 103.3 ms | 73.2 / 163.5 ms |
| C BDD+Spec | **9.83 Hz**（118/120） | 2/130 | **40.4 / 94.0 ms** | **48.7 / 122.4 ms** |
| D P2+Spec | 8.92 Hz（107/120） | 14/130 | 64.6 / 146.3 ms | 72.8 / 240.2 ms |

Specialist 结果率约 1.0 Hz；单次主预处理+RKNN+DFL/NMS p50 约 48–49 ms（C/D）。

---

## 4. Qt 内置 Raw-replay（60 fps 本地视频）

| 方案 | AI 结果 / 18 s | AI 结果 FPS | 请求到结果 p50/p95 | 主模型 p50/p95 | Qt 渲染回调 FPS | Qt 上传 FPS |
|---|---:|---:|---:|---:|---:|---:|
| A BDD | 288 | **16.00** | 84.1 / 118.1 ms | 47.6 / 66.5 ms | 25.4 | 16.03 |
| B P2 | 186 | 10.33 | 122.2 / 168.0 ms | 75.8 / 96.7 ms | 28.2 | 15.96 |
| C BDD+Spec | 276 | 15.33 | 85.9 / 134.5 ms | 47.5 / 67.9 ms | 22.7 | 17.30 |
| D P2+Spec | 182 | 10.11 | 122.7 / 198.0 ms | 76.5 / 95.8 ms | 28.0 | 16.50 |

> 这是短测；Qt 渲染回调 FPS 不是物理扫描率。AI 结果 FPS 是 bridge 结果率，不是 NPU 原始 FPS。

---

## 5. 四路文件回放（不是 PCIe 四路）

来自 2026-09-17 的异构四路实测：

```text
WRR-FAIR：27.91/28 FPS，达成率 0.997
FRONT 9.96 FPS；侧后约 6 FPS
1200 s：27.30/28 FPS，FRONT 9.85 FPS
```

预算：

- 四路共享一个 NPU，聚合上限约 27.5 FPS；
- 均分时每路约 6.4–6.9 FPS；
- 要让 FRONT 到 10 FPS，必须加权，让侧路让出时隙；
- 四路并发时 320×192 单次推理从 16.12 ms 升到 20.2–20.7 ms（+25%）。

**边界：这是文件回放 + 单板 AI，不是四路独立 PCIe 输入。**

---

## 6. 资源与瓶颈

### 6.1 当前瓶颈

1. **NPU 100%**：10 Hz 主模型已把 NPU 打满；再叠加 Specialist/四路会抢资源。
2. **CPU 单核 69.5%**：AI sidecar 的预处理/DFL/NMS/状态构建在 A55 上很重；Qt 渲染也抢 CPU。
3. **PCIe Gen2 x1**：LnkSta 5GT/s x1（IP 支持 x4，但实际 downgraded x1）；1280×720 RGB565@49.5fps 约 91 MB/s，链路能承载，但余量有限。
4. **导航 OCR 1.2 s**：0.1 Hz，无法做实时导航。
5. **HMI 与 AI 竞争**：历史测试 10 Hz AI 时 Qt 降到 30–36 FPS；5 Hz 时 42–44 FPS。当前截图 56 FPS，但 NPU 已满，不能保证长时间稳定。

### 6.2 带宽计算

```text
1280 × 720 × 2 bytes = 1,843,200 B/frame
@ 49.49 FPS ≈ 91.2 MB/s
@ 60 FPS    ≈ 110.6 MB/s
PCIe Gen2 x1 理论 500 MB/s（8b/10b 后约 500 MB/s 量级）
```

视频本身不是瓶颈；瓶颈在 NPU/CPU 后处理和显示调度。

---

## 7. 能说什么 / 不能说什么

**能说：**

- 当前 A 模型在 RK3566 实板 10 Hz 运行，主 RKNN p50 29.15 ms。
- 视频采集 49.49 FPS，HMI 自报 56 FPS。
- HUD V2 BAR1 写入被接受/应用（applied seq 增长，reject=0，error=0）。
- HSST 车道邮箱在实时更新。
- B/C/D 已在板端回放验证。

**不能说：**

- 不能说四路 PCIe 实时 27.9 FPS（那是文件回放）。
- 不能说 HMI 56 FPS 等于物理 HDMI 扫描率。
- 不能说 AI 10 Hz 等于显示 10 Hz（显示是 56 FPS，检测框保持/插值）。
- 不能说旧 YOLO11n 320/416 指标属于 B05N/P2。
- 不能说当前红绿灯/导航/车道已端到端可用。

---

## 8. 证据

- `docs/deployment/evidence/adas_A_live_default_summary_20261028.json`
- `docs/deployment/evidence/current_live_summary_20261028.json`
- `docs/deployment/evidence/real_pcie_rga_2_20261025_summary.json`
- `docs/models/evidence/B05P/B05P_bridge_runtime_20261027_{A,B,C,D}.json`
- `docs/models/evidence/B05P/B05P_qt_raw_replay_runtime_20261027.json`
- `docs/hmi/assets/current_live_20261028.png`
- 历史：`docs/competition_ai_inventory.md`、`docs/competition_multistream_budget.md`、`docs/rk3566_innovation_audit_20261025.md`
---

## 9. LVT1 升级测试环境性能（不是生产）

2026-10-29 在实板独立目录 `/home/cat/qt_gpu_probe_lvt1_dev_20261028` 测试，**未停止生产服务**。测试同时运行：

- 生产 Qt / 生产 AI；
- 测试 Qt；
- 测试 AI sidecar（端口 38280/38281，A 模型 5 Hz）；
- 本地 PGVF publisher（60 帧 1280×720，30 FPS）。

单路测试 JSON `docs/deployment/evidence/lvt1_fixture_mode_0_20261029.json`：

| 指标 | 数值 |
|---|---:|
| source_fps | 20.48 |
| Qt scene-graph callback | 10.24 FPS |
| AI result (Qt) | 4.36 FPS |
| AI result (test sidecar summary) | 4.07 Hz |
| NPU | 100% |
| GPU | 81% |
| CPU（整机） | 15.36% |
| RSS | 191.82 MB |
| duration | 16.37 s |

原因：两个 AI sidecar 共用 NPU，本地 publisher + 双 Qt，不是生产配置。  
**不能把 10.2 FPS / 4.36 Hz 当作升级后生产性能。** 生产基线仍是 49.49 FPS 视频 / 9.98 Hz AI / 56 FPS HMI；本次未部署、未重测生产。

截图与详情见 `docs/hmi/LVT1_REAL_DATA_HMI.md`。
