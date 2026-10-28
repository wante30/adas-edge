# PCIe 数据流

更新日期：2026-10-28  
范围：当前实机 PCIe 视频/控制链路、RK3566 本地 AI 链路、软件 ADSF/ADSR 契约。  
关键：**不要把“本地 UDP 回环”写成“PCIe 传输”**；不要把“软件 mock”写成“FPGA 联调”。

---

## 1. 实机链路总览

```text
┌────────────── FPGA (PG2L100H) ──────────────┐
│ 采集/拼接/ISP → 视频 DMA → BAR1 控制/HSST     │
│ PCIe Endpoint 0755:0755                      │
└───────────────┬─────────────────────────────┘
                │ PCIe Gen2 x1（实测 5GT/s x1）
┌───────────────▼─────────────────────────────┐
│ RK3566 Root Complex                          │
│  pango_video.ko → /dev/pango_video0          │
│  pango_video_capture → PGVF                  │
│  Qt ← PGVF；Qt ↔ ai_bridge（UDP 回环）       │
│  ai_bridge → HUD bridge → BAR1              │
└─────────────────────────────────────────────┘
```

实机枚举：

```text
01:00.0 Memory controller [0580]: Device [0755:0755]
  Region 0: BAR0 8K
  Region 1: BAR1 4K
  Region 2: 64-bit 8K
  LnkCap: 5GT/s x4
  LnkSta: 5GT/s x1 (downgraded)
  Kernel driver in use: pango_video
```

> IP 支持 x4，实际链路 x1。所有吞吐结论必须按 x1 理解。

---

## 2. FPGA → RK3566：视频

### 2.1 内核/用户态路径

```text
FPGA video DMA
  → pango_video.ko
  → /dev/pango_video0
  → pango_video_capture --continuous --fps 60 --publish-dir /dev/shm/pango-video
  → /dev/shm/pango-video/latest.pgv
  → Qt VideoItem 读取/CRC 校验/上传纹理
```

采集程序：

```text
/home/cat/pcie_test/pango_video_t05/pcie_video/app/pango_video_capture
```

状态文件：

```json
{"state":"running","frames":445008,"fps":49.492,"error":"","last_success_ns":...}
```

### 2.2 PGVF 格式（实机）

40 字节小端头 + 1,843,200 字节 RGB565 payload + CRC32：

| 偏移 | 字节 | 字段 | 说明 |
|---:|---:|---|---|
| 0 | 4 | magic | `PGVF` |
| 4 | 4 | version | 1 |
| 8 | 4 | width | 1280 |
| 12 | 4 | height | 720 |
| 16 | 4 | source_frame | 源帧号 |
| 20 | 4 | generation | 回放/会话代号；变化时 Qt 重置跟踪 epoch |
| 24 | 4 | crc32 | payload CRC32 |
| 28 | 4 | payload_bytes | 1,843,200 |
| 32 | 8 | timestamp_ns | 源时间戳（ns） |
| 40 | 1,843,200 | pixels | RGB565 little-endian，逐行无 padding |

Qt 校验：

- `stat` 文件大小必须等于 40 + 1,843,200；
- magic/version/width/height/payload_bytes/timestamp 必须合法；
- 默认校验 CRC32（`QT_PGVF_SKIP_CRC=1` 可跳过，仅本地调试）；
- `generation` 变化 → 新会话，清空跟踪；
- `source_frame` 回退/重复 → 丢弃。

### 2.3 带宽

```text
1,843,200 B/frame × 49.49 FPS ≈ 91.2 MB/s
1,843,200 B/frame × 60 FPS    ≈ 110.6 MB/s
PCIe Gen2 x1 理论 ~500 MB/s（编码后）
```

视频带宽不是当前瓶颈；瓶颈在 NPU/CPU 后处理和显示调度。

---

## 3. RK3566 → FPGA：HUD/控制

### 3.1 实际写回路径

```text
Qt / AI 状态
  → hud_bridge.py（UDP 127.0.0.1:38262）
  → mmap BAR1 resource1
  → HUD V2 寄存器
  → FPGA OSD/LED/状态机
```

Qt 不直接 mmap BAR1；root bridge 负责：

- 校验 HUD magic `0x31445548`；
- 只在白名单窗口写；
- 事务顺序：shadow → COMMIT → 轮询 applied；
- BAR 读回全 `FF` / magic 不匹配时拒绝写。

### 3.2 HUD V2 关键寄存器

| 偏移 | 名称 | 访问 | 说明 |
|---:|---|---|---|
| `0x600` | MAGIC | RO | `0x31445548` |
| `0x604` | VERSION | RO | `0x00010000` |
| `0x608` | HUD REQUEST | WO | legacy HUD/ADAS 位 |
| `0x60c` | HUD STATUS | RO | accepted/applied/effective |
| `0x610/0x614` | ACCEPTED/APPLIED SEQ | RO | 握手序号 |
| `0x620/0x624` | REJECT COUNT/LAST ERROR | RO | 诊断 |
| `0x630` | CAPS | RO | `0x000001ff` |
| `0x638` | SCENE_SHADOW | RW | 感知字段 |
| `0x63c/0x640` | TIME0/TIME1 | RW | BCD 时间 |
| `0x644` | SCENE_COMMIT | RW | 低 16 位非零序号 |
| `0x648` | SCENE_CDC_STATUS | RO | pending/ready |
| `0x64c` | SCENE_APPLIED_SEQ | RO | 真实锁存序号 |
| `0x658` | SCENE_STATUS | RO | ready/stale/error/HSST owner |
| `0x664` | CAPS | RO | `0x0002003f` |
| `0x680` | HSST SEQUENCE | RO | 车道邮箱序号 |
| `0x684` | HSST PAYLOAD | RO | 左右数量/最近类型 |
| `0x688` | HSST STATUS | RO | valid/stale |
| `0x68c` | HSST PROTOCOL_ID | RO | `0x4c4d0001` |

### 3.3 实机 HUD 证据

```text
0x600 = 0x31445548
0x604 = 0x00010000
0x630 = 0x000001ff
0x64c = 0x00004b9d   (applied seq 递增)
0x620 = 0
0x624 = 0
```

说明：HUD 写入被接受/应用，无拒绝/错误。  
**但不能由此证明 FPGA HDMI 上的 OSD 肉眼可见**；需要 FPGA HDMI 采集/拍照。

---

## 4. FPGA → RK3566：HSST 车道邮箱

实机读取（连续 3 次，约 1 s 间隔）：

```text
0x680 sequence: 0x2bad9 → 0x2baf7 → 0x2bb15   （递增，约 30 Hz）
0x684 payload:  0xfa0202 → 0xfa0102 → 0xf50101
0x688 status:   0x1（valid，not stale）
0x68c protocol: 0x4c4d0001
```

按交接文档解码：

- bits7:0 左边界数量；
- bits15:8 右边界数量；
- bits17:16 左最近类型（0 未知/1 实线/2 虚线）；
- bits19:18 右最近类型；
- bits20/21 左右数量有效；
- bits22/23 左右类型有效。

结论：**FPGA 确实在实时发布车道边界数量/类型**。  
未证实：算法是不是 MobileNet、坐标/曲率、稳定性、与 RK 检测的车道归属。

---

## 5. RK3566 本地 AI 链路（不是 PCIe）

### 5.1 MFR1 共享帧

Qt 把当前主视帧写成 MFR1，供 AI sidecar 读取：

```text
MFR1 header: <4sQQII> = 28 bytes
  magic "MFR1"
  frame_id u64
  timestamp_ns u64
  stride_bytes u32 (= 2560)
  payload_bytes u32 (= 1,843,200)
payload: 1280×720 RGB565
```

文件：`/dev/shm/qt_gpu_probe_ai.mfr1`。

### 5.2 MVR1/MVR2 请求

```text
MVR1: <4sQQ> = 20 bytes
  magic, frame_id, timestamp_ns
MVR2: <4sQQff> = 28 bytes
  magic, frame_id, timestamp_ns, input_fps, submitted_fps
```

UDP：`127.0.0.1:38260`。

### 5.3 MVA1/MVA2 结果

```text
MVA1 header: <4sQQIIB3xf> = 36 bytes
  magic, frame_id, timestamp_ns, count, flags, nav_direction, nav_distance_m
MVA2 header: <4sQQIIB3xfI> = 40 bytes
  + status_json_bytes
box V1: <7f> = 28 bytes
box V2: <7fIf> = 36 bytes (7 floats + class_id + confidence)
```

UDP：`127.0.0.1:38261`。  
`status` 是 JSON（≤4096 B），包含检测计数、交通灯、导航、模型选择等。

### 5.4 帧同步

- Qt 只对最新帧发请求；sidecar latest-only 队列可替换旧请求。
- 结果带 `frame_id`/`timestamp_ns`；Qt 丢弃过期结果。
- 当前 A 方案：请求 10 Hz，结果 9.98 Hz。

---

## 6. 软件 ADSF/ADSR 契约（mock）

`competition_io/protocol.py` 定义的是**软件层**帧/结果描述符，不是当前实机 PGVF/HUD 路径：

### 6.1 ADSF 帧头（52 B）

```text
<4sHHIQQII BBH III>  (packed, little-endian)
magic "ADSF", version 1, header_len 52
stream_id u32, frame_id u64, timestamp_ns u64
width u32, height u32, pixel_format u8, reserved u8, stride u16
payload_size u32, flags u32, crc_placeholder u32
payload...
```

### 6.2 ADSR 结果头（42 B）

```text
<4sHHIQQHBBBBII>
magic "ADSR", version 1, header_len 42
stream_id u32, frame_id u64, timestamp_ns u64
num_objects u16, traffic_light_state u8, lane_state u8,
adas_action u8, reserved u8, warning_flags u32, payload_size u32
detections: 12 B each (class_id u16, conf_q15 u16, x1/y1/x2/y2 u16)
```

状态：**mock/单元测试通过，未在实机 PCIe 使用**。  
`competition_io/mock_transport.py` 只排队序列化数据，不做 TTL/时钟域/硬件 I/O。

---

## 7. 端到端时序（当前实机）

```text
FPGA 源帧
  │  PCIe DMA
  ▼
pango_video_capture 发布 PGVF（≈49.5 FPS）
  │
  ├─ Qt 读 PGVF → 纹理 → 显示
  │
  └─ Qt 写 MFR1 → MVR1(10 Hz) → AI
        → MVA2 → Qt 框
        → HUD bridge(≈2 Hz 自动) → BAR1 → FPGA OSD
```

各段延迟：

| 段 | 实测/估计 |
|---|---|
| 视频发布 | 49.5 FPS（≈20.2 ms 周期） |
| AI 主 RKNN | p50 29.15 ms |
| AI 主模型总耗时 | p50 38.77 ms |
| AI 全工作 | p50 49.79 ms |
| AI 结果率 | 9.98 Hz |
| HUD 自动发送 | 约 500 ms 一次（2 Hz） |
| HUD applied | seq 递增，具体像素延迟未测 |

---

## 8. 能说什么 / 不能说什么

**能说：**

- FPGA→RK 视频经 PCIe 实机运行，PGVF 49.49 FPS。
- RK→FPGA HUD V2 实机写入被接受/应用。
- FPGA HSST 车道邮箱实机实时更新。
- RK 本地 AI 10 Hz，MFR1/MVR1/MVA2 实机运行。

**不能说：**

- 不能说 ADSF/ADSR 已在 PCIe 实机使用。
- 不能说 HUD applied 等于 HDMI 肉眼可见。
- 不能说 HSST 是 MobileNet 或已输出坐标。
- 不能说四路独立 PCIe 流已跑通。
- 不能说本地 UDP 回环是 PCIe 传输。

---

## 9. 证据

- `docs/deployment/evidence/adas_A_live_default_summary_20261028.json`
- `docs/deployment/evidence/current_live_summary_20261028.json`
- `docs/audit/PROJECT_INVENTORY.md`（HUD/HSST 读回）
- `deployment/lubancat/qt_gpu_probe/RK3566_PCIe_protocol_handoff.md`
- `docs/competition_pcie_protocol.md`（ADSF/ADSR mock）
