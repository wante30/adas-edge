# FPGA ↔ RK3566 接口说明

更新日期：2026-10-28  
FPGA：紫光同创 PG2L100H-6FBG676（交接文档记录）  
RK3566：LubanCat-1，PCIe Root Complex  
IP：VID:DID `0755:0755`，BAR0 8 KiB，BAR1 4 KiB，IP 支持 Gen2 x4  
实测链路：**5 GT/s x1（downgraded）**

> FPGA 完整工程在本机不可访问：交接文档记录的 `D:/QiansaiFPGA/source_code_3/PCIE_timig_fix` 不存在。本文只记录可从交接文档、RK 驱动和 BAR1 读回验证的接口；不编造 RTL/时序/资源。

---

## 1. 物理与枚举

```text
01:00.0 Memory controller [0580]: Device [0755:0755]
  Region 0: Memory at ... (32-bit, non-prefetchable) [size=8K]   BAR0
  Region 1: Memory at ... (32-bit, non-prefetchable) [size=4K]   BAR1
  Region 2: Memory at ... (64-bit, non-prefetchable) [size=8K]
  LnkCap: Port #0, Speed 5GT/s, Width x4
  LnkSta: Speed 5GT/s (ok), Width x1 (downgraded)
  Kernel driver in use: pango_video
```

- RK3566 是 Root Complex，FPGA 是 Endpoint。
- 实际 x1，不能按 x4 估算带宽。
- BAR0 是 PIO RAM；BAR1 是扩展 CSR/DMA 窗口。
- 控制写必须是**单条 4 字节对齐 DWORD 写**（TLP length=1 DWORD，FirstBE=0xf，LastBE=0x0）；不要用 64/128-bit store 或 `memcpy` 批量写 CSR。
- 小端；`0x00780003` 的内存字节是 `03 00 78 00`，不要额外 `htonl()`。

---

## 2. 地址窗口

| BAR1 偏移 | 用途 | 访问 | 当前软件 |
|---|---|---|---|
| `0x100..0x140` | 传统 RK→FPGA DMA | — | 旧驱动；当前未用 |
| `0x200..0x240` | 传统 FPGA→RK DMA | — | 旧驱动；当前未用 |
| `0x400..0x428` | 视频 DMA 全局控制/统计 | RW/RO | `pango_video` |
| `0x500..0x57c` | 视频 DMA 四槽描述符 | RW | `pango_video` |
| `0x600..0x634` | HUD 灯光/能力 | RW/RO | `hud_bridge.py` |
| `0x638..0x64c` | RK 场景/时间/commit | RW | `hud_bridge.py` |
| `0x650..0x664` | 行为/诊断/HSST 状态 | RO/RW | `pcie_protocol_service.py` |
| `0x670..0x67c` | 右侧导航 | RW | `hud_bridge.py` |
| `0x680..0x68c` | HSST 车道邮箱 | RO | 当前只读探测；HMI 未消费 |
| `0x700..0x75c` | ISP 四 tile 配置/事务 | RW | `pcie_isp_service.py` / `pcie_protocol_service.py` |
| `0x760..0x78c` | ALGO 事务/快照 | RW | 同上 |

> 新 MMIO ioctl 白名单只开放 `0x700..0x7fc`、`0x670..0x67c`、`0x630`；基本 HUD、场景、行为、HSST、视频 DMA 未完整开放给该 ioctl。不要假设“RTL 有寄存器 = 当前驱动能访问”。

---

## 3. HUD V1/V2

### 3.1 基本 HUD（0x600–0x634）

| 偏移 | 名称 | 访问 | 说明 |
|---:|---|---|---|
| `0x600` | MAGIC | RO | `0x31445548`（"HUD1"） |
| `0x604` | VERSION | RO | `0x00010000` |
| `0x608` | REQUEST | WO | 灯光完整请求，序号在高 16 位 |
| `0x60c` | STATUS | RO | accepted/applied/effective |
| `0x610` | ACCEPTED_SEQ | RO | 最后接受序号 |
| `0x614` | APPLIED_SEQ | RO | 真实 overlay 锁存序号 |
| `0x618` | RK_STATE | RO | 最后应用的低 13 位 |
| `0x61c` | EFFECTIVE_STATE | RO | 实际显示 9 bit |
| `0x620` | REJECT_COUNT | RO | 累计拒绝 |
| `0x624` | LAST_ERROR | RO | 最后拒绝错误 |
| `0x628` | REJECTED_SEQ | RO | 被拒序号 |
| `0x62c` | SNAPSHOT_ID | RO | 状态变化计数，不是全局版本 |
| `0x630` | CAPS | RO | `0x000001ff` |
| `0x634` | LAST_ACCEPTED_REQUEST | RO | 最后接受的完整请求 |

实机读回：

```text
0x600 = 0x31445548
0x604 = 0x00010000
0x630 = 0x000001ff
0x620 = 0
0x624 = 0
```

### 3.2 场景（0x638–0x64c）

| 偏移 | 名称 | 说明 |
|---:|---|---|
| `0x638` | SCENE_SHADOW | 感知字段；只写不生效 |
| `0x63c` | TIME0_SHADOW | BCD 日期 `0xMMDDYYYY` |
| `0x640` | TIME1_SHADOW | BCD 时间 `0x00SSMMHH` |
| `0x644` | SCENE_COMMIT | 低 16 位非零序号，原子冻结 |
| `0x648` | SCENE_CDC_STATUS | bit0 pending / bit1 ready |
| `0x64c` | SCENE_APPLIED_SEQ | 真实锁存序号 |

实机 `0x64c` 递增，`0x648` 有状态；具体值见 inventory。

### 3.3 导航（0x670–0x67c）

- shadow + commit + status/active；
- 方向/距离；序号高 16 位允许 0；
- 导航是提示，不是立即横向动作。

### 3.4 HUD V2 场景字段

`0x638` 的位定义（交接文档）：

| 位 | 字段 |
|---:|---|
| 0 | perception_enable |
| 1 | traffic_red |
| 2 | traffic_green |
| 3 | pedestrian |
| 4 | static_target |
| 5 | overspeed |
| 6 | lane_left_valid |
| 7 | lane_left_dashed |
| 8 | lane_right_valid |
| 9 | lane_right_dashed |
| 10 | lka |
| 11 | adas_enable |
| 12 | hud_enable |
| 13 | takeover |
| 14 | blink |
| 15 | reserved |

> HUD V2 没有黄灯字段；不要把黄灯硬塞进红灯/绿灯。当前软件将黄灯保持 UNKNOWN。

---

## 4. HSST 车道邮箱（0x680–0x68c）

| 偏移 | 名称 | 说明 |
|---:|---|---|
| `0x680` | SEQUENCE | 32 位发布序号 |
| `0x684` | PAYLOAD | 左右数量/最近类型/有效位 |
| `0x688` | STATUS | bit0 valid，bit1 stale |
| `0x68c` | PROTOCOL_ID | `0x4c4d0001` |

`0x684` 位域：

```text
bits 7:0   左边界数量
bits 15:8  右边界数量
bits 17:16 左最近类型 (0 unknown, 1 solid, 2 dashed)
bits 19:18 右最近类型
bit 20     左数量有效
bit 21     右数量有效
bit 22     左类型有效
bit 23     右类型有效
bits 31:24 保留 0
```

实机样本：

```text
0x680: 0x2bad9 → 0x2baf7 → 0x2bb15   （约 30 Hz）
0x684: 0xfa0202 → 0xfa0102 → 0xf50101
0x688: 0x1
0x68c: 0x4c4d0001
```

说明：

- HSST 整包新鲜时优先于 RK 场景车道；
- 新鲜但某侧未知/无效时，不能回填 RK 旧虚线；
- HSST 超时/掉链才回退 RK；
- 四个寄存器分别 MMIO 读取不是严格原子快照。

**未证实**：HSST 的算法是否为 MobileNet、是否输出坐标/曲率、训练数据/模型文件。当前只能证明“车道边界数量/类型邮箱在实时更新”。

---

## 5. ISP / ALGO（0x700–0x78c）

- ISP：四 tile 配置、global/tile shadow、CLEAR、COMMIT(APPLY)、QUERY/readback；
- ALGO：去雾/高斯/中值/Sobel 等 64-bit 配置；
- 事务状态：`submitted/applied/rejected/timeout_unconfirmed`；
- 实机历史：ISP/ALGO no-op APPLY 返回 status `0x00020002`，READ 成功；
- 当前正式 HMI 的 FPGA CTRL 页面可操作；
- 不要在 BAR1 读回 `0xffffffff` 时写。

---

## 6. 视频 DMA

- `0x400..0x428`：START/STOP、generation、背压/丢帧计数；
- `0x500..0x57c`：四槽描述符（64-bit DMA 地址、容量、所有权、帧状态）；
- `pango_video` 驱动 + `pango_video_capture` 当前使用；
- 实机采集 49.49 FPS；
- 当前是一路 1280×720 2×2 mosaic，不是四路独立流；
- 描述符写/legacy DMA 在当前调试服务中被阻止。

---

## 7. 软件接口

| 功能 | 文件 | 说明 |
|---|---|---|
| HUD V2 | `hud_bridge.py` | UDP 38262，root，BAR1 |
| 协议 v1.0.1 | `pcie_protocol.py` | 寄存器定义/编码/测试向量 |
| 调试服务 | `pcie_protocol_service.py` | HTTP + UDP 38264，HUD/SCENE/NAV/ISP/ALGO/DIAG/HSST |
| 旧 ISP 服务 | `pcie_isp_service.py` | 兼容 launcher，避免双 BAR1 owner |
| 控制服务 | `pcie_control_service.py` | 带 Bearer token 的 HTTP 控制 |
| 只读探针 | `tools/hud_bar1_probe.py` | magic/caps/applied/reject/error |

---

## 8. 未验证/缺失

1. FPGA 完整工程、bitstream hash、P50/P95 时序、资源利用率。
2. FPGA HDMI OSD 的肉眼可见性（只有 BAR1 applied seq）。
3. 四路独立采集/DMA（当前是 2×2 mosaic）。
4. MobileNet 车道模型细节/坐标输出。
5. 以太网动作链路（交接文档提到 Ethernet > PCIe 优先级；当前仓库没有实现）。
6. 旧 UAPI ioctl 11/12 的完整白名单（当前只开放部分窗口）。
7. FPGA 端对 ADSR/ADST 的消费实现。

---

## 9. 安全规则

- 只写文档化窗口；DMA/PIO 写默认阻止。
- BAR 读回全 `FF` / magic 不匹配 → 停止写。
- 控制写单 DWORD、4 字节对齐。
- 不把 `0x650/0x654` 当动作写。
- 不在生产服务运行时启动第二个 BAR1 writer。
- 不暴露无 token 的 MMIO HTTP API。

---

## 10. 证据

- `deployment/lubancat/qt_gpu_probe/RK3566_PCIe_protocol_handoff.md`
- `deployment/lubancat/qt_gpu_probe/RK3566_PCIe协议开发交接.md`
- `docs/pcie_protocol_matrix_20261024.md`
- `docs/pcie_debug_console_v101_20261025.md`
- `docs/audit/PROJECT_INVENTORY.md`（HUD/HSST 实机读回）
