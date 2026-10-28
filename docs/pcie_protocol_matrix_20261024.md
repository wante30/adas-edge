# PCIe 完整协议第一阶段能力矩阵

依据：`rk3566_pcie_complete_handoff_20261023.json`  
基线：`isp_base_hud_only_20261023`

## 实板状态（2026-10-24，重启恢复后）

```text
01:00.0 [0755:0755]        配置可读，Link up 4.0 Gb/s x1
pango_video                bound，/dev/pango_video0
pango capture              running ~59 FPS
ISP/ALGO HARDWARE          magic=0x31435049 version=1 caps=0xf
```

失效时曾出现：`rev ff`、BAR1=0xffffffff、capture invalid_frame。  
一次受控重启恢复；根因 UNKNOWN（无 AER/panic 记录）。

结论：当前实板 PCIe 链路不可用，不能把 BAR1 读成功当作已完成验收。  
以下“实板”列在没有链路时必须为 NO/UNVERIFIED。

## 功能矩阵

| 功能 | 现有实现 | 可访问 | 可读 | 可写 | 已确认应用 | 实板验收 |
|---|---|---|---|---|---|---|
| HUD v1 0x600–0x634 | `hud_bridge.py` | 是（正常链路） | 是 | 是 | 是（历史） | 历史通过，当前不可测 |
| HUD SCENE/TIME 0x638–0x64C | `hud_bridge.py` 基础 | 是 | 是 | 部分 | 历史 | 当前不可测 |
| DIAG 0x658–0x664 | `protocol()` 只读 | 是 | 是 | 否 | 否 | 否 |
| NAV 0x670–0x67C | `hud_bridge.py nav()` | 是 | 是 | 是 | 是（历史） | 历史通过，当前不可测 |
| FIBER 0x680–0x68C | `protocol()` 只读 | 是 | 是 | 否 | 否 | 否 |
| ISP 0x700–0x75C | `pcie_isp_service.py` | HARDWARE | 是 | 是 | 是 | 是（同值 APPLY/QUERY） |
| ALGO 0x760–0x77C | `pcie_isp_service.py` | HARDWARE | 是 | 是 | 是 | 是（dehaze on/off + 恢复） |
| ISP/ALGO snapshot 0x780–0x78C | 服务已读 | 是 | 是 | 索引写 | 否 | 否 |
| VIDEO 采集 CSR 0x400–0x428 | pango_video/capture | 否（当前） | 只读代码 | 否 | 否 | 否 |
| 4×DMA descriptor 0x500–0x57C | pango_video | 否（当前） | 是 | 否 | 否 | 否 |
| legacy DMA 0x100–0x240 | 旧驱动 | 否 | 否 | 否 | 否 | 否 |
| UAPI ioctl 11/12 | 参考驱动未部署 | 否 | 否 | 否 | 否 | 否 |

## 实际入口

- 视频：`pango_video` + `pango_video_capture`
- HUD/NAV/协议 dump：`hud_bridge.py`，UDP 38262，root mmap resource1
- ISP/ALGO：`pcie_isp_service.py`，UDP 38264，root；PCIe 不可用时自动 simulation
- Qt：`PcieIspControl` + `FpgaControlPanel.qml`，主界面 “FPGA CTRL” 入口
- 退出证据：`tools/run_qt_supervised.sh`

## 已实现事务

```text
ISP:
  QUERY -> validate -> write global/tile shadow -> CLEAR -> COMMIT(APPLY)
  -> poll busy/done/error/sequence -> QUERY readback
ALGO:
  QUERY/APPLY，64bit 配置；stage/tile/W/t_min/A/flags/threshold
NAV:
  SHADOW -> CLEAR -> COMMIT(APPLY) -> STATUS/CONTROL/ACTIVE
```

状态词：

```text
submitted
applied
rejected
timeout_unconfirmed
```

## 当前明确不支持

- 真实 ISP/ALGO 实板写验收：PCIe `rev ff`，无法安全写。
- 四 tile 的逐滑条 UI：服务接口支持四 tile 数组，当前页面只提供全局/预设操作。
- HUD/SCENE/DIAG/FIBER 的统一事务 UI：现有 bridge 仅提供读取/既有 HUD 写入。
- 视频 DMA 描述符写、legacy DMA：按任务要求未改。
- UAPI ioctl 11/12：当前 `/dev/pango_pci_driver` 不存在，使用既有 root BAR1 映射路径。

## 风险与约束

- 不允许在 `0xffffffff` 读回时写任何寄存器。
- ISP 写只允许 0x700–0x7FC 白名单；DMA/视频描述符不写。
- 模拟后端只用于软件联调，不能作为实板验收。
