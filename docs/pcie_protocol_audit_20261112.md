# RK3566 PCIe 协议核对与多灯接入说明

核对日期：2026-11-12

核对依据：FPGA 提供的 `RK_PCIE_FULL_CONTROL_PROTOCOL.md`、工程内
`deployment/lubancat/qt_gpu_probe/rk3566_pcie_protocol_v1.json`，以及板端现有
BAR1 bridge、Qt HUD 客户端和协议编码模块。

## 结论

多灯结果已经接入现有 HUD 场景链路。链路仍然是：

```text
multi_signal.signals[*].color
        -> build_hud_perception(...).traffic
        -> AI status.traffic / VideoItem::aiTraffic
        -> QML pcieHud.sendPerception()
        -> BAR1 0x638 -> 0x63C -> 0x640 -> 0x644
```

没有新增 PCIe 地址、没有把箭头方向写入 `left_lane/right_lane`，也没有修改
TIME0/TIME1 的字段定义。

`traffic` 的映射规则如下：

| 多灯结果 | `traffic` | 说明 |
|---|---:|---|
| 至少一个 `valid=true,color=RED`，且没有其他确认颜色 | 1 | 红灯 |
| 至少一个 `valid=true,color=GREEN`，且没有其他确认颜色 | 2 | 绿灯 |
| 黄灯、未知、过期、没有有效灯 | 0 | 释放/未知 |
| 红绿或黄灯与红绿混合 | 0 | 保守释放，保留冲突证据 |

`shape`、`observed_directions`、`directions`、`track_id` 等多灯细节继续留在
`multi_signal` JSONL/status 中。视觉箭头不等于本车道允许转向，因此不映射到
FPGA 的左右车道线型字段。

## 逐项核对

| 协议区域 | FPGA 偏移/字段 | 当前代码 | 结论 |
|---|---|---|---|
| BAR1 访问 | 4 字节对齐、完整 DWORD、小端 | `hud_bridge.py` 使用 `ctypes.c_uint32`；Qt 只发 UDP JSON | 一致 |
| HUD 基础 | `0x600/0x604/0x608..0x634` | `pcie_protocol.py`、`hud_bridge.py` | 一致 |
| 场景 shadow | `0x638`，traffic 位 `2:1` | `SCENE_BITS`、`Bridge.set()` | 一致 |
| 时间 shadow | `0x63C`、`0x640` | `Bridge.set()`、`HudV2Sender` | 一致 |
| 场景提交 | `0x644[15:0]` 非零序号 | `ext_seq` 递增并写 commit | 一致 |
| 场景确认 | `0x648` pending/ready、`0x64C` applied | bridge 轮询并返回 `ext_applied` | 一致；仍以板端状态为最终确认 |
| 场景动作结果 | `0x658/0x65C/0x660/0x664` | `pcie_control_service.py` 诊断读取，`pcie_protocol.py` 解码行为字段 | 一致 |
| 导航 | `0x670..0x67C` | `hud_bridge.nav()`、`scene` 协议服务 | 一致 |
| HSST 车道 | `0x680..0x68C` | `hud_bridge.protocol()`/`diag()` 只读 | 一致；不由多灯箭头代填 |
| ISP/ALGO | `0x700..0x78C` | 既有 ISP/ALGO 服务 | 本轮未改 |
| 视频/DMA | `0x100..0x240`、`0x400..0x57C` | 既有视频/诊断路径 | 本轮未改 |

当 `time_enable=true` 时，保持 FPGA 要求的四次写入顺序：

```text
0x638 scene
0x63C TIME0
0x640 TIME1
0x644 non-zero sequence
```

当 `time_enable=false` 时，TIME0/TIME1 现在只读回上一次 shadow 值，不再写零，
因此提交新场景会继续使用 FPGA 已保存的时间字段，符合完整协议的 shadow 语义。

## 附件与工程 JSON 的一个冲突

附件第 5 节描述了 `0x6C8` PCIe ACTION 写入窗口；工程内完整契约的
`behavior.pcie_action_write_supported` 明确为 `false`，并注明当前动作来源是
Ethernet behavior mailbox，PCIe 只提供场景/灯光/导航并读取动作结果。

本轮不发送 `0x6C8`，也不新增动作写入。等 FPGA/协议负责人确认两份定义一致后，
再单独设计动作窗口接入；这不会影响当前多灯 `traffic` 场景提交。

## 可复现测试

板端 ROG 工程执行：

```powershell
python -m pytest -q tests/test_fpga_perception_bridge.py \
  tests/test_pcie_protocol_service.py \
  tests/test_hud_v2_sender.py \
  tests/test_hud_bridge_scene.py \
  tests/test_pcie_full_protocol_contract.py
```

本轮结果：`41 passed`（五个定向测试文件合计）。完整工程回归仍需在 ROG
环境执行；没有把未连接 RK3566 的 PC/模拟结果写成板端实测。
