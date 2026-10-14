# RK3566 / FPGA HUD：本机实现与交接

日期：2026-10-14。当前范围：本机代码实现、协议对照、寄存器参考模型测试。正式 FPGA 工程与 RK3566 驱动源码位于另一台电脑。

## 1. 交付内容与验证边界

已实现 HUD v1 编码、事务确认、超时保留、复位通知、去雾覆盖查询；另外实现独立 ADAS 状态报文 ADST v1 和过期处理。HUD 寄存器按用户提供的 `rk3566_pcie_hud_rk_handoff.md` 实现；ADST 是本次新增的软件接口提案，尚未分配 FPGA 地址或 ioctl。

| 文件 | 用途 |
|---|---|
| `competition_io/hud_v1.py` | HUD 寄存器、状态位、命令编码与九组向量 |
| `competition_io/hud_client.py` | 通过注入的寄存器适配器完成发送与确认 |
| `competition_io/hud_simulator.py` | PC 寄存器参考模型，模拟帧边界、覆盖和异常 |
| `competition_io/adas_status_v1.py` | ADST 编解码、CRC、时钟域、有效期和接收策略 |
| `deployment/hud/*.h` | 可移植用户态 C 编码器和枚举定义 |
| `deployment/hud/codec_selftest.c` | C 向量、边界与双向互操作测试 |
| `tools/hud_control_demo.py` | HUD 全流程演示，输出 JSON / 逐帧 CSV |
| `tools/export_adas_status.py` | 将既有官方视频识别日志转换为 ADST |
| `tools/verify_hud_codecs.py` | 编译 C 并与 Python 逐字节对照 |

当前没有修改正式 RTL、驱动或板卡。Python 寄存器参考模型不能证明 PCIe、CDC、时序收敛、综合资源或 HDMI 输出正确。C 测试使用本机 Windows GCC，并非 ARM 交叉编译。现有检测模型没有重新训练或运行。

## 2. HUD v1 接口保持原定义

设备 `0755:0755`，目标 BAR1 为 4 KiB 非预取内存窗口。HUD 只占 `0x600–0x63F`，不能改写视频寄存器 `0x400–0x428` 或缓冲描述符 `0x500–0x57F`。

| 偏移 | 寄存器 | 语义 |
|---|---|---|
| 0x600 / 0x604 | MAGIC / VERSION | `0x31445548` / `0x00010000` |
| 0x608 | REQUEST | 单个 DWORD 命令，读回为零 |
| 0x60C | STATUS | busy、owner、去雾覆盖、有效标志、ready |
| 0x610 / 0x614 | ACCEPTED_SEQ / APPLIED_SEQ | 已接收 / 已提交到像素域的序号 |
| 0x618 | RK_STATE | 已提交 RK 目标，低 13 位含 owner / blink |
| 0x61C | EFFECTIVE_STATE | 实际可见控制状态，低 9 位 |
| 0x620 / 0x624 / 0x628 | REJECT_COUNT / LAST_ERROR / REJECTED_SEQ | 拒绝历史；错误保持至复位 |
| 0x62C | SNAPSHOT_ID | 一致性快照计数 |
| 0x630 / 0x634 | CAPS / LAST_ACCEPTED_REQUEST | 必需能力 `0x3F` / 最近接收完整字 |
| 0x638 / 0x63C | 保留 | 只读零 |

REQUEST：`sequence[31:16] | owner[12] | blink[11:9] | state[8:0]`。`[15:13]` 必须为零。序号从 1 到 65535，回绕至 1。释放控制权时 owner、blink、state 均为零。

state 的 bit0–8 依次为 HUD 可见、左转、右转、双闪、近光、远光、前雾灯、示宽灯、ADAS 固定文字可见。blink 的 bit0–2 分别对应左、右、双闪，必须同时启用相应图标。

九组完整命令：`00011101`、`00021303`、`00031505`、`00041909`、`000511C1`、`00061111`、`00071121`、`00081000`、`00090000`。

### 2.1 事务与恢复

1. 读取 magic / major version / caps；不兼容时拒绝写入。
2. 用 `SNAPSHOT_ID → 全部状态 → SNAPSHOT_ID` 读取一致快照，最多重试 8 次。
3. 若启动时已有 BUSY，请先确认并等待原请求；单设备始终只有一个在途事务。
4. 写 REQUEST；再读同设备 MAGIC 刷新 posted write。该读取只确认写通路，不证明已显示。
5. 默认每 1 ms 查询，最多等 100 ms；只有序号、完整接收字及 RK_STATE 匹配 APPLIED 记录时才返回成功。
6. 超时、写后异常或确认不一致返回 UNKNOWN 并保留请求。此时 `query_pending()` 查询原请求；发送新的状态会抛出 `OutstandingTransaction`。超时不会自动释放、复位或取消。
7. 像素时钟停止允许持续 BUSY；恢复后才能提交。实际 PCIe reset / remove / PERST / 链路错误由驱动通知 `notify_device_reset()`，取消软件缓存，禁止自动重放旧命令。

APPLIED 表示 RK 目标已经提交。`covered=True` 表示去雾覆盖，不能宣称实际画面等于请求；应同时检查 EFFECTIVE_VALID 与 EFFECTIVE_STATE。
EFFECTIVE_STATE 的 bit0=0 表示 HUD 不覆盖视频；其他位仍是内部锁存值，不代表对应图标可见。

### 2.2 去雾与闪烁

去雾覆盖位为 `0xCF`（可见、左右、双闪、雾灯、示宽灯），优先级保持最高。覆盖必须使用选择器，包括灭相，不能把 RK 常亮箭头 OR 进去。20 帧亮 / 20 帧灭由 FPGA 本地执行，60 Hz 时为 1.5 Hz；精确重复命令和其他灯位更新不重启闪烁。去雾期间仍保存最新已提交 RK 目标，退出覆盖后从亮相恢复。

HUD v1 是手动图标状态，应用退出但链路未复位时保持最后状态，没有租约。正常退出可在没有 UNKNOWN 在途事务时显式 `release()`。实时 ADAS 过期处理见下一节，不能把命令等待 100 ms 当作感知有效期。

## 3. ADST v1：独立感知状态提案

HUD v1 无法表达红绿灯颜色、左右车道类型、动作、报警或源帧。这些信息通过独立 **60 字节**小端报文表达；不能直接写入 HUD 的 0x608，也不能直接复用现有 ADSR 描述符或追加到其尾部。FPGA 端需要另行确定 BAR 寄存器或消息邮箱及接收实现。

| 字节偏移 | 类型 / 长度 | 字段 |
|---|---|---|
| 0 | 4 bytes | `ADST` |
| 4 / 6 | u16 / u16 | version=1 / size=60 |
| 8 / 12 | u32 / u32 | sequence（非零） / stream_id |
| 16 / 24 | u64 / u64 | source_frame_id / source_timestamp_ns |
| 32 / 33 / 34 / 35 | u8 各 1 | light / left_lane / right_lane / action |
| 36 / 38 | u16 / u16 | signal_confidence_q15 / clock_domain |
| 40 / 44 | u32 / u32 | valid / warnings |
| 48 / 52 | u32 / u32 | valid_for_ms / reserved=0 |
| 56 | u32 | IEEE CRC32，覆盖前 56 字节，与 `zlib.crc32` 一致 |

置信度量化为 `round(confidence × 32767)`；有效期默认 750 ms，可配置 1–60000 ms。本机回放的 750 ms 是软件配置，不代表已证明适合板卡链路。

| 字段 | 值 |
|---|---|
| light | UNKNOWN=0、RED=1、YELLOW=2、GREEN=3 |
| lane | UNKNOWN=0、SOLID=1、DASHED=2 |
| action | UNKNOWN=0、STOP=1、FORWARD=2、LEFT=3、RIGHT=4、BRAKE=5、WARNING=6 |
| clock_domain | VIDEO_REPLAY=1、HOST_MONOTONIC=2、FPGA_SOURCE_NS=3 |
| valid | bit0 灯色观察、bit1 本车道信号归属、bit2 左车道类型、bit3 右车道类型、bit4 动作决策 |
| warnings | bit0 红灯、bit1 实线、bit2 行人、bit3 碰撞、bit4 未知灯、bit5 未知车道、bit6 过期、bit7 超速、bit8 归属未验证、bit9 协议错误 |

action LEFT / RIGHT 在当前软件策略中表示已请求的左右变道，不能直接当作导航转弯箭头。道路、导航、车辆动作的数据来源仍需实际接入。

### 3.1 新鲜度与通行权限

- 使用源视频/采集时间判定过期，重复报文不会延长有效期；时序滤波保持的旧灯色也保留原观察时刻。接收侧必须定期调用 `current()`，不能只在收到新包时检查；停止输入后继续显示旧绿色会失去这层保护。
- 每个 `AdasReceiver` 对应一个 stream_id。忽略其他摄像头、旧序号，以及新序号却携带更旧源帧的报文；回放循环、设备重启或时间基准切换必须显式 `reset_session()`。
- CRC、字段或时钟异常会清除当前状态，返回 UNKNOWN 和协议错误。源时间过期或倒退返回 UNKNOWN / STALE。
- 检测到 GREEN 与“确认是本车道控制灯”是两件事。只有 SIGNAL_OBSERVED + SIGNAL_ASSOCIATED + DECISION 有效，才能显示 FORWARD；左右变道还要求相应车道类型已确认是 DASHED。
- 主机 monotonic 时间、视频时间及 FPGA 采集时间不能直接相减。跨 RK/FPGA 消费报文前必须共享时基或实现并验证时间转换；仅把 clock_domain 改成 3 不会建立同步。
- C 头文件只实现报文编解码。消费策略和定时过期逻辑在 Python 接收器中实现；FPGA/C 消费端须按该策略移植，不能只通过 CRC 就授予通行。

## 4. 正式驱动的接入合同

现阶段 `HudClient` 接收 `read32(offset)` / `write32(offset, value)` 对象，没有臆造 ioctl 号或开放 `/dev/mem` 映射。另一台电脑需要在已有驱动中补齐以下实际绑定：

1. 校验目标 PCI 设备和 BAR1 类型、长度至少覆盖 0x640。所有 HUD 访问均为对齐单 DWORD、full byte-enable。仅 REQUEST 可写。
2. 内核通过有序 `readl` / `writel` 访问寄存器；保留同设备读回刷新。Linux 原始语义见 [官方 device-io 文档](https://docs.kernel.org/driver-api/device-io.html)。
3. 驱动对同一个设备实现跨进程独占事务。Python 默认锁仅保护同一客户端；多客户端可共享同一锁，但这不能代替驱动的进程间排他。
4. remove / reset / PCI 错误时先使设备句柄失效，再通知客户端清理事务；恢复后重新 probe。不能因 HUD 超时复位整个视频链路。
5. 填写真实设备节点、ioctl 定义、驱动 commit、FPGA bitstream build_id 和工程路径后再接板。ADST 的物理通路需单独实现，当前没有占用任何地址。

用户态调用示例，`driver_adapter` 需要真实驱动提供上述合同：

```python
from competition_io.hud_client import HudClient
from competition_io.hud_v1 import Icon

client = HudClient(driver_adapter, lock=device_shared_lock)
result = client.submit_full_state(True, int(Icon.VISIBLE | Icon.ADAS_VISIBLE | Icon.LEFT), blink=1)
if result.phase == "UNKNOWN":
    result = client.query_pending()
```

## 5. 本机验证与复现

在 `C:\Users\JingyuWang\adas-edge` 中执行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_hud_v1_encoding.py tests/test_hud_transactions.py tests/test_adas_status_v1.py tests/test_hud_pc_tools.py tests/test_competition_scene_protocol.py --basetemp outputs/pcie_hud_pc_validation_20261014/pytest_run
.\.venv\Scripts\python.exe tools/hud_control_demo.py --output outputs/pcie_hud_pc_validation_20261014
.\.venv\Scripts\python.exe tools/verify_hud_codecs.py --cc C:/msys64/ucrt64/bin/gcc.exe --output outputs/pcie_hud_pc_validation_20261014
.\.venv\Scripts\python.exe tools/export_adas_status.py --input outputs/rk3566_config_sim_long_20261013/full/predictions.jsonl --output outputs/pcie_hud_pc_validation_20261014
```

SDK 和演示工具使用 Python 标准库；测试需要 pytest。原工程回归测试还依赖既有 NumPy / OpenCV。交接包包含新接口的 52 项测试、标准库依赖和本机证据，不携带 ML 模型或完整视频日志。

当前本机验证：58 项测试通过；其中 52 项为本次新接口与工具测试，6 项为现有竞赛策略/描述符回归。C99 编码器在 `-Wall -Wextra -Werror -pedantic` 下编译、运行通过；九组 HUD 字与 ADST 的 C/Python 双向编码一致；C 编码器还检查全部 4096 组 state/blink 组合。

演示验证九组命令、20/20 帧闪烁、去雾灭相、像素时钟停止/恢复、UNKNOWN 禁止新写、释放和复位取消。`hud_frame_trace.csv` 保存 120 帧可见控制状态，属于软件参考模型。

官方长视频 `video_20260929_113539.mp4` 的已有 PC FP32 ONNX 日志包含 2315 帧，导出 772 个主模型刷新点，60 字节每条，共 46320 字节。灯色报文计数 UNKNOWN=633、RED=63、GREEN=70、YELLOW=6；这是未标注视频的观察计数，不是检测准确率。没有本车道信号归属或车道模型结果，772 条动作均为 UNKNOWN。为避免用新包续命旧检测，信号源时间取参与检测器的保守较旧时刻；54 条滤波保持灯色保留原观察时刻，15 条较旧源帧在接收侧被忽略，过期返回 UNKNOWN 已验证。

证据目录：`C:\Users\JingyuWang\adas-edge\outputs\pcie_hud_pc_validation_20261014`。重点查看 `validation_summary.json`、`pytest.txt`、`codec_interop.json`、`hud_demo.json`、`adas_export.json` 和 `official_video_adst.jsonl`。

## 6. 另一台电脑仍需完成的硬件验收

| 项目 | 本机结果 | 后续实测要求 |
|---|---|---|
| 正式 RTL HUD 寄存器 / CDC / 本地闪烁 | 只有 Python 参考模型 | 将上述向量和异常用例移植至 RTL testbench |
| RK 驱动 BAR / ioctl / reset 通知 | 仅定义适配合同 | 接真实驱动源码并在 RK3566 编译 |
| ADST FPGA 消费与过期回退 | 软件提案与接收器已测试 | 定物理通路、时基和接收端，验证停止输入后的回退 |
| bitstream / STA | 未验证 | 原材料提到的负裕量必须在正式工程重新修复并收敛 |
| HDMI 可见输出 | 未验证 | 观察实际 HDMI；上行 PCIe 视频不包含最终 HUD |
| RK3566 识别 + 回传 + 显示延迟 | 未验证 | 分别测源帧年龄、识别时延、命令应用时延和显示效果 |

本次交付完成本机软件验证，不据此宣称板卡部署、FPGA HUD 或整机链路已经完成。
