# RK3566 → FPGA 比赛信号审查与接入状态

日期：2026-10-15。范围：原生比赛入口、HUD v1、ADST v1、早期 ADSF/ADSR、便携 C 编码器与 PC 参考接收器。

**最终更新：2026-10-16，已补充原生决策到 HUD 图标的显式适配与 PC 参考仿真，127项相关回归和原生窗口交互检查通过。** 初次审查的“没有发送调用”描述的是已保存 SHA 对应的旧入口。没有接入 RK3566 或 FPGA，未验证正式驱动、RTL、PCIe 或 HDMI。

## 1. 结论与接口范围

HUD/ADST 已定义的字节格式和事务保护通过本机复测；原生比赛动作与旧接口之间存在明确语义差异，必须显式转换。

HUD v1 表达可见、左右转向、双闪和车灯图标，没有 STOP、FORWARD、BRAKE、灯色、实虚线、完整报警原因或源帧的独立字段。完成 `native → HUD → PC 仿真` 后，也只能证明受支持图标的映射和参考提交正确，不能宣称完整比赛决策已送达 FPGA。完整比赛状态仍需版本明确的独立合同以及硬件消费实现。

## 2. 初次审查发现

| 项目 | 已核实事实 | 处理状态 |
|---|---|---|
| 原生入口没有发送调用 | 初审版本 `tools/run_competition_console.py` 决策后只渲染和写 JSON，无 competition_io 导入或实际发送调用；SHA 见原始证据 | 已接入显式HUD图标适配与PC仿真；正式驱动仍未接入 |
| LED 位图不兼容 | 比赛 LED 左/直/右为 1/2/4；HUD 位0可见、位1左、位2右。直接复制会丢左箭头、把直行变成左箭头 | 已按动作语义显式映射，新增15组用例验证；不直接复制LED位图 |
| 旧 ADST 不适合直接承载新原生策略 | 旧 LEFT/RIGHT 表示变道并要求相应虚线；FORWARD/LEFT/RIGHT 均要求关联绿灯 | 保留旧 ABI；原生路口转弯和无灯道路直行不可强制编码成旧语义 |
| ADSF 文档布局错误 | 旧文档 width/height 为 u16、stride 为 u32；实际分别为 u32、u32、u16，头部52字节 | 已按现有 struct 修正文档及精确偏移表，代码 ABI 不变 |
| 旧描述符验证不足 | ADSR 接受枚举255、非零 reserved、confidence=2.0；1280×720 帧仅3字节 payload 也可解包；ADSR无CRC | 此次未改旧代码；文档明确其打包/mock边界，正式消费端仍需完整验证 |
| 旧 mock 没有消费保护 | MockTransport 为队列和编解码，无定时TTL、预期源帧匹配或跨会话防旧包逻辑 | 文档已删去“已经执行”的表述，明确这些是待实现消费者合同 |

本次没有证据表明真实 FPGA 已收到错误方向；初审发现的是接线缺口和接线前的不兼容。不能把软件复现的错误映射案例描述成实机故障。

## 3. 已完成的本机复测

- 58 项已有协议相关测试通过，覆盖 HUD 编码与事务、ADST、工具和旧描述符基础回归；运行结果在原始证据的 `pytest.txt`。
- Windows GCC C99 严格编译通过；9组HUD向量、4096组state/blink组合、C/Python ADST双向逐字节一致通过。不是 ARM 编译或 FPGA 执行。
- HUD 验证了 accepted 与帧边界 APPLIED 分离、像素时钟停止时UNKNOWN保留在途、查询原请求、复位不重放、65535→1序号回绕、去雾覆盖和闪烁。
- ADST 验证了小端60字节、CRC32、字段有效性、重复包不续期、未来时间拒绝、时钟域错误、倒序/冲突序号、旧源帧忽略、session reset。
- C头文件只有编解码，Python接收器的时效/授权策略不自动存在于 C 或 RTL 中。

原始可复现证据目录：`C:\Users\JingyuWang\adas-edge\outputs\competition_cockpit_20261015\link_audit`。

| 文件 | 用途 |
|---|---|
| `audit.md` | 初次审查快照，保留发现经过，不随最终实现改写 |
| `reproductions.json` | 复现结果、入口及SDK源码SHA |
| `reproduce.py` | 只构造PC数据和参考模型的复现脚本 |
| `pytest.txt` | 58项既有回归结果 |
| `codecs/codec_interop.json` | C/Python互操作结果与编译参数 |

## 4. 本轮补充接入的验收门槛

1. 原生循环将同一个 `Decision` 交给显式适配器。左转与左变道可共享左箭头图标，但完整动作类型必须在结构化状态中保持；直行、停车、制动不能编码成不存在的 HUD 图标。
2. 报警原因与图标表现分别记录。若用双闪作展示提示，必须说明该映射是显示策略，不等于HUD已承载全部报警类型。
3. 输入动作、发送字、端序、序号、参考接收解码、APPLIED/EFFECTIVE及覆盖状态逐项关联。超时不能当成功；UNKNOWN在途不能被新写覆盖；释放/复位不重放旧命令。
4. 用真实视频驱动的原生决策检查受支持图标，同时用明确标识的软件夹具覆盖所有动作/报警。真实视频、模拟按键、规则夹具、参考寄存器必须在证据里分开。
5. 本轮结果只标 PC 仿真。没有实际驱动适配器时不得打开假设备、猜测 ioctl、映射 `/dev/mem` 或宣称已板上发送。

### 最终验证状态

新增 `competition_io/scenario_hud.py`、`tools/audit_scenario_hud.py`，原生入口通过 `--mock-hud` 将同一个决策投影到 HUD 图标，并读取参考接收器的 accepted/applied 状态。默认启动脚本已启用软件接收校验，硬件写入次数始终为0。

- 15组编码→提交→接收解码→读回用例通过，证据 `outputs/scenario_hud_20261015/audit/scenario_hud_audit.json`。
- 包括新增显示与适配的127项相关回归通过；原生窗口的鼠标切场景、暂停时视图开关、红灯违规、停止、逐帧、重播及HUD软件收据核验通过，证据 `outputs/competition_cockpit_20261015/gui_final/gui_check.json`。
- 当前编码：左箭头 state=0x103/blink=1，右箭头0x105/2，直行或普通停止0x101/0，报警0x109/4。HUD没有独立直行/停车图标，0x101只表示相关显示可见，绝不代表完整动作已编码。
- 完整动作、灯色、左右边界、源帧、源时间和有效期保留在 JSON sidecar。sidecar没有通过PCIe发送；没有生成旧ADST包。APPLIED是状态锁存确认，EFFECTIVE/覆盖/最终画面仍应分别核对。
- 真实视频红/绿片段的 `trace.jsonl` 每条记录带 `hud_software_receipt`，可将原生决策、发送字、小端字节、参考接收和未承载字段逐项对照。

### 失联保持旧图标：尚未解决的硬件合同

已在参考模型复现：左箭头应用后不再提交，推进约10秒，旧状态仍保持。HUD1没有源时间、TTL或心跳失联看门狗。新增适配器能在提交前拒绝过期、未来时间和不一致决策；视频自然结束会提交失效显示，但无法在RK崩溃/PCIe失联时替FPGA主动清旧状态。

必须在正式驱动/FPGA合同中明确心跳、超时、复位会话和失联时的安全显示，并在板上拔链路/停进程验证。不能仅用主机定时发清除包来解决主机本身失联的问题。该项列为P0，不算本次软件映射完成后的已解决项。

## 5. 上板仍需验收

| 层级 | 必须取得的证据 |
|---|---|
| RK用户态与驱动 | 芯片/系统/驱动/runtime/模型版本，ARM构建运行，设备互斥，MMIO顺序，posted write回读，remove/reset通知 |
| PCIe与FPGA寄存器 | 真实设备/BAR/窗口长度，寄存器偏移及字节序，物理读写，seq/accepted/applied对应，复位和断流 |
| FPGA消费与显示 | CDC、实际时基、周期过期处理、去雾覆盖、像素时钟停止恢复、最终LED/HDMI可见结果 |
| 完整比赛状态 | STOP/FORWARD/BRAKE、转弯/变道、灯色、边界、报警原因、源帧/有效期的独立协议及消费者 |
| 端到端 | 同一source_frame_id上的感知→决策→发送→接收→应用→可见画面，延迟实测和输入中断回退 |

## 6. 复现命令

```powershell
Set-Location 'C:\Users\JingyuWang\adas-edge'
.\.venv\Scripts\python.exe outputs/competition_cockpit_20261015/link_audit/reproduce.py
.\.venv\Scripts\python.exe -m pytest -q tests/test_hud_v1_encoding.py tests/test_hud_transactions.py tests/test_adas_status_v1.py tests/test_hud_pc_tools.py tests/test_competition_scene_protocol.py
.\.venv\Scripts\python.exe tools/verify_hud_codecs.py --cc C:/msys64/ucrt64/bin/gcc.exe --output outputs/competition_cockpit_20261015/link_audit/codecs_rerun
```

复现脚本重新运行会扫描当前入口版本，因此其入口结果可能随主任务进展变化；SDK缺口复现与原始SHA可据此核对。将本轮新增HUD适配判定为完成时，应以主任务最终发收测试为准。
