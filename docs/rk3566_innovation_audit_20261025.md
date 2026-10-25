# RK3566 端创新点审计（比赛文档用）

日期：2026-10-25。范围：RK3566 + FPGA PCIe V1.0.1、Qt HMI、四路 AI 调度、
ISP/ALGO、行为/HSST。  
原则：只写能被代码、协议或实测回执支撑的内容；不把 PC/回放结果写成 PCIe 实机结论。

## 一、建议写入比赛文档的 5 个创新点

### 创新点 1：FPGA/RK3566 异构闭环与“干净采集”分工

**问题**：ADAS 既要稳定显示 HUD/导航/告警，又不能让显示叠加污染算法输入。

**做法**：

- FPGA 负责视频采集、四区域 ISP/ALGO、HDMI 端 HUD/NAV/场景 OSD；
- RK3566 负责 YOLO、交通灯专模、OCR、行为/场景决策和车机 HMI；
- PCIe 同时承载视频 PGVF 与控制/状态寄存器；
- HUD/NAV overlay 只存在 FPGA HDMI 输出，**不进入上游 PCIe 采集帧**。

**证据**：

- 协议明文；`qt_gpu_probe/README.md` 明确“HUD overlay 不在上游 PGVF”；
- 实板 BAR1：HUD magic `0x31445548`、ISP magic `0x31435049` 均可读；
- 实板 HUD/SCENE/NAV/ISP/ALGO 事务已收到 accepted/applied 回执。

**比赛表述**：

> 构建了 FPGA 负责确定性显示/图像处理、RK3566 负责感知/交互的异构闭环；
> 显示叠加与算法采集解耦，避免 HUD 反馈进入检测输入。

**边界**：不能说已完成“四路实时 PCIe 全链路”；当前是单路视频采集 + 控制闭环。

---

### 创新点 2：四区域独立 ISP 与分级 ALGO 的图像质量控制

**问题**：四宫格/多路环视中不同区域光照、噪声、色彩差异大，全局 ISP 参数只能折中。

**做法**：

- `0x700–0x75C` 四 tile 独立 128 bit 配置；
- 每 tile 独立 gain R/G/B、signed offset、denoise、median、sharpen、tone、saturation；
- 字段按协议 lsb/width 封装，保留位写零；
- QUERY → 修改 → 完整 shadow → CLEAR → APPLY → STATUS → QUERY 回读；
- ALGO 的 DEHAZE bit56 作用于算法四宫格/PCIe 采集；GAUSSIAN/MEDIAN/SOBEL
  bit57–59 仅作用于 HDMI 后级。

**证据**：

- 协议 v1.0.1 JSON 的 123 个寄存器、ISP 17 项字段；
- 实板 ISP/ALGO no-op APPLY 返回 `status=0x00020002`，四 tile 回读一致；
- 四 tile 的 `gain_r=300 / offset_r=-7` 已在模拟测试中验证有符号和位域封装。

**比赛表述**：

> 将 ISP 从“全局一组参数”提升为“四区域独立参数 + 分级算法链”，
> 可针对前视/侧视/夜间区域分别调参，而 DEHAZE 与 HDMI 后级算法作用域分离。

**边界**：实板只验证了同值 APPLY/回读，尚未做画质主观/客观对比；不能写“画质提升 X%”。

---

### 创新点 3：面向安全显示的事务化 PCIe 控制协议

**问题**：普通 MMIO 写只能证明“发出”，不能证明 FPGA 已接收、已锁存、当前有效。

**做法**：

- HUD sequence 在 REQUEST[31:16]，SCENE 在 COMMIT[15:0]，NAV 在 CONTROL[31:16]，
  ISP/ALGO 在 COMMIT[31:16]，各域独立，避免混用；
- 同时读 ACCEPTED_SEQ、APPLIED_SEQ、EFFECTIVE_STATE、REJECT_COUNT、
  LAST_ERROR、LAST_ACCEPTED_REQUEST；
- SCENE 采用 shadow → TIME0 → TIME1 → COMMIT，501 ms 保持/过期；
- 行为 `action_active` 以 RESULT bit14 判断，不能只看序号；
- HSST 优先级：新鲜 HSST 整包 > 新鲜 RK 场景 > unknown；
- PCIe 没有油门/刹车/转向写 CSR，`0x650/0x654` 不作为动作发送地址。

**证据**：

- 实板 HUD `0x608=0x00011103`：accepted=1、applied=1、effective=0x103；
- 实板 SCENE seq42：`0x64c applied=42`；
- 实板 NAV：`NAV_ACTIVE=0x00780003`；
- 实板 ISP/ALGO APPLIED；
- 协议 golden vectors 与自动化测试 9 项通过。

**比赛表述**：

> 显示控制不是“写一个寄存器”，而是带序号、接受、锁存、有效态和粘滞错误的事务；
> 对 HUD、场景、导航、ISP/ALGO 使用不同序号域，支持失效/过期/拒绝诊断。

**边界**：HSST 优先级和 501 ms 过期由协议/RTL 定义；软件侧只做了读取/诊断，
未做故障注入证明。

---

### 创新点 4：四路多速率 AI 的加权轮询调度

**问题**：RK3566 单 NPU 同时服务前视/左/右/后四路，均分会让每路都达不到目标帧率。

**做法**：

- 前视 512×288@10 FPS，侧后 320×192@6 FPS；
- 队列 drop-oldest，深度按延迟目标选择；
- 用**目标帧率之比 10:6:6:6** 作为加权轮询权重，而不是凭直觉排优先级；
- 把“服务能力/流数”作为单路上限，不把请求值设到不可持续水平。

**证据（RK3566 文件回放实测）**：

```text
CONFIG-B-WRR-FAIR：27.91 / 28 FPS，达成率 0.997
FRONT 9.96、LEFT 5.99、RIGHT 5.99、BACK 5.98 FPS
FRONT e2e p95 154 ms（深度 1）
1200 s 长测：27.30 / 28 FPS，FRONT 9.85 FPS
SoC 最高约 38.9°C，NPU 600 MHz，无热降频
```

**比赛表述**：

> 用“目标帧率比例”推导四路加权轮询权重，把 RK3566 单 NPU 的四路聚合
> 从 23.86 FPS 提升到 27.91 FPS，前视达成率从 0.897 提升到接近 1.0。

**边界**：这是**文件回放 + 单板 AI**，不是 PCIe 四路实时输入；
不能把 27.91 说成真实四路 PCIe 吞吐。

---

### 创新点 5：软硬件可验证性与单一控制所有权

**问题**：调试台、Qt、脚本各自写 BAR1 会竞争；模拟成功容易掩盖真实掉线。

**做法**：

- RK3566 上一个 PCIe 控制 worker 独占 BAR1 控制窗口；
- 上层 Qt、浏览器、命令行共用同一服务；
- HARDWARE / DISCONNECTED / SIMULATION 明确区分，生产模式不自动降级为模拟；
- 123 个寄存器地址模型、白名单、完整 DWORD 写、事务日志；
- Web 控制台逐笔显示 offset、写入值、sequence、STATUS、accepted/applied、
  error/reject、epoch/source；
- DMA/PIO 未具备驱动所有权时明确阻断，不伪造成功。

**证据**：

- 实板 Web：`HARDWARE` 状态，HUD/SCENE/NAV/ISP/ALGO 操作可回读；
- 9 项协议测试通过；
- 单进程同时提供 UDP 38264 与 HTTP 8080；
- 服务在实板运行，旧 `pcie_isp_service.py` 已收敛为同一 worker 的启动器。

**比赛表述**：

> 把“能发命令”做成“可追溯的硬件事务”：单一 BAR1 所有权、硬件/模拟隔离、
> 原始寄存器回执和浏览器调试台，降低联调时误判和驱动竞争风险。

**边界**：Qt 原 FPGA 页面的 `control` 注入尚未验收；Web 是当前已验证入口。

## 二、不要写进比赛文档的内容

| 内容 | 原因 |
|---|---|
| “QSG 提升了帧率” | P3 短测无优势，已保留 FBO |
| “共享内存 60 FPS 是 PCIe 优化” | Ring/INPROC 是 LOCAL_REPLAY，PCIe 仍走 PGVF |
| “PCIe 四路实时 27.9 FPS” | 多路数据是文件回放，不是 PCIe 输入 |
| “DMA/视频 DMA 已完成” | 写控制 BLOCKED，只有只读状态 |
| “PIO/Ethernet 动作已接入” | 无驱动接口/无模块 |
| “P2 模型已上板” | B-05L 是 PC 评估，RKNN 未验证 |
| “所有 ISP 字段已画质验证” | 只做了同值 APPLY/回读 |
| “HUD 视觉已验收” | 只有寄存器 accepted/applied/effective，没有拍照比对 |

## 三、答辩 3 分钟主线

1. **为什么做异构**：FPGA 做确定性采集/ISP/OSD，RK3566 做 AI/交互；
2. **为什么协议复杂**：显示必须证明已锁存，不能只证明写过；
3. **为什么四区域 ISP**：不同区域光照/噪声不同，全局参数必然折中；
4. **为什么加权轮询**：四路共享单 NPU，权重应取目标帧率之比；
5. **怎么证明**：实板 HUD/SCENE/NAV/ISP/ALGO 回执 + 四路 27.9/28 文件回放 +
   9 项协议测试 + Web 原始事务日志。

## 四、证据索引

- `deployment/lubancat/qt_gpu_probe/rk3566_pcie_protocol_v1.json`
- `docs/pcie_debug_console_v101_20261025.md`
- `docs/competition_heterogeneous_multistream.md`
- `docs/competition_multistream_budget.md`
- `tests/test_pcie_protocol_service.py`
- 实板截图：`outputs/pcie_web_hardware.png`
