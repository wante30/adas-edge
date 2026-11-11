# ARM → FPGA 多交通灯调试协议草案 TLS1

## 对接边界

TLS1 是独立的离线调试草案；本次没有分配 BAR 地址、修改控制寄存器、修改自动制动状态机或赋予新的硬件许可。旧 HUD 的 RED/GREEN/OFF 和现有 PCIe 通路继续使用原协议。新方向数据只存在于 ARM JSON/MVA2 status 扩展和离线 TLS1 包。FPGA 同学应先完成接收、CRC、超时与离线回放，控制接入另行评审。

ARM 的 `competition_ai/signal_protocol.py` 提供编码/解码，无任何设备打开或写入操作。`signal_fpga_backend.py` 是未来 ROI 分类器结果的适配接口，同样没有硬件访问。

## ARM 逻辑结果

`multi_signal` 位于 MVA2 JSON status 中，`schema_version=1`，`debug_only=true`，`source_id=Q4`。所有 bbox 均为原始 Q4 局部像素，实机 640×360；绝不采用画面左右位置作为箭头证据。

| 字段 | 语义 |
|---|---|
| context | AI 实例 bridge_id、模式 epoch、ROAD/MOSAIC、模型 ID、物理象限；任一变化使旧结果失效 |
| frame_id / timestamp_ns | 本次属性观测对应的真实源帧与 CLOCK_MONOTONIC 捕获时间 |
| published_at_ns | 发出时间，仅用于延迟诊断，不能刷新 timestamp_ns |
| signals[] | 最多 8 盏灯；track_id、bbox、检测/颜色/形状置信度、颜色、箭头、来源模型、证据、有效期、错误原因 |
| direction | 灯自身视觉箭头，可为 left/straight/right/uturn；圆形与未知为 null |
| ego_relevant | YES 或 UNKNOWN；视觉箭头本身不能证明本车道路归属 |
| association | ASSOCIATED 或 UNASSOCIATED；独立记录 head_id、group_id、校准证据和其有效期 |
| observed_directions | 画面内可见箭头的描述性汇总；可能属于其他道路，不能替代本车方向信号 |
| parent_head_bbox | 原始检测灯头框；独立发光面 bbox 可来自同一灯头 |
| directions.left/straight/right | color、valid、confidence、signal_ids、groups、timestamp_ns、expires_at_ns、reason |
| rejections | 错帧、过多候选、身份匹配歧义等原因 |

颜色枚举 UNKNOWN/RED/YELLOW/GREEN；形状 UNKNOWN/ROUND/LEFT_ARROW/STRAIGHT_ARROW/RIGHT_ARROW/UTURN_ARROW。置信度为算法分数，尚未经过真实比赛数据概率校准，不能解释为通行成功概率。

## TLS1 二进制布局

所有整数小端，无浮点，包长为 `40 + 3×8 + N×24`，N≤8，最大 256 字节。source 固定物理 Q4，坐标固定 640×360；其他分辨率需要新协议版本。版本 1 的 flags 必须等于 1，表示 DEBUG_ONLY。

### 40 字节头

| 偏移 | 类型 | 字段 |
|---:|---|---|
| 0 | char[4] | ASCII TLS1 |
| 4 | u8 | version=1 |
| 5 | u8 | flags=1，调试数据，禁止作为动作许可 |
| 6 | u16 | header_bytes=40 |
| 8 | u32 | sequence，递增序列 |
| 12 | u32 | session_epoch，发送方为实例/模式/模型切换分配的新 epoch |
| 16 | u64 | source frame_id |
| 24 | u64 | ARM 发送时 CLOCK_MONOTONIC ns，仅诊断，不假定 FPGA 同步 |
| 32 | u16 | payload_bytes=24+N×24 |
| 34 | u16 | lamp_count=N |
| 36 | u32 | CRC32/IEEE，Python zlib.crc32；计算整包时此字段置零 |

### 三条 8 字节方向记录

固定顺序 LEFT、STRAIGHT、RIGHT，每条 `<BBBBHH>`。

| 偏移 | 类型 | 字段 |
|---:|---|---|
| 0 | u8 | direction_id：1/2/3 |
| 1 | u8 | bit0 有效；bit1 可信关联信号冲突；其余零 |
| 2 | u8 | color：0 UNKNOWN、1 RED、2 YELLOW、3 GREEN |
| 3 | u8 | confidence，0–255 |
| 4 | u16 | capture_age_ms，饱和到 65535 |
| 6 | u16 | remaining_ms，发包时剩余有效时间，最大 1200 |

有效位清零必须同时输出 UNKNOWN、remaining=0。冲突会清除有效位，并在 ARM JSON 保留参与 ID、颜色及组名。圆形灯只有在校准证据明确给出控制方向时才能进入方向记录。

### N 条 24 字节灯记录

每条 `<IBBBB4HBBHHH>`。

| 偏移 | 类型 | 字段 |
|---:|---|---|
| 0 | u32 | track_id，在同一 AI 实例内不复用 |
| 4 | u8 | color，同上 |
| 5 | u8 | shape：0 UNKNOWN、1 ROUND、2 LEFT、3 STRAIGHT、4 RIGHT、5 UTURN |
| 6 | u8 | bit0 观测有效；bit1 本车道路关联有证据；其余零 |
| 7 | u8 | detector_confidence，0–255 |
| 8 | u16[4] | Q4 xyxy，x∈[0,640]，y∈[0,360] |
| 16 | u8 | color_confidence，0–255 |
| 17 | u8 | shape_confidence，0–255 |
| 18 | u16 | capture_age_ms |
| 20 | u16 | remaining_ms |
| 22 | u16 | reserved=0 |

## FPGA 接收与超时建议

先检查 magic、版本、DEBUG_ONLY、长度、N、CRC、枚举和坐标，再原子发布整个包，禁止半包更新方向。非法包不刷新有效期；同 epoch 内重复或倒退 sequence 也不刷新。模型/相机/模式切换应清空旧数据，重新建立会话 epoch。ARM 重启后的序列重置必须经过新的会话握手，不能仅把更小 epoch 当成新包接受。

FPGA 与 ARM 时钟未同步，因此 FPGA 以本地接收时间加 remaining_ms 形成截止时间，并扣除已知最大传输延迟预算。1200 ms 是灯观测上限；方向有效期还受 500 ms 的道路关联证据有效期限制。过期、链路断流、AI 停止、错误包持续到超时、冲突时方向必须为 UNKNOWN，不能保持最后一个 GREEN。接收层保存原因计数用于比赛排障。重复包不得续期。

本版本所有包均为调试数据；即使 color=GREEN、valid=1，也不代表转向或加速许可。

## FPGA ROI 后端接口

调用 `FPGAAttributeBackend(lookup)`，lookup 接收 frame_id、timestamp_ns、bbox、context、source_id，返回相同身份字段及 color、shape、color_confidence、shape_confidence。身份不完全相同、置信度非有限数或枚举非法时输出 UNKNOWN。FPGA 的分类器结果不能自行填充 ego_relevant；关联层仍需独立道路、视角和信号头证据。

`--signal-bindings` 可读取独立进程原子更新的 JSON 数组，最多 16 条、32 KiB，每条有效期≤500 ms。必须包含 context、source_id=Q4、coordinate_space=q4_native、head_id、group_id、calibration_id、bbox、directions、confidence≥0.8、timestamp_ns、expires_at_ns、ego_road_confirmed=true、viewpoint_confirmed=true，evidence_type 为 CALIBRATED_VISUAL_HEAD 或 FPGA_CALIBRATED_HEAD。当前 FPGA 车道线类型字段不足以构成此证据，所以没有从现有硬件位自动生成绑定。
