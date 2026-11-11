# 多交通灯实现与验收记录

## 基线

2026-11-11 通过 SSH 核对实机进程、启动脚本、源码 SHA-256 与 Qt 二进制。实际 AI 为 `/home/cat/adas_issue9_20261102/src/ai`，Qt 为 `/home/cat/adas_issue10_lifecycle_20261103`；不是直接使用 GitHub main 或 afternoon 的历史快照。逐文件哈希见 `board_baseline_20261111.json`。本分支基线标签 `rollback/multi-signals-board-20261111`。

## 审查结论

- `competition_ai/traffic_light.py`：单 ROI HSV 可复用；`classify_traffic_lights` 是旧全局聚合，不能作为新方向输出。
- `ai/tools/competition_console/fpga_perception.py`：先按 IoU 去重，再汇总 RED/GREEN/OFF；黄色不受旧 HUD 协议支持。保留为兼容路径，不接入新信号控制。
- `ai/model_profiles.py`：实际有 FAST-Q4、FAST-MOSAIC、MTL-FAST 和含 Specialist 的模型。新模块不加载检测模型。
- `FrontPipeline`：主框 750 ms、Specialist 框 400 ms 缓存；新属性路径只接收本帧真正更新的检测，禁止把缓存框当成本帧观测。
- `MosaicRoadAdapter`：全 mosaic 框转换为指定前向相机局部坐标；新增路径固定物理 Q4，其他前向选择失效。跨接缝框另外验证。
- 现有 `AsyncTrafficLightTracker` 会缓存最多 60 张全帧及灰度图，不适合当前 CPU/内存预算；新路径使用小规模 IoU 匹配、单待处理任务和 2 Hz 属性分析，不额外做全帧光流。
- Qt 通过 MVA2 的 JSON status 扩展调试信息。正式 HUD、PCIe 寄存器、硬件许可和自动刹车不读取 multi_signal。

## 第一阶段验证

本地独立模块 7 项测试通过：多色独立、双模型去重、缓存框拒绝、过期、上下文切换、坐标转换、闪烁稳定。合成图仅用于这些单元测试。

隔离运行部署测试需设置 `ADAS_EDGE_ROOT` 和 `PYTHONPATH` 为 `deployment/lubancat/afternoon_mosaic/ai`，不能先载入仓库根的同名旧包。部署现有 62 项测试中 61 项通过；`test_auto_brake.py::ControlEligibilityTest::test_hardware_q4_rejects_nonphysical_and_wrong_context` 的旧 fixture 未设置实机版新增 physical_input 条件，返回 nonphysical_hardware_input。该控制模块和测试均未因本专项调整。

## 当前交付状态

ARM 多灯调试路径已可运行，默认关闭。每个真实 Q4 ROI 独立输出颜色、形状、ID、置信度、观测时间和期限；最新任务覆盖，禁止积压。主/Specialist 先按检测头去重，再把明确分离的发光面拆成独立灯，保留 parent_head_bbox。同色但不相接的灯面也可拆分；相邻 LED 碎片保持同一符号，无法区分时 UNKNOWN。

本阶段使用既有 HSV、发光轮廓、开闭运算、实心/空心/断开箭杆模板、轮廓凹陷和分数间隔。大 ROI 只缩小、不放大小 ROI。亮芯恢复要求附近存在已测量的彩色像素。算法不使用检测框的画面左右位置推断箭头。模板分数和 HSV 比例不是校准概率。

`observed_directions` 仅描述画面上可见箭头；`directions` 必须另有本车道路/视角/信号头绑定证据。现有 FPGA 车道线类型不够证明某个灯头的道路归属，因此当前没有自动生成此证据，默认本车方向 UNKNOWN。外部绑定和可替换 FPGA ROI 后端契约已实现。导航意图不赋予视觉证据。

Qt 已在 RK3566 编译并打开调试面板，Tesla 页面运行截图见 `evidence/qt_signal_debug.png`。显示 VISIBLE、EGO、每盏灯属性、年龄和原因。视频模式提供 Q4/四宫格坐标框；断流、AI 离线、模型切换、超时均主动变 UNKNOWN。正式硬件控制不消费新结果。

## 验证结果与限制

截至本次提交，新增 41 项测试通过，其中 10 项使用真实照片裁剪。另有部署既有 61 项通过，1 项已知旧 fixture 失败按完整节点 deselect。RK3566 Python 3.8.10、OpenCV 4.10.0 上 9 个固定真实裁剪回归全部通过，详见 `evidence/arm_photo_regressions.json`。这些固定回归含应拒判样本，不是 100% 识别准确率。

真实照片来自 Wikimedia Commons，逐图 URL、下载字节 SHA-256、作者、许可保存在 `photo_sources.json`；人工原生坐标标签在 `photo_labels.json`。15 张照片、33 个 ROI 包括红/绿/黄、左右直行、掉头、圆灯、暗灯与模糊样本。03 为真实照片在不同时刻的拼图，不能当作红/绿掉头同时亮。22 是示意图，明确排除；18 视角倾斜，排除方向统计。仓库中的真实裁剪附有各自来源许可，派生结果延续原图许可，见 `tests/fixtures/real_signals/ATTRIBUTION.md`。

| 样本组 | 颜色正确/已知真值 | 箭头正确/箭头真值 | 说明 |
|---|---:|---:|---|
| development | 12/13 | 5/8 | 开发时已查看，含阈值与模板决策依据 |
| exploratory | 4/10 | 1/4 | 也已查看，不作为独立测试集 |
| regression_after_holdout_review | 8/9 | 1/4 | 首轮留出后用于修复，现为回归集 |

首轮留出测试发现 photo20 的红圆灯被误判掉头，已补充凹陷深度否决；修复前的冻结报告保存在 `evidence/photo_v2_frozen_report.json`，修复后不得再宣称该组是独立留出准确率。当前完整诊断报告为 `evidence/photo_report.json`。照片由手工 ROI 输入，因此检测召回率和本车方向关联准确率均为 null；不可混称端到端效果。

两段真实比赛原视频共 498 帧，生成每帧 JSONL、对照视频和失败裁剪。只在各 3 帧人工检查，每段 6 个灯头标注；检测使用既有 PC 审计缓存 COCO 主模型+专项模型，不代表实机 B05N 检测器。红片原始检测召回 5/6、有效属性轨颜色 2/5、箭头 0/5；绿片检测召回 1/6，没有匹配的有效属性轨，属性准确率为 null。已纠正用历史调试框冒充检测召回的问题。详见 `evidence/red_video_report.json`、`green_video_report.json`。

红片源图可见红左箭头和红直箭头；降到原生 Q4 后有效发光约 8×9 像素，低于形状门槛，必须 UNKNOWN。不能通过锐化或放大补造细节。该视频不是左绿直红混合场景，当前没有用它声称满足最终验收。

| 要求场景 | 已运行证据 | 尚缺 |
|---|---|---|
| 红、绿、黄 | 真实照片及两段比赛视频；黄颜色回归 | 夜间/比赛黄色视频 |
| 左绿直红 / 左红直绿 | 合成单元测试；真实照片分别验证左右直箭头 | 同一路口实拍混合箭头的端到端标注 |
| 三方向同时 | 三方向关联单元测试 | 真实三方向同时视频 |
| 圆灯不猜道路 | 真实 ROI、UNKNOWN 与绑定单元测试 | 自动道路关联视觉模块 |
| 小灯、模糊、漏检 | 原生 Q4/实拍失败样本、短漏检与 TTL 测试 | 遮挡全过程人工真值 |
| 双模型重复 | 去重/灯头包含/实拍拆分回归 | 更大规模实机复杂灯组 |
| Q4、mosaic、切换 | 坐标接缝、模型 epoch、既有模式测试 | 全套真实比赛切换回放验收 |
| 同方向重复与冲突 | 显式绑定及可见箭头冲突单元测试 | 真实冲突视频 |
| 断流、AI 停止、过期 | worker/跟踪/Qt 主动失效；隔离回放恢复记录 | 拔线台架验收 |

## 性能记录

用户随后明确先完成功能、暂不考虑性能，因此停止性能优化。早期算法做过相同真实 Q4 视频 30 秒 A/B，其余三摄像头空白、OCR 关闭、DRY_RUN，不是约 95% CPU 的真实整机基准：

| 方案 | AI FPS A→B | Qt FPS A→B | CPU A→B | B 属性均时 |
|---|---|---|---|---|
| FAST-Q4 | 17.25→16.27（-5.68%） | 28.06→28.76 | 83.48→83.39% | 4.46 ms |
| MTL-FAST | 6.76→6.69（-1.03%） | 27.82→27.70 | 77.83→77.92% | 1.20 ms |

结果见 `evidence/board_ab_summary.json`。早期 FAST-Q4 超过原 5% 门槛，不能合入正式部署。新箭头模板与拆分版本未重做整机 A/B；9 个清晰原生照片 ROI 在 RK3566 运行约 162–298 ms/个，已用位打包减少模板比较开销，但不把此微测试当作整机 FPS。结果年龄统计包含已失效的历史调试行，不能拿它声称有效信号延迟。

## 复现

```powershell
python -m pytest tests/test_multi_signal.py tests/test_signal_shape_association.py tests/test_signal_worker_backend.py tests/test_signal_protocol.py tests/test_real_signal_photos.py -q
python tools/check_signal_backend.py --output outputs/native-regression.json
python tools/collect_signal_photos.py --output outputs/photos --manifest docs/multi_signal/photo_sources.json --ids 00 01 02 03 04 08 09 10 11 12 13 14 18 19 20 21 22 23
python tools/evaluate_signal_photos.py --images outputs/photos --output outputs/photo-check
```

照片下载可能被源站 429 限流，脚本保留缓存并校验字节；固定真实裁剪回归无需网络。真实比赛视频不随源码重新分发，需使用报告指定 SHA-256 的原文件和原 PC 检测缓存。`tools/replay_multi_signals.py --help` 给出输入、标签、输出参数。部署、打包与回退见 `DEPLOY_ROLLBACK.md`；FPGA 字段和 CRC/超时语义见 `FPGA_HANDOFF.md`。

## 箭头传统算法失败后的备选

不继续无限调整阈值，也未增加第二个神经网络。失败样本已有原生 ROI、原因、HSV 像素与模板分数：photo01/04 的绿色箭头亮芯分裂，photo23 的直箭头经模糊后模板间隔不足，比赛红片发光像素太少；photo20 是形态误判反例，已明确拒判。

后续若授权训练轻量 ROI 分类器，可使用同一 AttributeBackend 契约，输入真实检测 ROI，输出颜色与形状以及质量/UNKNOWN，独立测试后替换 ARM 后端或经 FPGAAttributeBackend 注入。候选可为 48/64 像素的小型量化分类器，不能作为已有成果或当前准确率；先采集不同路口/距离/相机的带真值样本，以路口和视频分组划分训练与留出，重点增加圆灯、过曝、计时数字、行人灯和无法辨方向的负例。8×9 像素信息不足不能保证靠分类器解决，应先评估光学或采样分辨率。
