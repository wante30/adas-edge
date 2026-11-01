# 单路 / 四路集成验收记录

交付整理日期：2026-11-02（Asia/Shanghai）。开发会话标识：20261101。

## 结论与使用条件

本次交付基于指定下午稳定提交 `9b396e8b346bfbbfa9856950937e1cd3f54516f4`，独立恢复 Qt/AI 源码，复用 PR #5 的 BDD C API。单路与四路切换、整图一次推理、Q1 危险事件、自动刹车请求、自适应请求及实时 FPS 已在 RK3566 板端实际运行。

按用户后续指示，最终验收使用已有的 1350 帧 1280×720 RGB565 长录制，循环回放，源请求 60 FPS。**当前没有真实 PCIe，最终成绩是板端本地回放结果。** 不能将其描述为真实 PCIe 吞吐、LVT1 实车状态或实际制动成功。

## 与下午稳定版的对应关系

| 功能 | 交付行为与证据 |
|---|---|
| 道路模型 B / B-CAPI / MTL-FAST / MTL-FULL | 原路径、模型能力与菜单保留；四种模型均有实际完成结果 |
| 车道 / 可行驶区域 | 道路模式沿用原 MTL 结果和显示；切到四路时清除不适用分割 |
| 四路检测 | 1280×720 整图缩放为 512×288，一次 BDD RKNN 推理；同一结果在各象限绘制全局坐标框 |
| 危险事件 / 自动刹车 | 原 AutoBrakeController 和道路语义保留；只有完整位于 Q1 的目标进入危险事件链路；正常端口 38262 |
| 导航 OCR | 独立 Q2 OCR、2 Hz 请求保留；本回放导航可读性有限，见后文 |
| HUD / LVT1 | 原适配及显示保留；本次无硬件，显示未知状态 |
| Tesla BEV | 道路目标继续驱动 BEV；已截图 |
| AI FPS | 最近 1.2 秒唯一有效结果，200 ms 更新；离线归零，切换清空旧窗口 |

## 代码修改范围

| 文件 / 目录 | 修改目的 |
|---|---|
| `ai/inference_mode.py` | 明确模式、epoch、坐标空间和实例身份；整图后端与 Q1 安全过滤 |
| `ai/ai_bridge.py` | 模式握手、模型加载/释放、请求身份检查、最新帧处理、滚动统计 |
| `hmi/video_item.cpp`、`hmi/video_item.h` | 请求节流、唯一结果验证、切换与重连、离线检查、FPS 和日志 |
| `hmi/main.cpp`、`hmi/main.qml` | 默认自适应参数与现有视图切换事件接入 |
| `fast_mosaic/` | 选择性复用 PR #5；独立通用 C API，不覆盖道路库 |
| `tools/`、`tests/` | 独立启动/恢复、回放/验证、结果复算与回归测试 |

`ai/auto_brake.py`、`ai/competition_ai/`、`ai/q2_navigation_reader.py` 相对恢复点 `97ea22c7` 无修改。没有更改正式驱动、位流、RKNN runtime、原 `libp2capi.so` 或生产自启文件。

协议、坐标与跨象限处理见 [MODE_PROTOCOL.md](../MODE_PROTOCOL.md)。上游文件哈希、9 输出 / 3 尺度与 24016 字节 ABI 见 [UPSTREAM.md](../fast_mosaic/UPSTREAM.md)。

## 实测结果

稳定区间 **612.179 秒**，期间每分钟切换道路/四路；源和 AI 计数持续增长，窗口保持 IsViewable，无重复结果身份，无推理/监听错误或崩溃标记。此前已测试四个道路模型、连续 ROAD→MOSAIC→ROAD→MOSAIC→ROAD、停止 AI 后离线/0 FPS、同一 HMI 重连新 AI 实例。

| 项目 | 实测 |
|---|---:|
| 回放源请求 | 60 FPS |
| 回放源实际 | 54.603 FPS |
| 纹理上传 | 28.676 FPS |
| Qt scene graph | 29.444 FPS 均值 |
| ROAD_Q1 / MTL-FAST | 6.899 FPS，2109 个结果 |
| MOSAIC_FULL / FAST-MOSAIC | 14.534 FPS，4445 个结果 |
| 道路 pipeline / RKNN 平均时间 | 104.883 / 81.189 ms |
| 四路 pipeline / RKNN 平均时间 | 31.981 / 16.150 ms |
| CPU | 四核总容量的 23.856% 均值 |
| GPU | 驱动负载 36.239% 均值 |
| NPU | 驱动报告 100%；未独立验证利用率 |

B、B-CAPI、MTL-FULL 的短切换检查分别为 32 / 29 / 12 个结果，9.285 / 13.739 / 5.124 FPS；这些不能代替各自的十分钟稳定性测试。四路数值是当前 Qt 端到端结果，不能用上游独立推理的 25.61 FPS 替代。

自适应设置为 `--ai-hz 0`，最多一个在途请求，默认上限 30 Hz。根据完成反馈发送最新帧，1.5 秒请求超时；GUI 更新超过 12 ms 或渲染低于 25 FPS 时增加 30 ms 退让。保留 `--ai-hz 10` 兼容模式与 `ADAS_AI_MAX_HZ` 配置。

## 危险事件、刹车和导航的验证边界

稳定区间记录 **384 帧带危险事件的结果**。四路截图实际显示 Q1 货车的“前车距离过近”，其他象限的目标仍正常显示。

下载的回放日志含 **125 次正常端口刹车请求、125 次 timeout**。当前 PCIe/HUD 硬件桥接已不在，不能确认物理制动。阶段二真实 PCIe 测试曾收到 `ok:true`、`last_error:0`，该历史结果与最终无硬件回放分开保留。

导航结果为 VALID 83、STALE 4、UNREADABLE 6468。当前录制 Q2 导航卡片的尺寸/位置与识别器预期存在差异，主要失败原因是 `Q2_ARROW_NOT_FOUND`。导航代码未删除、OCR 仍请求 2 Hz，但不能声称本回放全程导航正常。

PCIe 断开后的遗留硬件状态通过重启清除；本次未实现驱动热恢复。无真实 PCIe 时无法完成断线重连和 LVT1 硬件验收。截图保留了原界面的部分 PCIe 标签/online 字段，这些不作为连接证据。

## 证据与复算

- [最终指标 JSON](final_summary.json)：按单调时钟计算，区分模式与实例；包含原始日志 SHA-256。
- [完整验收事件](stage5_acceptance.json)：模型/模式切换、离线、重连、十分钟连续采样及窗口状态。
- [原始日志快照 ZIP](stage5_logs.zip)：AI / HMI JSONL、文本日志、验收报告及 runner 输出。
- [四路危险事件截图](final_mosaic.png)、[道路 MTL](stage5_model_MTL-FAST.png)、[BEV](stage5_bev.png)、[离线](stage5_offline.png)。
- [运行产物 manifest](runtime_manifest.json)：HMI / 独立 C API / 外部模型 SHA-256、基础提交、板型与验证 boot ID。

截图为实际板端画面；原始文本日志是在应用继续运行时下载的快照。`final_summary.json` 中的哈希对应这些快照。性能计算使用最终稳定区间；runner 事件中的短尾窗口 FPS 仅用于诊断，不作为最终基准。10 项 Python 回归测试通过；Qt 和独立 C API 已在 RK3566 实际编译运行。

## 启动与恢复

运行包包含 AArch64 `build-stage4/qt_gpu_probe`、独立 `fast_mosaic/capi/libp2capi_generic.so`、源码/资源和脚本；不包含板端模型、Qt/Mali/RKNN/OCR 依赖或原始长录制。解压到独立目录后：

```sh
sh tools/start_local_replay.sh /home/cat/qt_gpu_probe_ai_20261019/local_quad_full.rgb
sh tools/restore_afternoon.sh
```

需要重新编译时运行 `sh tools/build.sh`；恢复硬件后可运行 `sh tools/start_pcie.sh`，实际硬件重验尚待进行。首次使用必须能捕获原 HMI 的启动/恢复状态；若无原 HMI，本地回放允许复用该板此前已恢复会话的参数记录。运行包不携带旧 PID 状态。原正式自启恢复后在没有 PCIe 的情况下仍会等待输入。
