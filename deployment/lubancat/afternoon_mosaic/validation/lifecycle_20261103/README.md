# RK3566 Tesla BEV 轨迹生命周期修复

日期为 Windows 主机北京时间 2026-11-03。板端系统时钟不准确，运行验证使用启动 ID 和单调时钟。

## 改动与原因

对照用户 Windows 项目 `.deps/Tesla-BevCam-Clone/main.py` 的 `KalmanTrack` / `BEVByteTracker`：Windows 在 BEV 空间维护位置和速度，预测短暂漏检，并做两阶段关联。原板端 AI 的 ByteTracker 会保留失联轨迹，但只发布实际测量；原 BEV 显示立即建立单次观测对象、静态保留，依赖原服务器 ID，并把淡出同时用于模型高度。因而上游 ID 断开时会重新生成模型，退出时还会缩入路面。

本次在显示层新增 `BevLife::Tracker`：

- 两次不同时间的真实观测后确认三维对象；单次观测仍走原二维检测和危险控制链路。
- 保留上游 `server_track_id`，另设显示 ID。优先匹配相同上游 ID，再按类别、预测框位置、框比例和 BEV 位置做高/低置信度两阶段全局匹配；歧义关联放弃。
- 车辆与行人不互相继承身份。关联为启发式，并不等于经人工真值验证的物理身份正确率。
- 常速度 Kalman 状态、三点中值、创新限制和渲染限速；短时预测最多 450ms。确认轨迹最多保留 1.4s，550ms 后透明淡出。
- 淡出保持模型实际尺寸。重新检出时，从当前透明度平滑恢复，避免再次突然变实。
- 来源、模型或跟踪 epoch 变更时清空旧上下文，离线时不伪装为当前检测。
- 保留道路示意动画。位置仍为相对显示，未引入标定深度、真实车速或完整 Windows 深度网络。

未修改 AI 模型、推理精度、危险判定或 FPGA 控制代码。部署只替换 HMI，Issue9 AI/硬件桥继续原进程运行。

## 同源实际 Qt 回放

输入是已有真实 RKNN 测量记录，343 批、70.707 秒，含车辆和行人；两版读取同一份 `measurements.jsonl` 和同一个板端原始视频文件。不是重新跑两次模型后比较不同的检测结果。基线播放器来自此前 Issue10 版本；候选使用本次渲染器。输入和二进制 SHA256 见 `replay_receipt.json`、`live/provenance.json`。

`review/before_recorded_rknn.mp4` 与 `review/after_recorded_rknn.mp4` 是 RK3566 X11 实际录屏。播放器不接控制端口，窗口明确标注 RECORDED RKNN / LOCAL REPLAY。两次录屏启动有少量时差，因此视频相同播放秒数不代表严格相同测量时刻；定量比较使用轨迹的单调时间戳。`review/person_vehicle_sequence.png` 仅是新版实际录屏的连续取帧，未添加检测结果。

| 记录指标 | 原 Issue10 | 新版 |
|---|---:|---:|
| 出现过的显示轨迹 ID 数 | 239 | 154 |
| 上游 ID 改变后的显示轨迹接续 | 0 | 25 |
| 漏检恢复时过快的淡入变化 | 8 | 0 |
| 等效 60Hz 位移 P99（相对单位） | 0.2308 | 0.2016 |
| 等效 60Hz 最大位移（相对单位） | 0.3284 | 0.2108 |
| 非正上游 ID 的绘制记录 | 0 | 0 |

新版实际显示轨迹包括 118 条 car、8 条 truck、28 条 person；这些是分上下文统计的显示轨迹，不能当作不同真实物体的数量。显示 ID 减少同时来自确认延迟和重关联，不能当作检测准确率提高。两次确认会增加约一个推理周期的三维显示延迟，另有约 200ms 的淡入；二维告警未增加该门槛。

`comparison.json` 保存完整指标与计算口径。速度为相对单位；仅比较同一轨迹的有效连续样本，位移归一化到 60Hz，避免直接拿不同渲染帧率下的单帧位移比较。预测使目标在测量间继续运动，不能把所有位移变小当作效果更好。录屏时的场景 FPS 是播放器指标，不是完整车机吞吐率。

第一轮候选回放发现重新检出后透明度直接恢复的问题，已修复并重新完整播放；只有 `review/after_recorded_rknn.mp4`、最终回执和本报告中的新版指标作为最终结果。重启前中断的数据、第一轮试验留在本地排查目录，不计入验收。

## 构建与测试

RK3566 原生 qmake / g++ 构建成功，正式 HMI 和回放播放器链接同一份最终 `tesla_scene_item.o`。三项原生测试退出码均为 0，详见 `unit_tests.json`：

- `bev_lifecycle_test`：轨迹确认、重复测量、短时漏检后新 ID 接续、车辆/行人分离、一对一关联、上下文清空、单帧轨迹退出、全局匹配、Kalman 速度、重新出现的透明度连续性、过期 ID 重用、无效数字过滤。
- `bev_display_state_test`：保留的原显示辅助类回归。
- `road_motion_test`：等时位移、断流暂停、相位和限步回归。

解压 `raw_evidence.zip` 可取得两版 trace、scene 日志和原始测量。使用 `tools/summarize_lifecycle_evidence.py` 复算；参数 `--before` / `--after` 指向解压目录即可。录制脚本 `tools/run_lifecycle_ab.py` 支持明确输入、两版二进制和独立输出目录，并在结束或中断后恢复现场 HMI。

## 真实 PCIe 部署与验证

正式候选部署目录：`/home/cat/adas_issue10_lifecycle_20261103/`。

二进制 SHA256：`cb8a7f415314d5b72c6b1de1b0c8714ef0065d46b6e941b8ef08830e1d50a948`。

实际 HMI PID 71096；AI 继续 PID 5568。`live/acceptance.json` 记录约 112 秒时的验收：555 个 HMI 样本，启动 5 秒之后 AI 离线样本为 0，视频帧号 88207→94933，CRC 错误 0，OpenGL 错误 0。实时为 Q4 + FAST-Q4 空间视图，道路继续移动。真实 FPGA 桥返回 `ADAS_REAL_PCIE_V1`、`ok=true`、`error=0`、`reject_count=0`。

`live/live_pcie.mp4` 是部署后 14 秒实际 PCIe 录屏，和前述历史回放明确分开。这次是显示层验证和只读控制状态检查，不宣称重新完成制动执行器验收。

## 回退和运行包

原 Issue10 程序和上一版 road_v2 均原路径保留，哈希见 `live/provenance.json`。部署目录保存 `previous_hmi.json` / `previous_session.json`；同次启动内可执行：

```sh
python3 /home/cat/adas_issue10_lifecycle_20261103/restore_lifecycle_hmi.py --deployment /home/cat/adas_issue10_lifecycle_20261103
```

该脚本仅恢复 HMI，不重启 AI 或重配硬件桥，并检查启动 ID 与当前进程归属。脚本经过语法检查；本次没有为了测试回退而再次打断已运行的新版本。

Windows 本地运行包为 `delivery/rk3566_lifecycle_20261103.tar.gz`，包含 ARM64 二进制、QML/素材、显示层源码、部署回执和回退脚本。依赖板端已安装的 Qt/Mali 与 Issue9 AI/硬件运行环境，不是可在空系统直接安装的完整固件。未改写原开机自启动；再次重启后需要按保存的部署目录恢复新 HMI，不能把旧启动项当作新版本。
