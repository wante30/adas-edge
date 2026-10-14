# 竞赛原生测试台：本机第一阶段

日期：2026-10-14。工程：`C:\Users\JingyuWang\adas-edge`。

本阶段实现比赛场景决策、原生按键交互，以及官方前视视频的路口灯控对照。显示直接复用当前 Tesla-BevCam-Clone 工作副本的 `render_tesla_ui()`，窗口使用 OpenCV，前视视频和比赛动作作为主要内容。

## 1. 启动与操作

双击：[run_competition_console.cmd](C:/Users/JingyuWang/adas-edge/run_competition_console.cmd)。默认打开官方 `video_20260929_113539.mp4` 的 12 秒位置并暂停；按 G 可以检查红灯下强制行驶报警，按 S 可清除闯红灯报警，空格继续播放。

原生窗口内的按钮与键盘共享同一个控制器：

| 输入 | 功能 |
|---|---|
| 1 / 2 / 3 | 模拟车辆在左转 / 直行 / 右转车道 |
| 0 / S / G | 自动模拟 / 停止 / 强制行驶 |
| 空格 / . | 暂停继续 / 单帧前进 |
| R / O | 重播并清理旧绑定 / 打开其他本地视频 |
| B | 在前视画面拖框，绑定当前所选车道的完整横向红/黄/绿灯组 |
| I / D | 路口 / 道路规则模式；已绑定的控制灯在切换场景后仍然生效 |
| Q / E | 请求左 / 右变道；当前视频链路车道类型未知，不能授予变道 |
| Esc | 退出并保存本次交互日志 |

按键使用英文输入法，无需按 Shift；主要操作也可点击按钮。加载另一段视频会清除原素材的灯组配置。选择 B 后，只接受画面内有效的灯组框；绿色需要不同源帧的连续确认。

自定义视频可用：

```powershell
Set-Location 'C:\Users\JingyuWang\adas-edge'
.\.venv\Scripts\python.exe tools/run_competition_console.py --video 'D:\your_front_video.mp4' --paused
```

默认权重是已有冻结 FP32 ONNX：主模型 `512×288`，灯专项 `512×192`。可分别用 `--main-model`、`--specialist-model` 指定同一输入/输出合同的其他权重；没有自动下载或重新训练。推理用 PC CPU 的 ONNX Runtime。

## 2. 灯色与车道关联的真实来源

程序同时运行主模型和红绿灯专项模型，其检测候选、耗时和源帧分别记录。绑定灯组还使用独立的像素颜色后处理：在明确标定的横向 R/Y/G 三灯位内统计亮色连通区域，并跟踪整个灯壳。

这个补充分支解决了测试中“红灯周围绿叶在整框颜色统计中占优势”的问题。灯壳外观匹配不足、移出画面或源观察过期时返回未知，不从其他候选中随意选绿灯。此方法目前只验证了上述横向灯壳布局。

示例配置 [competition_native_official.json](C:/Users/JingyuWang/adas-edge/configs/competition_native_official.json) 校验视频 SHA256，并在源帧 360 / 1800 初始化两个人工核对的灯组框。

**配置明确假设：三条模拟车道共享所选圆灯组。它不证明模型已自动确认真实车道的控制灯，也不证明具备多方向箭头灯识别。** 配置只提供位置和关联，没有红/绿颜色答案。窗口、配置和日志都保留该来源说明。

“模拟车辆正在行驶”与“系统建议停车”分别显示。视频里的真实车辆轨迹由原素材决定，按键不代表已经从视频估计了自车位置、航向或速度。

## 3. 新决策与旧接口的关系

[scenario.py](C:/Users/JingyuWang/adas-edge/competition_ai/scenario.py) 定义场景、模拟车辆状态、路口转弯和道路变道等独立类型。[scenario_policy.py](C:/Users/JingyuWang/adas-edge/competition_ai/scenario_policy.py) 根据这些输入给出建议动作、许可、报警及 PC LED 位。

- 路口：只有所选灯组关联明确、观察新鲜且颜色允许时授予通行；红灯强制行驶单独报警。
- 道路：没有红绿灯时仍可直行；请求变道必须有对应边界的有效虚线结果。实线禁止变道并报警。
- 路口转弯与跨线变道分别判断，不将“实线禁止变道”用于阻止路口转弯。
- 导航方向作为独立提示，不自动触发横向动作；新鲜危险输入具有最高优先级。

已有 `SceneInput` / `CompetitionAdas`、HUD v1 和 ADST v1 的字节格式、语义与代码保持原状。本阶段使用新的内部类型及 JSON 事件日志，没有把新动作强行编码进旧报文。实际 PCIe 消费端仍需按后续明确合同接入。

## 4. 本机验证结果

| 验证 | 结果 | 证据范围 |
|---|---|---|
| 新场景、绑定灯组、控制器与原 SDK 回归 | 78 项通过 | Python 本机代码 |
| 原生窗口事件 | 强制行驶报警、停止清报警、切车道、逐帧、重播清理、Esc、大小写快捷键均通过；切场景不能绕过已绑定红灯 | Windows 子进程窗口，脚本发送原生按键消息 |
| 红灯片段，源帧 360–779 | 420 帧 RED / STOP；强制行驶的 90 帧报警 | 真实视频像素＋明确灯组绑定＋脚本模拟车辆输入 |
| 绿灯片段，源帧 1800–1979 | 首 3 帧确认中，其后 177 帧 GREEN；左/右转各 30 帧，直行 87 帧，停车 33 帧 | 同上；模拟方向由按键选择 |
| 演示录像 | H.264、1440×900、30 FPS、600 帧、20 秒，完整解码检查通过 | 两段本机演示拼接，无原音频 |

这些是带明确场景配置的功能对照，不是全视频、多灯组、未知现场素材或 RK3566 INT8 的识别准确率。主模型与专项模型仍可能漏检；绑定像素分支与模型候选分别记录，不能把补充观察算作模型检测命中。

证据目录：`C:\Users\JingyuWang\adas-edge\outputs\competition_native_20261014`。

- [演示录像](C:/Users/JingyuWang/adas-edge/outputs/competition_native_20261014/official_intersection_native_demo.mp4)
- [原生窗口实际捕获](C:/Users/JingyuWang/adas-edge/outputs/competition_native_20261014/gui_release/native_client.png)
- [红灯强制行驶示例](C:/Users/JingyuWang/adas-edge/outputs/competition_native_20261014/red_release/frame_00423.png)
- [绿灯左转示例](C:/Users/JingyuWang/adas-edge/outputs/competition_native_20261014/green_release/frame_01833.png)
- [原生交互检查](C:/Users/JingyuWang/adas-edge/outputs/competition_native_20261014/gui_release/gui_check.json)
- [汇总结果](C:/Users/JingyuWang/adas-edge/outputs/competition_native_20261014/validation_summary.json)

## 5. 复现

```powershell
Set-Location 'C:\Users\JingyuWang\adas-edge'
.\.venv\Scripts\python.exe -m pytest -q tests/test_competition_scenario_policy.py tests/test_bound_signal.py tests/test_competition_console_core.py tests/test_hud_v1_encoding.py tests/test_hud_transactions.py tests/test_adas_status_v1.py tests/test_hud_pc_tools.py tests/test_competition_scene_protocol.py
.\.venv\Scripts\python.exe tools/run_competition_console.py --start 12 --end 26 --headless --record --scripted-controls --profile configs/competition_native_official.json --output outputs/new_red_check
.\.venv\Scripts\python.exe tools/run_competition_console.py --start 60 --end 66 --headless --record --scripted-controls --profile configs/competition_native_official.json --output outputs/new_green_check
.\.venv\Scripts\python.exe tools/verify_competition_native_window.py --output outputs/new_native_gui_check
```

使用新的输出目录，避免覆盖原证据。GUI 验证脚本只向所启动子进程及其后代拥有的窗口发送按键，不操作其他应用。每次交互保存 `trace.jsonl`；退出保存 `summary.json`。记录中的车辆输入注明为脚本按键或原生窗口输入，灯色始终由帧像素产生。

## 6. 本阶段边界

当前交付的是可操作的本机路口测试台。实虚线模型、实际导航解析、真实危险区判定尚未接入这条视频链路；相关规则已有独立测试，尚不等于对应视频测试通过。

原项目 BEV 是辅助示意，位置未标定、没有自车姿态或车道模型，不能据此宣称精致 3D 和真实转弯/变道效果已达标。FPGA 正式 RTL、驱动、PCIe 回传、LED/HDMI 输出与 RK3566 联合性能仍需另一台电脑验证。
