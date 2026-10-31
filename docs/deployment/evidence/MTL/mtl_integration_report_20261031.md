# B-2a / MTL / Seg-256 集成报告

日期：2026-10-31
板端：鲁班猫 RK3566（RKNN Runtime 2.3.2）
默认模型：B（未改变）
测试输入：本地 60 帧 RGB565 回放 + 本地 PGV 直播回放

## 1. 最终结果

- 新增模型：`MTL-FULL`、`MTL-FAST`，均可用。
- 模型总数：21 个；原有 A/B/C/D、B-CAPI 和历史方案全部保留。
- Qt 模型菜单已能显示并切换到 MTL-FULL / MTL-FAST / B / B-CAPI。
- MTL 检测 + 三个分割任务已进入 Qt 道路画面：车道类别、实线/虚线、斑马线、direct/alternative 可行驶区域。
- 切回 B 后分割覆盖层立即消失，目标检测与交通灯继续正常。
- 正式服务已恢复为默认 B，`ActiveState=active`。

## 2. 模型注册表

| id | 后端 | 任务 | 模型 | 输入 |
|---|---|---|---|---|
| B | rknn_lite | detect | b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn | 512×288 |
| B-capi | rknn_capi | detect | b05n_p2_multiclass_capi.rknn（符号链接） | 512×288 |
| MTL-FULL | rknn_capi | mtl | /home/cat/models/b05n_p2_mtl_b2a/b05n_p2_mtl_b2a_int8.rknn | 512×288 |
| MTL-FAST | rknn_capi | mtl | /home/cat/models/b05n_p2_mtl_b2a_seg256/b05n_p2_mtl_b2a_seg256_int8.rknn | 512×288 |

`ModelProfile` 新增 `backend`、`task` 字段。检测器选择不再依赖文件名，而是由 profile 的 `backend`/`task` 决定；`same_main` 同时比较路径、backend、task，避免符号链接或同文件不同后端被错误复用。

## 3. 交通灯过滤

结论：**正式 AI Bridge 中的 B 没有丢失交通灯**。

- 独立测试直接用 `RknnConsoleDetector`（base）时，base 会无条件丢弃 `traffic light`，所以出现 “B 8 框、B-CAPI 11 框” 的差异。
- 正式桥接使用的是 `CompetitionConsoleDetector` 适配器；它重写了检测循环，在 `single_model=True` 时保留交通灯。实测 B 的结果包中包含 class id 9（traffic light）。
- 同时修复了 B-CAPI 的真实集成问题：C API 结果绕过了适配器归一化，曾把 `pedestrian` 直接传出，而危险预警逻辑只认 `person`。现在 C API 结果统一执行：
  - `pedestrian -> person`
  - `traffic sign` 丢弃
  - motorcycle/bicycle 置信度阈值 0.85
  - 仅保留 `COMPETITION_CLASSES`
- 模型权重、conf、NMS 均未修改。

## 4. 分割数据链路

- C 库新增 `p2_detect_mtl()` 与 `p2_seg_size()`，一次 `rknn_run` 读取 15 个输出。
- 前 12 个输出沿用 Raw-head 解码/NMS；后 3 个输出按 `output_contract.json` 做 INT8 argmax：
  - `lane_cat` 9 类，288×512 / 144×256
  - `lane_style` 3 类
  - `drivable` 3 类
- 不把 mask 放进 MVA2 JSON。检测器在 `/dev/shm/adas_mtl_overlay.png` 写 RGBA PNG，JSON 元数据写 `/dev/shm/adas_mtl_overlay.json`，两者都用 `os.replace` 原子替换。
- PNG 编码在独立后台线程完成，合并最新帧；检测主路径 `overlay_ms=0`，不会阻塞推理。
- 分割 mask 使用模型输入坐标系，并按 letterbox 的 `pad_x/pad_y/resized_w/resized_h` 裁剪回道路画面内容区域。

Qt 叠加：
- `main.qml` 在 `VideoItem` 之后增加一个 `Image`，仅当 `video.aiModelId` 为 `MTL-FULL`/`MTL-FAST` 且 AI 在线时可见。
- 每 160 ms 用 URL fragment 刷新（Qt 对带 query 的 file URL 不渲染，必须用 `#`）。
- 四路布局时覆盖层跟随 `roadQuadrant`；单路全屏模式（viewMode=1）时覆盖层只占对应象限。
- 切换到非 MTL 模型时检测器 `close()` 删除 shm 文件，QML 同时立即隐藏覆盖层。
- 配色：斑马线洋红、direct 半透明绿、alternative 半透明蓝、实线红、虚线青。

## 5. MTL 效果

- MTL-FULL 输出分割 512×288；MTL-FAST 输出 256×144。
- 单帧统计示例（回放第 0 帧，道路 Q3）：
  - lane 前景像素 1925（crosswalk）+ 784（curb）+ 535（single white）
  - style：solid 1123，dashed 2121（仅统计 lane 前景）
  - drivable：direct 1769，alternative 2342
- Qt 实际截图与录屏见下。

## 6. 热切换实测

正式安装文件、本地回放、MVM1/MMS1 协议（独立进程，不占用正式 UDP 端口）：

| 切换 | 结果 | 耗时 |
|---|---|---|
| B → B-CAPI | APPLIED | 204.7 ms |
| B-CAPI → MTL-FULL | APPLIED | 296.0 ms |
| MTL-FULL → MTL-FAST | APPLIED | 288.0 ms |
| MTL-FAST → B | APPLIED | 1028.3 ms |

Qt 直播回放环境下的同一协议实测：

| 切换 | 结果 | 耗时 |
|---|---|---|
| B → MTL-FAST | APPLIED | 158.9 ms |
| MTL-FAST → MTL-FULL | APPLIED | 290.6 ms |
| MTL-FULL → B | APPLIED | 1103.6 ms |
| MTL-FAST → B-CAPI | APPLIED | 222.2 ms |
| B-CAPI → MTL-FAST | APPLIED | 149.3 ms |
| MTL-FAST → B | APPLIED | 640.7 ms |

另外单独做了 3 轮 B → B-CAPI → B 循环，6 次全部 APPLIED（113.7 / 1050.9 / 604.4 / 1027.0 / 293.1 / 1390.1 ms），无异常、无进程崩溃。切换后实际检测器类型：
- B：Python RKNN Lite，`capi=None`
- B-CAPI：`CapiDetector`
- MTL-FULL：`CapiMtlDetector`，mask 512×288
- MTL-FAST：`CapiMtlDetector`，mask 256×144

切换失败路径保留原检测器；成功切换后旧检测器 `close()` 释放 RKNN context，同一时刻只保留一个主模型。

## 7. AI Bridge 性能

同一 60 帧、10 Hz 请求、`b05p_competition_bridge` 完整链路（正式安装文件 + 新 C 库）：

| 模型 | 实际结果 FPS | 主模型 p50 / p95 | RKNN p50 / p95 | 解码/NMS p50 / p95 |
|---|---|---|---|---|
| B（Python） | 8.56 | 66.2 / 209.3 ms | 50.2 / 167.4 ms | 9.5 / 16.2 ms |
| B-CAPI | **9.39** | **44.0 / 138.7 ms** | **23.0 / 53.2 ms** | **2.3 / 3.1 ms** |
| MTL-FULL | 5.00 | 152.9 / 412.4 ms | 78.3 / 170.5 ms | 23.5 / 59.5 ms |
| MTL-FAST | 6.78 | 104.4 / 248.5 ms | 72.3 / 173.1 ms | 6.3 / 7.2 ms |

所有运行 `inference_errors=0`。B-CAPI 相对 B：FPS +9.7%，主模型 p50 -33.5%，解码 -75.8%。

## 8. Qt 渲染与叠加开销

GUI 实测采用本地 PGV 回放，Qt 使用 llvmpipe（软件 OpenGL，当前环境未加载 rockchip DRI，属于本机原有渲染路径）：

| 指标 | B（无覆盖层） | MTL-FAST（有覆盖层） |
|---|---|---|
| scene_graph_fps p50 / p95 | 4.2 / 6.7 | 3.9 / 6.2 |
| source_fps p50 | 16.2 | 15.2 |
| CPU（全核）p50 | 52.0% | 46.9% |
| AI 结果更新 p50 | 17.5 ms | 15.7 ms |

- 覆盖层带来的差异在噪声范围内（约 0.3 FPS），没有发现额外卡顿。
- PNG 编码（后台线程）：FULL 36.9 ms，FAST 9.9 ms；检测主路径耗时增量为 0。
- PNG 大小：FULL 约 6.4 KB，FAST 约 2.2 KB（单帧，压缩级别 1）。
- Qt 模型菜单显示的 AI FPS：B 约 7.2，MTL-FAST 约 5.6，MTL-FULL 约 5.6。

## 9. 截图与演示

- `research/b05n_p2_mtl/deployment/qt_overlay/screenshot_mtl_fast.png`
- `research/b05n_p2_mtl/deployment/qt_overlay/screenshot_mtl_full.png`
- `research/b05n_p2_mtl/deployment/qt_overlay/screenshot_b_overlay_cleared.png`
- 演示视频：`research/b05n_p2_mtl/deployment/qt_overlay/mtl_demo.mp4`（16 s，B → MTL-FAST → MTL-FULL → B）

截图中可看到：
- MTL-FAST：绿色可行驶区域、洋红斑马线、红色实线、青色虚线，叠加在道路 Q4 上。
- MTL-FULL：更细的分割边缘。
- B：覆盖层完全消失，车辆/行人/交通灯框保持。

## 10. 修改文件

板端 `/home/cat/qt_gpu_probe_ai_20261019/`：

- `model_profiles.py`：新增 `backend`/`task`、`wire()`、`MTL-FULL`/`MTL-FAST`。
- `b05p_competition_bridge.py`：按 backend/task 选择 `CapiDetector`/`CapiMtlDetector`；新增测试用 `--main-model` 覆盖。
- `ai_bridge.py`：
  - 构造/热切换时传递 backend/task；
  - `same_main` 比较路径 + backend + task；
  - profile 路径匹配改用 `absolute()`（修复符号链接被识别成 B）；
  - MVA2 状态中的模型选项改为紧凑格式 + 脏标记/周期重发，避免 4096 字节上限。
- `capi_detector.py`：
  - 新增 `CapiMtlDetector`；
  - 统一类别归一化（pedestrian→person 等）；
  - 异步写 RGBA PNG + JSON 元数据。
- `libp2capi.so`：新增 `p2_detect_mtl`、`p2_seg_size`（重新编译，兼容原 B-CAPI）。
- 新增符号链接 `/home/cat/cityscapes_arch_compare_20261027/models/b05n_p2_multiclass_capi.rknn`。

Qt `/home/cat/qt_gpu_probe_build_20261030_time/`：

- `main.qml`：新增 MTL 分割覆盖层和 160 ms 刷新 Timer。
- `qrc_qml.cpp` / `qrc_qml.o` / `qt_gpu_probe`：重新生成并链接。

未修改模型文件、FPGA/PCIe、视频采集、HUD、内核、RKNN Runtime 或系统配置。

## 11. 备份与回退

板端备份：

`/home/cat/b05n_p2_capi_integration_20261031/backup/mtl_stage/`

包含：
- `ai_bridge.py`、`model_profiles.py`、`b05p_competition_bridge.py`、`capi_detector.py`、`libp2capi.so`
- `qt/qt_gpu_probe`、`qt/main.qml`、`qt/qrc_qml.cpp`、`qt/qrc_qml.o`

回退：

```bash
BK=/home/cat/b05n_p2_capi_integration_20261031/backup/mtl_stage
cp $BK/ai_bridge.py $BK/model_profiles.py $BK/b05p_competition_bridge.py \
   $BK/capi_detector.py $BK/libp2capi.so /home/cat/qt_gpu_probe_ai_20261019/
cp $BK/qt/qt_gpu_probe $BK/qt/main.qml $BK/qt/qrc_qml.cpp $BK/qt/qrc_qml.o \
   /home/cat/qt_gpu_probe_build_20261030_time/
sudo systemctl restart adas-ai-bridge-live.service
# 若 Qt 正在运行，需重启 Qt 进程以恢复旧 qrc/binary
```

本地记录：`research/b05n_p2_mtl/deployment/capi_bridge/`、`deployment/qt_overlay/`。

## 12. 测试结束状态

- `adas-ai-bridge-live.service`：`ActiveState=active`
- 启动参数仍为默认 B：`b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn`
- `/dev/shm/adas_mtl_overlay.png` 不存在（无残留覆盖层）
- Qt 已恢复为正式参数：`--source-label pgv-live --road-quadrant 0`
- 无 PCIe 输入；服务仍会因 `pango-video-capture.service` 缺失输入而周期性重启，这是集成前已有的状态，不是 AI 代码故障。

## 13. 说明与限制

- GUI 验证是在本地 PGV 回放 + llvmpipe 软件 OpenGL 下完成的；Qt 的 scene_graph_fps 受软件渲染限制（约 4 FPS），不代表板端 Mali 正常显示时的上限。
- 未使用鼠标实际点击 Qt 菜单；切换通过 Qt 菜单同一个 MVM1 协议完成，Qt 状态栏正确显示 `MTL-FAST`/`MTL-FULL`/`B`，模型菜单数据源为同一份 `model_selection.options`。
- 未恢复 PCIe 实时输入，因此“真实 PCIe + Qt 实时”验收需等输入恢复后复测；桥接、切换、分割显示与清理已在本轮回放中全部验证。
