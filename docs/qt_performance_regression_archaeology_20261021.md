# RK3566 Qt 视频吞吐历史回归溯源

日期：2026-10-21。只读 Git 历史与历史结果，没有修改 FPGA、模型或 Qt 布局。

## 1. 结论先行

1. **44–51 FPS 确实存在，但不是当前完整 ADAS 配置。**
   - 真实 PCIe 60 FPS 输入；
   - 旧版较轻 HUD；
   - AI 关闭或仅 3–5 Hz；
   - 没有当前 Tesla LIVE 场景、完整 OCR 联合负载。

2. **独立 EGL 的 59.9 FPS 是合成内存输入。**
   - `stage_A/B/C/synthetic_source_final` 都是 `"evidence": "SYNTHETIC"`；
   - 它证明 Mali/EGL 路径能到 60，不是 Qt/QML + 当前 AI 的端到端成绩；
   - 报告自身写明 `actual_scanout_fps=null`。

3. **当前 14.78 FPS 不能直接和 44–51 比较。**
   - 当前输入是本地 30 FPS PGVF；历史是真实 60 FPS PCIe；
   - 当前 AI 约 9.5 Hz + Specialist + OCR + Tesla；
   - 历史高帧率时 AI 约 3–5 Hz，旧 HUD，无 Tesla LIVE。

4. **没有证据表明“时间戳跟踪器”本身导致视频下降。**
   - `d20c3ee` 对 `video_item.cpp` 的改动集中在 `runAiBridge` 的轨迹历史/时间戳/预测；
   - 没有改 producer、上传路径或纹理格式；
   - 当前 result age p95 约 194 ms，早期修复后约 138–191 ms，没有明显变差。

5. **真正的视频路径变化候选是 `cba87c2`**：
   - 同一提交加入完整新 HUD（`main.qml` +1679 行）；
   - 把 `latest_frame_ / takeLatest()` 改成 `DisplayQueue` + 每帧 queued `requestFrameUpdate()`；
   - 同时加入 AI sidecar/tracking；
   - 但这个提交前后的高 FPS 基线是**未提交工作区**，不能做严格同条件归因。

6. **一个确定的代码缺陷**：
   - `DisplayQueue::take()` 当前取 `frames_.front()`，注释却说“newest”；
   - 这是 FIFO，不是 latest-only；最多积压 4 帧，增加显示年龄；
   - 它解释“框/画面更旧”，但不解释 51→14 的吞吐差。修复应改 `back()`，但本轮未擅自改。

7. **当前 30 FPS 本地输入的理论上限就是 30。**
   - 无 AI 实测 27.54，已达源端约 92%；
   - 44–51 是 60 FPS 源，不能要求 30 FPS 源达到 44–51。

## 2. Git 时间线

| 提交/版本 | 时间 | 输入/AI | 唯一视频 FPS | AI/YOLO | 证据 |
|---|---|---|---:|---:|---|
| `3a6ac50` EGL Mali | 10-19 02:52 | SYNTHETIC 60，无 AI | 59.886 | 0 | `stage_A_video_only_60s` |
| `01d5c37` synthetic | 10-19 03:01 | SYNTHETIC 60，无 AI | 59.968 | 0 | `synthetic_source_final_60s` |
| `8d42623` Qt 初版 | 10-19 18:01 | 本地 replay 30，无 AI | 25.50 | 0 | `qt_no_ai_real_ch2_30` |
| `8d42623` Qt 初版 | 10-19 18:01 | 本地 replay 30，AI10 | 22.11 | 6.86 | `qt_ai_stream_30` |
| 未提交工作区 | 10-19 20:59 | **真实 PCIe 60，无 AI** | **51.36** | 0 | `qt_ui_no_ai.json`（untracked） |
| 未提交工作区 | 10-19 20:59 | **真实 PCIe 60，AI 5 Hz** | **46.62** | 4.86 | `qt_ui_optimized_5hz.json` |
| 未提交工作区 | 10-19 20:59 | **真实 PCIe 60，AI 3 Hz** | **49.36** | 0.77 | `qt_ui_optimized_3hz.json` |
| 未提交工作区 | 10-19 20:59 | **真实 PCIe 60，AI 5 Hz 单路** | **44.17** | 4.92 | `qt_ui_optimized_pcie.json` |
| `cba87c2` | 10-19 23:40 | 本地 replay 30，AI10 | 18.43 | 8.5–9.4 | `local_tracker.json` |
| `d20c3ee` | 10-20 00:28 | 本地 replay 30，AI10 | 17.33 | 9.4 | `navoff_app` |
| `d20c3ee` 联合 | 10-20 00:28 | 本地 replay 30，AI10 + OCR | 14.12 | 9.5 | `joint3.json` |
| `f9125a1` OCR 隔离 | 10-20 15:56 | 本地 replay 30，AI10 | 14.53 / 9.49 | — | `ab_off/ab_on` |
| `b61309b` Tesla LIVE | 10-20 22:02 | 本地 30，完整 AI | — | — | 功能提交 |
| `ebd0c49` 场景 A/B | 10-20 22:22 | 本地 30，完整 AI | 14.96 / 13.72 | 9.0 | `ab_scene_off/on` |
| `591ed39` GL 修复 | 10-20 22:48 | 本地 30，完整 AI | 13.72 | 9.0 | `ab_scene_on` |
| `bdd9626` 消融 | 10-20 23:16 | 本地 30，A/B/C/D | 27.54 / 17.63 / 16.68 / 14.18 | 9.24 | `p0_*` |
| `081e152` 当前 | 10-20 23:56 | 本地 30，完整 AI | **14.78** | 9.49 | `ai_roi_test2` |

注意：44–51 行的 JSON 文件为 **untracked**，时间 20:59，位于 `8d42623`（18:01）和 `cba87c2`（23:40）之间；精确 commit 不存在，标为 UNKNOWN。`
`qt_ui_optimized_5hz.json` 的 `frame_source=pgv-live`，`pgv_file=/dev/shm/pgv_qt_hud/latest.pgv`，`requested_input_fps=60`，`source_fps=51.55`，不是本地 replay。

## 3. 早期高帧率的真实条件

`qt_ui_no_ai.json`：

```text
frame_source = pgv-live
pgv_file = /dev/shm/pgv_qt_hud/latest.pgv
requested_input_fps = 60
source_fps = 60.30
unique_source_fps_average = 51.36
QSG = 60.30
AI = off
source_dropped_frames = 207
```

`qt_ui_optimized_5hz.json`：

```text
source_fps = 51.55
unique = 46.62
AI = 4.86 Hz
QSG = 56.70
GPU = 42%
NPU = 100%
old lighter HUD
```

`qt_ui_optimized_3hz.json`：

```text
source_fps = 56.12
unique = 49.36
AI = 0.77 Hz（末窗；名义 3 Hz）
```

结论：

- 输入是 **60 FPS 真实 PCIe**，不是 30 FPS 本地；
- AI 是 **3–5 Hz**，不是当前 10 Hz；
- 没有 Tesla LIVE，HUD 比当前轻；
- 即便如此最高也只有 51.36，不是严格 60。

## 4. 独立 EGL 59 FPS 是什么

`deployment/lubancat/mali_video/results/stage_A/B/C`：

- `evidence=SYNTHETIC`；
- 内存生成 RGB565 输入；
- 60 FPS；
- 无 AI 59.886；
- 有同步合成检测层 59.869；
- 完整 AI 5 分钟报告 `unique_source_submit_fps=58.216`；
- 但 `actual_scanout_fps=null`，且是独立 EGL 程序，不是 Qt Quick/QML 车机。

所以它证明“Mali 显示路径可到 60”，不证明当前 Qt 车机 + 感知栈可到 60。

## 5. 检测框滞后的真实原因

`docs/qt_yolo_tracking_fix_20261019.md` 的实测：

| 指标 | p50 | p95 |
|---|---:|---:|
| capture → AI infer start | 29.4 ms | 62.7 ms |
| RKNN infer | 58.1 ms | 104.3 ms |
| result age at Qt apply | 88.1 ms | 138.5 ms |
| display frame id - AI source frame id | 1 帧 | 3 帧 |

`d20c3ee` 后：

| 指标 | p50 | p95 |
|---|---:|---:|
| result age at Qt apply | 128.1 ms | 191.0 ms |
| display frame id - AI source frame id | 3 帧 | 5 帧 |

原因不是固定“4 帧预测”本身，而是：

1. AI 10 Hz，结果本身已经晚 88–191 ms；
2. 旧 UI 把这份旧检测直接画在当前视频帧；
3. latest-only 只保留最新请求，但请求/推理/发布链路仍有 1–5 帧年龄；
4. OCR/CPU 竞争会拉大结果间隔，旧框被保留。

修复：

- `cba87c2`：C++ IoU tracker、速度、有限预测、`takeLatest`/视频帧同步；
- `d20c3ee`：`detect_timestamp_ns`、8 个历史状态、按显示时间戳推进、`TRK TIME`；
- QML 用 `displayTimestampNs - detectTimestampNs` 外推，不再用固定 4 帧；
- 这不是给视频增加每帧推理；tracker 每 100 ms 一次，CPU 成本很小。

## 6. 关键代码差异

### `cba87c2`：视频刷新机制 + 全量 HUD

真正同时改的不只是 tracker：

- 旧的 `latest_frame_` + `takeLatest()` → 新的 `DisplayQueue`；
- 新增 `runPgvProducer`，每帧 `stat/open/read/CRC32`；
- 新增 `runAiBridge`、MFR1、AI 线程；
- `DisplayQueue::push()` 最多 4 帧，每帧调用 queued `requestFrameUpdate()`；
- `main.qml` 从 210 行扩到 1679 行，加入完整 HUD/卡片/模式；
- `setMirrorVertically(true)`、viewport、纹理复用；
- `metrics_timer_` 从 500 ms 改成 200 ms。

其中 **`DisplayQueue::take()` 取 `front()`**：

```cpp
auto frame = frames_.front();
frames_.pop_front();
```

这是 FIFO，不是 latest-only。`push()` 也在 4 帧时 `pop_front()`。这会让渲染拿到更旧的帧，最多积压 4 帧；它解释“画面更旧/框延迟”，但不直接解释 51→14 的吞吐。

### `d20c3ee`：时间戳跟踪

- 只改 `runAiBridge` 的 `Track` 历史、`detectTimestampNs`、alpha-beta、QML 外推；
- 没有改 producer、DisplayQueue、RGB565 上传或 MFR1 大小；
- 因此不能把视频下降归因于这一提交。

### `f9125a1`：OCR 隔离

- OCR 子进程、队列、TTL、状态；
- 减少 OCR 对 YOLO 的 GIL 阻塞；
- 仍消耗 CPU，但不是视频 producer 的锁。

### `b61309b` / `ebd0c49`：Tesla

- A/B 场景关闭 14.96，开启 13.72；约 -8%；
- 不是 14.78 的主因。

## 7. 为什么现在只有 14.78

当前同条件（本地 30，完整 AI）消融：

```text
A 无 AI        27.54
B YOLO         17.63
C +Specialist  16.68
D +OCR         14.18
当前 ROI 版    14.78
```

- A→B 是最大断点，约 -10 FPS；
- B→C -1 FPS；
- C→D -2.5 FPS；
- Tesla -1.2 FPS；
- 无 AI 27.54 / 30 = 91.7%；
- 历史无 AI 51.36 / 60 = 85.6%。

**同口径比例看，当前无 AI 视频路径没有退化，反而略好。**

## 8. 回答核心问题

### 为什么当时能接近输入帧率？

因为当时是：

- 真实 PCIe 60 FPS；
- 无 AI 或 AI 3–5 Hz；
- 旧轻量 HUD；
- 无 Tesla LIVE；
- latest-only 显示，允许丢中间帧。

### 当时检测框为什么滞后？

AI 结果年龄 88–191 ms，UI 又把旧结果画在当前帧；不是 Qt 视频慢，而是感知结果旧且没有时间戳推进。

### 后来修改了什么？

- `cba87c2`：完整 HUD、DisplayQueue、AI sidecar、tracker；
- `d20c3ee`：时间戳、历史、alpha-beta、QML 外推；
- `f9125a1`：OCR 进程隔离；
- `b61309b`/`ebd0c49`：Tesla LIVE；
- 之后 AI 主频提高到约 9.5–10 Hz，并保留 OCR。

### 哪项修改导致视频下降？

**没有一项被证据证明是单点代码回归。**

最可信的是：

1. **测试条件不同**：60 FPS 真实 PCIe → 30 FPS 本地回放；
2. **AI 负载增加**：3–5 Hz → 约 10 Hz + Specialist + OCR + Tesla；
3. **UI 变重**：旧轻量 HUD → 当前完整车机；
4. **`cba87c2` 的 DisplayQueue FIFO** 是真实缺陷，增加帧年龄，但不是 51→14 的主因。

### 能否保留低延迟跟踪并恢复高 FPS？

可以，但不是回滚 tracker：

1. 保留 `d20c3ee` 的时间戳跟踪；
2. 把 `DisplayQueue::take()` 改成 `back()`，真正 latest-only；
3. 恢复真实 60 FPS PCIe 后再评估；
4. AI 主频从 10 Hz 降到 5 Hz 时，历史同条件可到 46.6/60；
5. OCR 保持独立进程/低频；
6. Tesla 保留，成本约 8%；
7. 不要期望 30 FPS 本地源达到 44–51。

## 9. 最小修复建议

优先级：

1. `DisplayQueue::take()` 从 `front()` 改 `back()`，并保留 4 帧上限/清空逻辑；
2. 在真实 60 FPS 输入下复测 AI 5 Hz / 10 Hz 两档；
3. 若 10 Hz 仍低，降低 AI 主频或把 OCR 进一步移出关键 CPU；
4. 不做 tracker 回滚；
5. 不以 30 FPS 本地回放宣称 44–51 回归。

## 10. 证据缺口

- 44–51 结果文件是 **untracked**，没有对应 commit；无法做 `git bisect` 到具体代码行；
- 当前 FPGA PCIe 不在线，无法复现真实 60 FPS 输入；
- 早期高帧率与当前不是同一 HUD/AI/输入条件；
- `physical_scanout_fps` 在历史记录中均为 `null`，QSG 不能当扫描输出。
