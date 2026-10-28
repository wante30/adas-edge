# P1：Qt 视频消费通量与 30/60 FPS 对照

日期：2026-10-21。P0 latest-only 修复保留，不修改 AI 频率、模型、Tesla 或 HUD 布局。

## 1. 统计口径修正

发现 `VideoItem::updateMetrics()` 的统计错误：

```cpp
const double seconds = stats_clock_.elapsed() / 1000.0;
input_fps = (produced - last_produced) / seconds;
```

但 `stats_clock_` 没有 restart，`last_produced/last_consumed/last_rendered` 也没有更新，导致显示为“启动以来累计帧数 / 启动以来时间”，并产生 17.24、48.39 等不稳定数字。

修复：

```cpp
const double seconds = stats_clock_.restart() / 1000.0;
input_fps = (produced - last_produced) / seconds;
consumed_fps = (consumed - last_consumed) / seconds;
render_fps = (rendered - last_rendered) / seconds;
last_produced = produced;
last_consumed = consumed;
last_rendered = rendered;
last_cpu_ticks_ = cpuTicks;
```

同时新增 `display_queue_taken` 计数和 `sourceTimestampNs` 保留。  
读取 PGVF header 的 Publisher 实测：

```text
30 FPS 配置：29.97 FPS（5 秒 150 次）
60 FPS 配置：59.97 FPS（5 秒 300 次）
```

时间基准：

- Publisher `timestamp_ns`：`time.monotonic_ns()`；
- Qt `steady_clock`：Linux `CLOCK_MONOTONIC`；
- AI `time.monotonic_ns()`；
- 三者同一时钟域；原始素材索引另存为 `sourceIndex`，未与运行时 frame_id 混淆。

## 2. 30 FPS 对照

| 指标 | A 无 AI | B 全 AI |
|---|---:|---:|
| Publisher（header 实测） | 29.97 | 29.97 |
| Qt producer 接受帧/时长 | 568 / 19.84 = **28.63** | 560 / 20.12 = **27.83** |
| DisplayQueue 取出帧 | 561 = **28.28** | 364 = **18.09** |
| GPU 上传唯一帧 | 561 = **28.28** | 364 = **18.09** |
| 消费时丢弃旧帧 | 7 | **194** |
| 最大队列深度 | 4 | 4 |
| 帧年龄 p50 | 27.4 ms | 32.9 ms |
| 帧年龄 p95 | 30.3 ms | 57.2 ms |
| QSG FPS | 60.0 | 54.2 |
| AI FPS | 0 | 9.54 |
| CRC 错误 | 0 | 0 |
| GL 错误 | 0 | 0 |

30 FPS 结论：

- Publisher 稳定 29.97；
- Qt producer 27.8–28.6，已接近源；
- 无 AI 时 renderer/upload 28.28，几乎追平 producer；
- 全 AI 时 upload 降到 18.09，丢帧发生在 **renderer/DisplayQueue 消费侧**，不是 Publisher；
- 队列每次消费平均丢弃约 0.53 帧（194/364），latest-only 正在正常工作。

## 3. 60 FPS 对照

使用真正连续变化的 60 FPS 官方四宫格插帧序列（477 个 60fps 帧，不是固定图重复）。

| 指标 | C 无 AI | D 全 AI |
|---|---:|---:|
| Publisher（header 实测） | 59.97 | 59.97 |
| Qt producer 接受帧/时长 | 1126 / 19.82 = **56.81** | 905 / 20.09 = **45.04** |
| DisplayQueue 取出帧 | 929 = **46.87** | 462 = **22.99** |
| GPU 上传唯一帧 | 929 = **46.87** | 462 = **22.99** |
| 消费时丢弃旧帧 | 196 | **439** |
| 帧年龄 p50 | 28.4 ms | 26.5 ms |
| 帧年龄 p95 | 31.0 ms | 56.4 ms |
| QSG FPS | 66.7 | 58.4 |
| AI FPS | 0 | 9.54 |
| AI 结果年龄 p50/p95 | — | 116.6 / 170.3 ms |
| CRC 错误 | 0 | 0 |
| GL 错误 | 0 | 0 |

60 FPS 结论：

- 无 AI：Publisher 59.97，Qt producer 56.81，upload 46.87；
- 全 AI：producer 45.04，upload 22.99；
- AI 维持 9.54 FPS，没有为了显示关闭 AI；
- 显示丢帧主要发生在 producer→renderer 之间的 CPU/渲染竞争；
- 帧年龄仍保持 p95 56.4 ms，P0 低延迟没有回退。

## 4. 当前真正的瓶颈

按每秒不同帧拆解：

```text
30 全 AI:
Publisher 29.97
  → Qt producer 27.83   (-2.1)
  → renderer/upload 18.09 (-9.7)
  → queue dropped 194

60 全 AI:
Publisher 59.97
  → Qt producer 45.04   (-14.9)
  → renderer/upload 22.99 (-22.1)
  → queue dropped 439
```

无 AI 时：

```text
30: producer 28.63 → upload 28.28
60: producer 56.81 → upload 46.87
```

因此：

1. Publisher 不是瓶颈；
2. Qt producer 在 60 FPS 时受 CPU 调度影响；
3. **最大丢帧点是 Qt renderer/FBO 消费**；
4. 纹理上传 API 本身不是主因，主要差异出现在有 AI 时；
5. 队列 latest-only 正在丢弃过期帧，这是预期行为，不是回滚理由。

## 5. 本轮代码修改

- 修复 `updateMetrics()` 的时间窗口和 last 计数；
- 新增 DisplayQueue `taken` 统计；
- 新增 `sourceTimestampNs` 保留和帧年龄 p50/p95 统计；
- 保留 P0 的 `take() = back() + clear()`；
- 未改 AI 频率、模型、OCR、Specialist、Tesla、HUD 布局；
- 未新增 CPU 绑核。

## 6. 验收

- P0 latest-only 保留；
- 30 FPS 无 AI 28.28，全 AI 18.09；
- 60 FPS 无 AI 46.87，全 AI 22.99；
- 帧年龄 p95 51–57 ms，未回退到 P0 前；
- AI 9.54 FPS，OCR/Specialist/Tesla 正常；
- 无 GL/CRC 错误；
- 未用旧帧排队或固定图片重复制造高 FPS。

目前没有发现可用单点小修把全 AI 60 FPS 拉到 60；继续提升需要更深层的 AI/渲染资源隔离或改变传输协议，不属于本轮最小修复。
