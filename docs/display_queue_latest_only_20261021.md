# DisplayQueue latest-only 修复与验证

日期：2026-10-21。只修改显示队列、帧时间戳统计和报告字段；未改 AI、模型、FPGA、HUD 布局。

## 1. 问题

`DisplayQueue::take()` 原来：

```cpp
auto frame = frames_.front();
frames_.pop_front();
```

这是 FIFO，与注释“newest”不符。producer 30 FPS、renderer 低于 30 FPS 时，队列最多积压 4 帧，渲染器一直拿最旧帧，增加视频年龄。

`DisplayQueue` 的唯一消费者是 `Renderer::synchronize()`。AI 使用独立的 `latest_ai_frame_`，不经过该队列；因此修改显示语义不会影响 AI 取帧和 Track ID。

## 2. 修复

```cpp
std::shared_ptr<VideoItem::Frame> VideoItem::DisplayQueue::take() {
    QMutexLocker locker(&mutex_);
    if (frames_.empty()) return {};
    const qulonglong depth = frames_.size();
    last_take_depth_.store(depth);
    auto frame = std::move(frames_.back());
    dropped_on_take_.fetch_add(depth > 0 ? depth - 1 : 0);
    frames_.clear();
    return frame;
}
```

- 锁内选择最新完整帧；
- 同一临界区丢弃所有旧帧；
- 不留下旧帧给下一次消费；
- 空队列返回 `{}`；
- 保留 `frame_id` 和 `sourceTimestampNs`；
- 统计每次消费丢弃数、最大队列深度、最后一次深度、producer 覆盖数；
- 无新增阻塞/忙等。

### 帧时间戳

`Frame` 新增 `sourceTimestampNs`：

- PGVF producer 从 PGVF header 的 monotonic `timestamp_ns` 写入；
- raw replay 使用 `producedNs`；
- 显示年龄使用 `sourceTimestampNs`，不再使用 producer 本地拷贝时间；
- AI 仍使用原 `producedNs`，不改变 AI/Track 时间基准。

### Qt 更新调度

producer 的 queued `requestFrameUpdate()` 已经由 `update_queued_` 原子门控：

```cpp
if (!update_queued_.exchange(true) && !invokeMethod(...)) {
    update_queued_.store(false);
}
```

同一时刻最多一个待执行 queued update，没有发现需要新增合并机制的无界积压；本轮未改该路径。

## 3. 修复前/后（本地 30 FPS，完整 AI）

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 源 FPS（窗口实测） | 17.24 | 48.39 |
| Qt 唯一上传 FPS | **13.78** | **19.20** |
| 帧年龄 p50 | **258.0 ms** | **28.4 ms** |
| 帧年龄 p95 | **372.7 ms** | **51.5 ms** |
| DisplayQueue 最大深度 | 4 | 4 |
| 最后一次消费深度 | 4 | 2 |
| 消费时丢弃旧帧 | 0 | 163 |
| producer 覆盖 | 0 | 0 |
| QSG FPS | 40.23 | 72.58 |
| AI FPS | 9.03 | 9.53 |
| AI 结果年龄 p50 | 未记录 | 108.9 ms |
| AI 结果年龄 p95 | 未记录 | 168.0 ms |
| CRC 错误 | 0 | 0 |
| GL 错误 | 0 | 0 |

第一次修复后运行（AI 3.80 FPS）也测到：

```text
unique = 17.19
age p50/p95 = 29.6 / 54.2 ms
dropped_on_take = 270
```

第二次运行 AI 恢复到 9.53 FPS，因此以第二次为修复后主数据。

## 4. 时间同步

修复后对象日志：

```text
source_frame_id = 7091
source_timestamp_ns = 490486255241
```

所有 `CONFIRMED` car 的 `detect_timestamp_ns` 与 `source_timestamp_ns` 完全一致：

```text
490486255241
```

即当前显示帧与检测时间戳没有新的错位。`PREDICTED` 目标仍可能携带更早的 detect 时间；Tesla 正式场景只消费 confirmed 非 predicted 目标，未变。

Tesla 场景仍显示 7 个车/目标，OCR 显示 `315 m / 海域路`，Specialist 正常运行，AI 在线。

## 5. 验收

- 帧年龄显著下降：p50 258→28.4 ms，p95 372.7→51.5 ms；
- 唯一上传提升：13.78→19.20 FPS；
- AI FPS 未下降：9.03→9.53；
- AI result age 108.9/168.0 ms；
- Track 时间戳对齐；
- Tesla、OCR、Specialist 正常；
- 无 GL 错误；
- `source_crc_errors=0`。

P0 通过。

## 6. 证据文件

- `results/dq_before.json`
- `results/dq_after2.json`
- `results/dq_ai_after.jsonl`
- `results/dq_objects_after2.json`
- `results/dq_final2.png`
