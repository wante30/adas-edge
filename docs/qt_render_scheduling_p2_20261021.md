# P2：Qt VideoItem 渲染调度定位

日期：2026-10-21。保留 P0 latest-only、P1 统计修复；本阶段不降低 AI、不改 Tesla/HUD。

## 1. 先统一一个关键指标

`qt_scene_graph_render_callback_fps` 在当前工程里实际来自：

```cpp
VideoItem::Renderer::render() {
    metrics_->rendered_frames.fetch_add(1);
    ...
}
```

它统计的是 **VideoItem FBO render 次数**，不是整棵 Qt Scene Graph 的回调次数。  
因此历史日志里的“QSG 58 FPS”实际上应读作 **视频 FBO render 58 次/秒**，不是视频上传 58 帧/秒。

真正不同视频帧上传数来自：

```cpp
metrics_->uploaded_frames.fetch_add(1);
```

## 2. 60FPS 全 AI 基线

P1 已测得：

| 指标 | 全 AI 60 |
|---|---:|
| Publisher | 59.97 |
| Qt Producer | 45.04 |
| DisplayQueue 取出 | 22.99 |
| GPU 唯一上传 | 22.99 |
| VideoItem render | 58.44 |
| 消费丢弃 | 439 |
| 帧年龄 p50/p95 | 26.5 / 56.4 ms |
| AI FPS | 9.54 |
| AI 结果年龄 p50/p95 | 116.6 / 170.3 ms |

矛盾：

```text
VideoItem render ≈ 58/s
不同视频上传 ≈ 23/s
```

这不是 GPU 上传 API 慢：`glTexSubImage2D` 的 CPU 调用 p95 历史约 2–3 ms；差异发生在 **render 被调用但没有新 frame 可取/可取但被 pending 卡住** 的调度层。

## 3. 最小 A/B 实验

### A：现有路径

- render 末尾保留 `update()`；
- synchronize 仅在 `!pending_` 时 take；
- 全 AI 60：唯一上传 22.99，render 58.4，帧年龄 p95 56.4 ms。

### B：删除 render 末尾 `update()`

只用于诊断：

- 假设：58 次 render 是无新帧的空转。
- 结果：VideoItem render 58.4 → **43.5**，唯一上传 23.0 → **21.3**，帧年龄 p95 56.4 → 64.4 ms。
- 结论：**变差**，末尾 `update()` 目前是维持 Qt 重绘链的必要部分；已回退。

### C：synchronize 每次强制替换 pending

只用于诊断：

```cpp
if (auto newer = display_->take()) pending_ = std::move(newer);
```

- 结果：全 AI 60 唯一上传 **23.04**，与 A 基本不变；帧年龄 p50 24.8 ms，p95 63.5 ms；render 65.2。
- 结论：没有吞吐收益，p95 反而略差；已回退。

## 4. 为什么 render 58 次只有 23 次上传

已排除：

- Publisher 稳定 59.97；
- producer 45.04；
- DisplayQueue latest-only 正确；
- 上传 API 单次 2–3 ms，不是 58→23 的原因；
- 强制每次 take 最新帧无改善；
- 删除空转 render 反而更差。

当前证据指向：

1. `QQuickFramebufferObject` 的 render 回调中包含大量 **旧纹理重绘**；
2. `synchronize()` 的 `pending_` 生命周期和 producer 的 `requestFrameUpdate()` 不能保证每个新视频帧都对应一次同步；
3. 仅靠调整 update/take 无法解决，因为 render 与 synchronize 的 Qt 内部调度不由这两个函数完全控制；
4. 继续在 FBO 路径上小修会与 Qt 内部重绘链对抗。

**这不是一个可以用一两行代码修好的热点，而是 QQuickFramebufferObject 的合成路径问题。**

## 5. 为什么此轮不强行做 updatePaintNode 原型

候选方案是把 VideoItem 改为：

```text
QQuickItem::updatePaintNode()
  + QSGSimpleTextureNode / QSGTexture
  + 复用的 RGB565 GLES 纹理
```

它可以避免“先渲染到 FBO，再由 Scene Graph 合成 FBO”的额外一层。

但该项需要：

- 新写一个 QQuickItem；
- 管理 QSG render thread 的 GL 上下文与纹理生命周期；
- 处理 RGB565 纹理格式、mirror、viewport；
- 与现有 HUD/Tesla 和 QML 切换共存；
- 保留旧 VideoItem 作为回退。

它属于显示架构原型，不是最小改动；本轮 B/C 已证明 FBO 路径内的 update/take 调整无收益，继续在未隔离原型的情况下改会违反“不能证明收益就不要扩大改造”的约束。

## 6. 当前结论

- 60FPS 无 AI：唯一上传 **46.87**；
- 60FPS 全 AI：唯一上传 **22.99**；
- 全 AI 的 producer 45.04，renderer 23；
- 两段瓶颈必须分别解决：
  1. producer 45 vs publisher 60：AI/CPU；
  2. renderer 23 vs producer 45：QQuickFramebufferObject 调度/合成。
- 单独修好 renderer 也不能让全 AI 达到 60，因为 producer 仍只有 45。
- 继续优化需要 QSGTextureNode 原型或重构显示路径，不能靠 FBO 内小修完成。

## 7. 保留与回退

- P0 latest-only 保留；
- P1 窗口统计保留；
- P2 只增加 `synchronize_calls` 诊断计数，未改变视频消费语义；
- 删除 render update / 强制替换 pending 的试验代码已回退；
- 当前可随时回退到 `ff7e3fb` 的 P0 路径。

## 8. 证据

- `results/p2_render60.json`（删除 update，变差）
- `results/p2_sync60b.json`（强制替换 pending，无收益）
- `results/p2_ai60.jsonl`
- P1 基线：`results/p1_60_full.json`
