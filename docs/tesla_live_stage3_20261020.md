# Tesla LIVE 第三阶段：GL 修复、帧率定位、动态与视觉回归

日期：2026-10-20。分支：`live/stage2`（第三阶段改动未另开分支）。

## 1. P0：GL_INVALID_ENUM 1280

### 定位过程

在 `VideoItem::render()` 和 `TeslaSceneItem::render()` 增加分阶段 `glGetError` 检查，发现：

```text
Tesla GL error initGL 1280
```

进一步定位到 `TeslaSceneItem::initGL()` 中已废弃的 billboard 初始化：

- 2D billboard shader；
- billboard VBO/VAO；
- `QOpenGLTexture` 后视纹理。

这些资源已经不再被 LIVE 使用（周边车辆已改 3D mesh），但仍被初始化，并在 Mali-G52 上留下 `GL_INVALID_ENUM`。随后 `VideoItem` 的 RGB565 上传前读到这个遗留错误，误报为 `RGB565 texture upload failed`。

### 修复

1. 删除已废弃的 billboard GL 初始化，不再创建其 shader/VBO/VAO/texture。
2. `VideoItem` 的 RGB565 上传：
   - 内部格式改为 `GL_RGB565`（尺寸化格式）；
   - 上传前设置 `glActiveTexture(GL_TEXTURE0)`；
   - 上传前显式检查并记录遗留错误，不再把它当作本次上传失败。
3. 保留分阶段检查，确认 FBO 渲染后不再产生错误。

### 结果

重新运行带真实视频的本地回放（`local_gl_test2.log`）：

- 日志中没有 `Tesla GL error`；
- 没有 `RGB565 texture upload failed`；
- 主视频持续更新。

证据：`results/gl_fixed_local.log`、`results/local_gl_test2.json`。

## 2. P1：唯一视频帧率损失定位

### 真实 PCIe A/B（AI 开启）

| 指标 | 场景关闭 | 场景开启 |
|---|---:|---:|
| source FPS | 25.16 | 25.64 |
| 唯一帧上传 FPS | **14.96** | **13.72** |
| QSG FPS | 50.31 | 57.69 |
| YOLO FPS | 8.99 | 8.97 |
| GPU | 25% | 36% |
| CPU（全核） | 31.45% | 28.85% |

结论：**场景只造成约 8% 的唯一帧下降；关闭场景也只有 14.96 FPS，Tesla 不是主因。**

### AI 开启/关闭（同一 PGVF 本地回放）

| 指标 | AI 开 | AI 关 |
|---|---:|---:|
| source FPS | 14.29 | **30.46** |
| 唯一帧上传 FPS | **12.37** | **26.08** |
| QSG FPS | 57.14 | 60.91 |
| YOLO FPS | 8.25 | 0 |

结论：**AI/OCR 竞争是唯一帧损失的主因。** 关闭 AI 后唯一帧从 12.37 恢复到 26.08，接近源 30 FPS；Tesla 场景关闭/开启只有小的影响。

### 没有做的事

- 没有通过降低 AI 频率、关闭 OCR 或改 RKNN 来“制造”高 FPS；
- 没有把 QSG 58 FPS 当作视频 58 FPS；
- 下一步若要修复，优先做 AI/OCR 与 Qt 的 CPU 隔离/亲和性，而不是改 Tesla 渲染。

## 3. P2：动态验收

- 真实 PCIe 10 秒录屏：`results/stage2_live_sync10.mp4`（在本轮 GL 修复前采集）。
- 真实 PCIe 当前再次掉线：端点 `rev ff`、`/dev/pango_video0` 消失、`capture_io`；重启板卡后仍未恢复。
- 因此本轮 GL/帧率验证使用已有真实录制 `local_quad.rgb` 的 PGVF 回放，界面明确显示 `LOCAL REPLAY`，不冒充 PCIe LIVE。
- Track 日志：`results/stage3_final_objects.json`（`source_id=pgv-local`，含 track_id/class/frame/timestamp/validity）。
- LIVE 中无 GIF、无随机车辆；目标来自 `video.aiBoxes`。

## 4. P3：场景视觉

- 自车为程序化三维车身，位于自车车道，和道路共用相机；不再使用 QML 大贴图。
- car 使用下车身 + 梯形座舱 + 后窗/侧窗 + 车顶 + 车轮 + 尾灯。
- truck/bus/person 仍为低面数程序化模型。
- 四车道 + 路肩；扩大地面与城市剪影；小窗使用紧凑相机，空间页使用完整相机。
- 没有真实 ego speed/yaw，不滚动道路。

![LOCAL REPLAY + LIVE Tesla 小窗](<C:/Users/JingyuWang/adas-edge/deployment/lubancat/qt_gpu_probe/results/stage3_live_local.png>)

## 5. 当前限制

- 真实 PCIe 当前不可用，最终截图标为 LOCAL REPLAY。
- 唯一视频帧率仍受 AI/OCR 竞争限制；本轮只定位，没有做 CPU 亲和性修改。
- 没有 Q1 标定，所有目标仍是 `RELATIVE_ONLY`。
- 车型为低面数程序化，不是高精模型。
- 场景独立 FPS 未单独计数；QSG 不是感知更新率。
- 没有 20 分钟长测。
