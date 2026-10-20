# Tesla-BevCam-Clone 原版恢复（第一轮）

日期：2026-10-20。分支：`restore/tesla-original`。本分支只做视觉恢复，不继续扩展 GPU 场景。

## P0 审计

### 原版资产

- `deployment/lubancat/qt_gpu_probe/assets/tesla_bev_demo.gif`
- 尺寸 720×460，40 帧，原版空间演示动画。
- 历史提交 `cba87c2` 的 `main.qml` 中，空间页使用 `AnimatedImage` 播放这个 GIF。
- 历史截图 `results/qt_ui_tesla_spatial.png` 显示过右下角 `Tesla BEV 空间` 卡片，中央为同一 Tesla 动画。

### 当前版本错在哪里

- 最近提交 `bba64cb` 起，右下角一直是 `ADAS 空间识别 / LiveBev`，不是 Tesla 卡片。
- 提交 `01e2d6d` 新增的中央 `TeslaSceneItem` 是新的 OpenGL 替代场景，不是原版复用。
- 自车使用 205 px QML 贴图，和场景不共用相机/尺度/遮挡。
- 周边车辆使用同一张后视 billboard，LIVE 位置由非标定框映射得到，导致拥挤和跳变。
- 原版完整视觉被替换，且同时存在中央 GPU 场景和右下 2D LiveBev 两套空间表达。

结论：**上一版不能算原版恢复；它是一个新场景。**

## P1 本轮恢复

### 单路页面

中央：真实 Q1 道路视频 + YOLO 框。

右下角：

- 标题恢复为 `Tesla BEV 空间`；
- 徽标 `DEMO`；
- 内容为原版 `tesla_bev_demo.gif`；
- 页脚 `TESLA-BEVCAM-CLONE · 原版动态演示 · 非标定`。

### 空间视图页面

- 点击“空间视图”后，放大同一份 `TeslaBevView`；
- 与右下角共用 `components/TeslaBevView.qml` 和同一 GIF 资产；
- 不再实例化 `TeslaSceneItem`；
- 不再维护中央 GPU 场景与 LiveBev 两套空间逻辑。

### 移除的替代物

- 中央 `TeslaSceneItem` 实例（C++ 文件保留在历史分支，不再由 QML 使用）；
- 205 px `egoOverlay` 贴图；
- `SCENE LIVE/SCENE DEMO` 新场景模式切换；
- 右下角 LiveBev 的 `ADAS 空间识别` 卡片。

## 实际截图

- 单路 + 右下角原版 Tesla 卡片：`results/restore_single.png`
- 空间视图放大同一组件：`results/restore_spatial.png`
- 历史原版参考：`results/qt_ui_tesla_spatial.png`

## 如实说明

1. 原版 GIF 本身没有清晰可见的自车三维模型；它不是实时场景，也没有可靠的 ego 运动。
2. 因此本轮没有伪造自车。P2 要求的“自车可见且比例正确”如果指原版 GIF 内没有的自车，需要下一轮基于原版 `render_tesla_ui` 的 ego 重新生成资产，而不是继续用贴图。
3. 右下卡片为 88 px 高的裁切显示，保持原版道路/车辆主体；空间页显示完整 GIF。
4. 当前真实 PCIe、AI、HUD 仍在运行，但 Tesla 卡片是 **DEMO 动画**，不冒充 LIVE。
5. 本轮按任务要求停在第一轮视觉恢复，等待用户确认外观后再做动态和性能。

## 文件

- `deployment/lubancat/qt_gpu_probe/components/TeslaBevView.qml`
- `deployment/lubancat/qt_gpu_probe/main.qml`
- `deployment/lubancat/qt_gpu_probe/qml.qrc`
- 结果：`results/restore_single.png`、`results/restore_spatial.png`
