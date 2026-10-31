# MTL 分割截图

| 文件 | 模型 | 说明 |
|---|---|---|
| `screenshot_mtl_fast.png` | MTL-FAST | 256×144 mask 叠加，绿色可行驶/洋红斑马线/红实线/青虚线 |
| `screenshot_mtl_full.png` | MTL-FULL | 512×288 mask 叠加 |
| `screenshot_b_overlay_cleared.png` | B | 切回 B 后覆盖层清除，检测框保留 |

条件：RK3566 板端 Qt 1280×720，本地 PGV 回放，道路 Q4，llvmpipe 软件 OpenGL。
演示视频不进入普通 Git，本机路径见 `docs/hmi/MTL_SEGMENTATION_OVERLAY.md`。
