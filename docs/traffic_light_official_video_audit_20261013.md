# 官方视频红绿灯对照与修复（2026-10-13）

本机对照和软件修复已完成；RK3566 当前未接入，新增路径的板端速度和 INT8 视频质量仍待实测。没有训练、替换网络或重新量化模型。

## 1. 输入、模型和证据

源视频目录：`D:/BaiduNetdiskDownload/紫光同创杯赛题三参考素材/`。

| 视频 | 完整帧数 | 用途 |
| --- | ---: | --- |
| `video_20260929_112507.mp4` | 1349 | 行驶、远处绿灯、转弯、行人灯与负样本 |
| `video_20260929_112454.mp4` | 106 | 两盏车辆箭头红灯；逐帧裁剪复核全部 106 帧 |
| `video_20260929_113421.mp4` | 392 | 另一个接近路口的绿灯片段，检查修改是否产生红灯误报 |

主模型：COCO-80 `yolo11n.pt`，512×288；专项：`tl-512x192/weights/best.pt`，512×192。为了与板端输入一致，首先运行这两个模型原来构建 RKNN 所用的静态 split ONNX，采用相同 RGB letterbox 和坐标还原。主模型 ONNX SHA256 `967f5382c65ac39396c0c129a6702f98848e541e4923ae2e9f31d17e0af12fde`；专项 ONNX SHA256 `e67bff26a5a91c51c58731b7477b3ac74aff7fe8634b6406f8f7f67a29e3d459`。

原有 RKNN 文件哈希本轮重新核对一致：主模型 `7a697ccc476954f6f36f7441b842257a2d7a019653cc4bad2fbf8de327c50591`；专项 hybrid2 `becfaada5da49c4dd6c143e815f8f957fb7b7ed759040d4e18c685a9c434e6f8`。

全部结果保存在 `C:/Users/JingyuWang/adas-edge/outputs/tl_official_audit_20261013/`，含视频、原始模型预测 JSONL、逐帧 CSV、哈希、测试输出和 summary。原始 `.001` 低阈值候选用于定位阈值之前是否有框；显示/融合的主模型阈值仍为 `.25`，专项候选阈值 `.05`。

## 2. 复现出的原因与对应修复

1. **输入缩小和视野稀释**：512×192 的全宽 top60 裁剪仍会把远处灯压得很小。新增可选 `adaptive` 裁剪，初始使用 top60；没有有效车辆灯时下一次刷新改用 x=20%–80%、y=25%–65% 的前方区域；检出有效灯时保持当前窗口。每次刷新只执行一次同尺寸模型推理，恢复 x/y 原点后再融合。第 600 帧的远处两灯从缺失状态补出。
2. **二次阈值丢框**：专项 `.05` 候选曾被统一 `.25` 的颜色过滤再次丢弃。现在来源分开设阈值，车辆、行人等主模型工作点不变。
3. **旧框覆盖新像素**：3 Hz 支路的旧矩形在转弯后覆盖树叶或标牌，重新分类会误报颜色。现在只在推理原帧验证发光颜色与暗色灯壳，保存该帧观测颜色和帧号；缓存最多 400 ms，时间倒退/重播时清空。后续帧复用有期限的观测，不在旧矩形里重新寻找颜色。
4. **显示范围与本车关联混用**：右侧仍可见的灯曾被 x<=76% 范围排除。显示候选范围扩展至 95%，本车道关联仍是单独未验证的功能，`decision_state` 保持 UNKNOWN。
5. **灯珠框形状误导**：专项有时只框一个灯珠，其宽高比不能证明是行人灯。新增最多 256 像素的局部暗色横向灯壳线索；无法确认类型的专项色块保留未知，不提升为车辆灯色。该线索是几何候选，不能证明朝向或本车道归属。
6. **刷新时机漂移**：浮点 `>=1/3` 判断有时把精确的 30 FPS 第 30/40 帧刷新推迟，累计成 16 次而非 17 次更新。加入微小边界容差；169 帧 3 Hz 回放稳定更新 17 次。PC `.pt` 验证还增加与 RKNN 一致的显式画布及正确 BGR/RGB 转换，避免比较自动 padding 与固定图的不同输入。

## 3. 统一帧率对照结果

原始静态模型已逐帧推理完整三段视频。对照随后复用同一份原始输出，按源视频时间模拟主支路 10 Hz、专项 3 Hz，排除因不同电脑速度造成的刷新差异。这是 PC 缓存回放，不是板端 INT8 结果。

`112507` 最后 169 帧使用既有人工复核标签：41 帧可见车辆绿灯、111 帧已复核车辆绿灯负样本，17 帧过渡排除。指标是颜色状态帧级覆盖，不是逐灯检测 AP，也不是安全通行准确率。

| 方案 | 车辆绿灯 TP / 41 | 车辆绿灯 FP / 111 | 行人红灯 TP / 13 | 行人红灯 FP / 139 |
| --- | ---: | ---: | ---: | ---: |
| 主模型与原后处理 | 30 | 0 | 8 | 7 |
| 专项与原后处理 | 0 | 18 | 0 | 0 |
| 主模型+专项，原融合 | 30 | 18 | 8 | 7 |
| 专项+修复后的处理 | 38 | 0 | 8 | 1 |
| 主模型+专项，修复后融合 | 38 | 0 | 8 | 1 |

对应 `adaptive_frozen_observation/summary.json`。绿色覆盖由 73.17% 到 92.68%，但这只覆盖上述 41 帧，并且此段参与了调试，不能当泛化结果。行人灯仍有漏检及一个延迟显示帧。

逐帧复核的 `112454` 车辆红灯 106 帧中，主模型与修复后融合均正确显示 103 帧；前三帧是 10 Hz 确认延迟。专项单独修复后只有 27/106，**不能替代主模型**。该片段只有红灯正例，不能据此计算红灯 precision。对应 `adaptive_red_final/summary.json`。

另一段 `113421` 的诊断状态输出：原融合 123 帧 GREEN，修复后 336 帧 GREEN，0 帧 RED。该段没有逐帧检测框 GT，不能把 336/392 称为识别准确率。对应 `adaptive_green_final/summary.json`。

## 4. 实际 PC 运行入口复核

除静态 ONNX 的可控回放外，也真实运行 `.pt` 主模型和专项模型，经实际运行适配器、车道候选、颜色过滤与视频编码。最终 `pt_scheduler_final` 的 169 帧中：车辆绿灯 33/41、FP 0；行人红灯 11/13、FP 0。与统一 10 Hz 回放的 38/41 不同，不能混用：现有 benchmark 主模型逐帧执行，确认计数和取样时机不同，且 PyTorch 与导出图存在数值差异。

该实际入口仍有 8 个车辆绿灯正例帧未稳定显示，不能宣称全部漏检已消除。`pt_scheduler_final/quality.json` 保留了完整指标。

## 5. RK3566 边界与待验收项

本轮没有板端连接，因此 `RK3566_GATE=NOT_RUN_BOARD_DISCONNECTED`。

设计保持主模型 512×288、专项 512×192、专项 3 Hz，每次一次推理，两个 RKNN 文件均未改动。新功能是裁剪、坐标还原和有界 CPU 后处理，不涉及更大的模型。统一缓存回放的 PC 后处理均值约 0.185 ms/源帧；这不能代替 ARM CPU 结果。原专项模型的历史 NPU 均值 25.29 ms 已有记录，但不能直接作为改造后整个系统的速度证明。

板端必须重新测：真实 INT8 是否保留该视频改善、主 10 Hz + 专项 3 Hz 的刷新率、预处理/解码/后处理分项、p95 结果年龄、CPU/RSS，以及三段视频中的误报。若预算失败，应保留主模型并关闭 adaptive 或降低支路频率，不能用 PC FP32 结果跳过验收。

## 6. 复现入口

PC：

```powershell
cd C:\Users\JingyuWang\adas-edge
.\.venv\Scripts\python.exe tools\benchmark_front_pipeline.py `
  --video "D:\BaiduNetdiskDownload\紫光同创杯赛题三参考素材\video_20260929_112507.mp4" `
  --model .\yolo11n.pt `
  --traffic-light-model "D:\Datasets\adas_edge_training\tl_specialist\tl-512x192\weights\best.pt" `
  --traffic-light-roi-mode adaptive --traffic-light-fps 3 `
  --confidence 0.25 --input-size 512x288 --lane-stride 3 `
  --start-frame 1180 --max-frames 169 --output-dir outputs\tl_new_pc_check
```

RK3566 接入后，在包含最新版源码的板端 checkout 运行，先按实际路径调整视频及模型：

```sh
/usr/bin/python3 tools/benchmark_front_pipeline.py \
  --video videos/front_eval_original.mp4 \
  --model rect_models/yolo11n_rk3566_int8_split_512x288.rknn \
  --traffic-light-model tl_check/tl_specialist_512x192_hybrid2.rknn \
  --traffic-light-roi-mode adaptive --traffic-light-fps 3 \
  --confidence 0.25 --input-size 512x288 --lane-stride 3 \
  --max-frames 300 --output-dir outputs/tl_fix_board_check
```

视频：`traffic_light_before_after_h264.mp4` 是 1349 帧完整对照；`far_signal_comparison_h264.mp4` 是远处灯的 8 秒对照。原始图和其他 trial 保留用于追溯。
