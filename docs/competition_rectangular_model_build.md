# 矩形 ONNX / RKNN INT8 模型构建记录

更新日期：2026-09-17
源检查点：COCO `yolo11n.pt`，SHA-256 `0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1`
（**未做任何 fine-tune**：本轮冻结结论第 3 条，比赛 detector 继续使用原始 COCO-80 权重）

## 1. 构建链路

```
yolo11n.pt
  -> 静态矩形 ONNX（imgsz=(H,W)，dynamic=False）
  -> split-output ONNX（把最后的 concat 拆成 boxes / scores 两个输出）
  -> RKNN INT8（rknn-toolkit2 2.3.2，per-channel，target rk3566）
```

`boxes [1,4,N]` 与 `scores [1,80,N]` 分开输出是既有已验证做法：YOLO11 导出后原本是单个 `[1,84,N]`，框坐标（0–W / 0–H）与类别概率（0–1）混在一个张量里，整张量 INT8 量化时尺度被坐标范围主导，恢复出的类别分数全为 0。拆开后两个分支各自量化。

## 2. 产物与哈希

| 画布 | 静态 ONNX 输入 | 静态 ONNX 输出 | split 输出 | ONNX SHA-256 | split SHA-256 | RKNN INT8 SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| 320×192 | `[1,3,192,320]` | `[1,84,1260]` | `boxes[1,4,1260]` + `scores[1,80,1260]` | `4B97801214C789CA9FD0E304C682F5C7EAE5AA25C4667DCC5DE9CD1FEE93D4C2` | `7C4BA018A38CCA427EFB3E969329CE0AFA350801BA17D87DEE54EB7681F72345` | `543B8C572737A1D5B89AFFB6D31EA8D3343741A9525FA75404CDC60F084BCF87` |
| 416×256 | `[1,3,256,416]` | `[1,84,2184]` | `boxes[1,4,2184]` + `scores[1,80,2184]` | `0CF5CA53D3132A89689106B5F458A5A4B326A049699A6F1D847088C6F475FFE6` | `58D5A6EC5A4D71F76047DF9AA7C18B3A8B46122610EF92693B481D7AA4A81D6D` | `29E8BEFBA1C006AE1B8FDB2027DC02FEF7606835B982FEAB43037B21B022B7E8` |
| 512×288 | `[1,3,288,512]` | `[1,84,3024]` | `boxes[1,4,3024]` + `scores[1,80,3024]` | `5B0C54591A5DBB7D27425B74F030FA9B7B59DA3A93E6E2E3B2207B8E677B5EF5` | `967F5382C65AC39396C0C129A6702F98848E541E4923AE2E9F31D17E0AF12FDE` | `7A697CCC476954F6F36F7441B842257A2D7A019653CC4BAD2FBF8DE327C50591` |
| 640×352 | `[1,3,352,640]` | `[1,84,4620]` | `boxes[1,4,4620]` + `scores[1,80,4620]` | `279A668F71648CD414159C5CEBDD7C1637610F521E158F40BF78557E185942A4` | `F66471E24CDBE65D58EA191292F5474C31E36E3F87DC0A81621E267CC8972047` | `66D9EA0B965B9D0014E23503E1D085030D3802CACBDF031115A8E718A478D1C0` |

锚点数自洽（stride 8/16/32 三层求和）：1260 = 10·6 + 20·12 + 40·24；2184 = 13·8 + 26·16 + 52·32；3024 = 16·9 + 32·18 + 64·36；4620 = 20·11 + 40·22 + 80·44。

**640×360 未构建**：360 不是 32 的倍数。构建 640×352 作为最接近的合法画布（见 [指标协议](competition_detection_metric_protocol.md) §5）。

## 3. 量化配置

| 项 | 值 |
| --- | --- |
| toolkit | `rknn-toolkit2 2.3.2` |
| 转换环境 | WSL Ubuntu，`~/.venvs/adas-rknn310`，Python 3.10.20，onnx 1.16.1，protobuf 3.20.3 |
| target | `rk3566` |
| mean / std | `[[0,0,0]]` / `[[255,255,255]]` |
| `quantized_method` | `channel`（逐通道，toolkit 默认） |
| `quantized_algorithm` | `normal`（toolkit 默认） |
| `optimization_level` | 3（默认） |
| 板端运行时 | rknn-toolkit-lite2 / librknnrt 2.3.2，RKNPU driver v0.8.2 |
| 板端输入 | uint8，NHWC，RGB |

板端输入约定（uint8 0–255 + `mean=0,std=255`）与 ONNX 静态图约定（float32 0–1）是**同一套归一化的两种写法**，这一点经实测确认：把 0–255 直接喂给 ONNX 会得到 class 67、置信度 1.0 的垃圾输出。

## 4. 校准样本

上一轮 INT8 用的是 **25 张 COCO128 原图**（`tmp/rk3566_profile/calibration_wsl.txt`），既不是道路域、也不是部署画布，toolkit 还会自己做一次 resize。本轮换成：

| 项 | 值 |
| --- | --- |
| 生成器 | `tools/prepare_rect_calibration.py` |
| 样本数 | 每画布 **76** 张 |
| 布局 / 通道 / dtype | **NCHW** / RGB / uint8 |
| 素材 | `outputs/traffic_unannotated_source.mp4`（1920×1080，43.4 s）+ `outputs/road_sample_source.mp4`（640×360） |
| 冻结 holdout | **未使用**（工具会在源头拒绝 holdout 路径） |
| 预处理一致性 | 用与部署同一条 `letterbox_bgr`，灰边 padding 直接在数据里 |

两个只能靠实测发现的约束，已写进工具：

1. **RKNN 的 `.npy` 数据集要求 NCHW**。传 NHWC 会直接报
   `ValueError: The input(...) shape (1, 192, 320, 3) is wrong, expect 'nchw' like (1, 3, 192, 320)!`
   —— 这也正是选择写 `.npy` 而不是写图片路径的原因：图片路径会走 `cv2.imread`（BGR），布局和通道顺序都不可控。
2. 画布必须两侧都是 32 的倍数，否则 `check_imgsz` 静默取整。

已知限制：**校准只覆盖白天/密集交通**。仓库里只有已标注渲染版的夜间视频，没有可用的原始夜间素材；把带绿框的渲染帧喂进量化会污染数据，所以没有使用。

## 5. 构建正确性（Phase 7 摘要）

在同一批 40 张道路帧、同一 conf 0.25 / NMS IoU 0.7、同一坐标还原下：

| 画布 | 后端 | 匹配率（vs PyTorch FP32） | 平均 IoU | 平均 ｜Δ置信度｜ | 判定 |
| --- | --- | ---: | ---: | ---: | --- |
| 320×192 | ONNX | 1.0000 | 0.9988 | 0.0004 | 一致 |
| 320×192 | RKNN INT8 | 0.8922 | 0.7373 | 0.0273 | 一致（>0.80 门限） |
| 416×256 | ONNX | 1.0000 | 0.9986 | 0.0004 | 一致 |
| 416×256 | RKNN INT8 | 0.8455 | 0.7246 | 0.0444 | 一致 |
| 512×288 | ONNX | 1.0000 | 0.9993 | 0.0004 | 一致 |
| 512×288 | RKNN INT8 | 0.8586 | 0.6997 | 0.0349 | 一致 |
| 640×352 | ONNX | 0.9988 | 0.9994 | 0.0004 | 一致 |
| 640×352 | RKNN INT8 | 0.8304 | 0.7436 | 0.0570 | 一致 |

- **导出无损**：ONNX 与 PyTorch 的检测框数量逐画布完全一致（269/269、699/699、792/792、807/808），匹配率 1.0。
- **INT8 有明确退化**：83–89% 的框仍能在 IoU ≥ 0.5 下匹配，置信度漂移 0.03–0.06；但虚检明显膨胀（640×352 上 808 → 1480 个框）。
- 退化机理：`boxes` 分支的数值范围 ∝ 画布像素，画布越大框坐标的 INT8 量化步长越粗（512×288 的上限 512 → 约 2 px/step），这与观察到的"画布越大平均 IoU 越低"方向一致。

按 Phase 7 的要求，**先排查量化再进 benchmark**：导出已确认无损，因此退化来自 INT8 本身。

### 量化设置已排查：`normal` + `channel` 就是更优解

额外用 `--quantized-algorithm mmse` 重建了 512×288 并做了同样的板端比对：

| 后端 | 匹配率 | 平均 IoU | 平均 ｜Δconf｜ | 板端 NPU mean | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| PyTorch FP32（参考） | — | — | — | — | — |
| ONNX | **1.0000** | 0.9993 | 0.0004 | — | 一致 |
| RKNN INT8 `normal`（采用） | **0.8586** | 0.6997 | 0.0349 | 30.42 ms | 一致（>0.80 门限） |
| RKNN INT8 `mmse` | 0.7336 | 0.6599 | 0.0596 | 32.41 ms | **gross mismatch** |

mmse 又慢（+6.5%）又更不准（匹配率 0.734 < 0.80 门限），因此**保留 toolkit 默认的 `normal` + `channel`**。

同时确认 `quantized_method` 的 toolkit 默认值本来就是 `channel`（逐通道），此前已验证的模型并未退化成逐张量量化。

**结论**：观测到的 INT8 退化（IoU ≥ 0.5 下丢约 14% 的框、置信度漂移 0.03–0.06）不是配置问题，
而是在这个画布尺寸下 INT8 的固有代价——`boxes` 分支的数值范围正比于画布像素（512×288 下上限 512，
约 2 px/量化步长），这正是画布越大平均 IoU 越低的原因。

原始数据：`D:/Datasets/adas_edge_training/phase10/cmp_mmse.md`。

## 6. 复现

```powershell
# 1) 静态 ONNX + split
foreach ($s in @("320x192","416x256","512x288","640x352")) {
  python deployment/rknn/export_onnx.py --weights yolo11n.pt --imgsz $s --output "models/yolo11n_$s.onnx"
  python deployment/rknn/split_yolo_output.py --input "models/yolo11n_$s.onnx" --output "models/yolo11n_${s}_split.onnx"
}
# 2) 校准（道路素材，绝不用 holdout）
python tools/prepare_rect_calibration.py `
  --source outputs/traffic_unannotated_source.mp4 --source outputs/road_sample_source.mp4 `
  --width 512 --height 288 --limit 80 --stride 15 `
  --output-dir models/calib_512x288 --manifest models/calib_512x288.txt `
  --linux-root /mnt/d/.../models/calib_512x288
```

```bash
# 3) 在 WSL 里转换
~/.venvs/adas-rknn310/bin/python deployment/rknn/convert_rknn.py \
  --onnx /mnt/d/.../models/yolo11n_512x288_split.onnx \
  --dataset /mnt/d/.../models/calib_512x288.txt \
  --output /mnt/d/.../models/yolo11n_rk3566_int8_split_512x288.rknn \
  --precision int8
```

产物目录：`D:/Datasets/adas_edge_training/rect_models/`（含 `manifest.json` / `manifest.md`）。
清单生成：`python tools/rect_model_manifest.py --models-dir <dir> --out manifest.json`。
