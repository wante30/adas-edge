# 检测指标评测协议（冻结）

更新日期：2026-09-17

本文件定义本项目**唯一允许**的检测指标口径。此后任何报告出现检测数字，都必须写明它属于下表哪一行，并写明输入画布。

## 1. 三套指标互不相等，禁止混用

| 口径 | 定义 | 工具 | 用在哪 |
|---|---|---|---|
| **macro recall（项目历史口径）** | 每类做贪心一对一匹配，单点置信度（默认 0.25）、单点 IoU（0.5），再对 7 个类取平均 | `tools/evaluate_bdd_yolo.py` | 与 2026-09 之前的历史报告对齐 |
| **AP50** | IoU=0.5 下 PR 曲线下面积，按类计算 | pycocotools | 论文可比 |
| **AP50:95** | 10 个 IoU 阈值（0.50:0.05:0.95）AP 的平均，COCO 标准头条指标 | pycocotools | 论文可比 |

**三者不是同一个量的不同精度，而是不同的量。** 尤其在 512×288 上 mAP50 = 0.1998 而 macro recall@0.25 = 0.1795 —— 数值接近纯属巧合：前者是整条 PR 曲线，后者是单个工作点；换一个置信度，macro recall 会变而 mAP50 不变。

硬规则：

- 报告 mAP 时必须写 `mAP50` 或 `mAP50:95`，不许写"mAP"。
- 不许把 macro recall 写成"recall/mAP"或与 mAP 并列求平均。
- 不许用 macro recall 的数字去和论文的 mAP 比较。
- AP 类指标（AP50 / AP50:95 / AP_small）**从不**依赖置信度阈值；贪心匹配类指标**必然**依赖。报告 AP 时附 conf 是无意义的，报告 recall 时**必须**附 conf。

## 2. 冻结评测集

- 路径：`data/competition_eval/bdd100k_validation/bdd100k/`
  - 图像：`images/100k/val/*.jpg`（10,000 张，1280×720）
  - 标注：`bdd100k.csv`（186,033 行逐框标注）
- **使用切片**：按文件名排序的**前 1000 张**，名单固化在 `D:/Datasets/adas_edge_training/bdd_holdout_1000_names.txt`。
- 永久冻结。**不得**用于：训练、量化校准样本选择、超参搜索、任何形式的模型选择。
- 校准样本必须来自别处（见 §6）。

> 运维提醒：这 1000 张图同时被 `local-round2-full-history` 分支跟踪，而发布分支的 `.gitignore` 忽略 `data/`。切换到不跟踪它们的分支会**把它们从工作区删除**（2026-09-17 实际发生过一次，已从 git 对象恢复）。切换分支后先确认 `images/100k/val` 有 10,000 个文件。

## 3. 类别映射（唯一允许的一套）

| BDD100K | 本项目类别 | COCO category id |
|---|---|---|
| `pedestrian` | person | 1 |
| `bicycle` | bicycle | 2 |
| `car` | car | 3 |
| `motorcycle` | motorcycle | 4 |
| `bus` | bus | 5 |
| `truck` | truck | 6 |
| `traffic light` | traffic light | 7 |

**丢弃**：`rider`、`traffic sign`、`train`、`trailer`、`other vehicle`、`other person`。

`rider`（骑车人）**不**并入 `person`：并入会抬高行人召回并破坏与历史报告的可比性。丢弃数量在报告里逐项给出（`rows_outside_holdout` 与 `dropped_rows` 分开计数，前者是"不属于这个切片"，后者是"类不在分类体系内或框退化"）。

## 4. COCO 评测参数（必须逐项一致）

| 参数 | 值 |
|---|---|
| `iouType` | `bbox` |
| `catIds` | 上述 7 个 id，按 person…traffic light 顺序 |
| `areaRng` | `[0,1e10]`、`[0,32²]`、`[32²,96²]`、`[96²,1e10]` |
| `maxDets` | `[1, 10, 100]` —— 每图最多 100 框是 COCO 标准 |
| 预测分数下限 | 0.001（为了让 PR 曲线完整） |
| `iscrowd` | 全部 0（BDD100K 无 crowd 标注） |

预测的 dump 会保存到 `preds-<label>.json`，包含每图全部低分框；阈值诊断（Phase 5）复用同一份 dump，**不重跑模型**，因此不同 conf 下比较的是同一批预测。

## 5. 输入画布契约

1. 画布必须**静态且显式**：`--width` 与 `--height` 分开给，没有"一个 int 当正方形"的捷径。
2. 画布两侧必须是 32 的倍数（YOLO 最大 stride）。
3. 部署用**固定画布**（`rect=False`）。Ultralytics 预测默认 `rect=True`，会按批内宽高比缩成最小矩形 —— 那是逐帧变化的动态形状，和冻结的 ONNX/RKNN 图不是一回事，两者不可混用。报告里记录**网络实际收到的张量形状**。
4. **640×360 不是合法画布**：360 不是 32 的倍数。`check_imgsz` 会把它**静默取整成 640×384**，不会报错。实测：

   ```
   requested 640x360  stride_legal=False  -> check_imgsz(max_dim=2) = [384, 640]
   requested 640x352  stride_legal=True   -> check_imgsz(max_dim=2) = [352, 640]
   ```

   因此高质量前向候选用 **640×352**（离 360 最近向下取整，16:9 内容只多 8 px 竖向 padding），并把 **640×384** 作为"静默替换会发生什么"的对照一起测。
5. 精确 16:9 且 stride 合法的画布只有 `512k × 288k`：k=1 是 512×288，k=2 是 1024×576（远超 RK3566 单路预算）。16:9 内容在 320×192 / 416×256 上各约有 6% / 9% 画布面积是灰边。

## 6. 量化校准

RKNN INT8 的 `dataset` 列表里的图**必须已经是目标画布的 letterbox 结果**。若直接给原始 1280×720 道路帧，工具会自行 resize 成模型输入，拉伸方式与本项目的 letterbox 不一致，量化看到的数据分布就不是推理时看到的那一个。

生成器：`tools/prepare_rect_calibration.py`（用与部署同一套 `letterbox_bgr`）。校准帧只能来自 holdout 之外的道路素材。

## 7. 复现命令

```powershell
# 标准 COCO 指标（PC FP32）
python tools/evaluate_bdd_coco.py --model yolo11n.pt ^
  --bdd-root data/competition_eval/bdd100k_validation/bdd100k ^
  --width 512 --height 288 --limit 1000 --device 0 ^
  --out out/BASE-512x288.json --dump-predictions out/preds-BASE-512x288.json

# 历史口径（macro recall），仅用于与旧报告对齐
python tools/evaluate_bdd_yolo.py --model yolo11n.pt ^
  --bdd-root data/competition_eval/bdd100k_validation/bdd100k ^
  --imgsz 512x288 --geometry fixed --limit 1000 --device 0 ^
  --out out/hist-512x288.json
```

依赖：`pycocotools==2.0.11`（装在 conda `yolo` 环境；pip 走清华镜像，直连 pypi 会被代理掐断）。
