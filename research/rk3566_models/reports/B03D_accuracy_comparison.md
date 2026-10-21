# B-03D RK3566 双模型真实精度对比

**状态：PARTIAL（可复现筛选结果，官方 `det_val.json` 直取仍受网络阻塞）**

本轮严格使用 B-03C-R 的 `holdout100` 前 50 张文件清单，没有混用 B-02 的前 50 张。板端两款模型都在同一批图上以 `conf=0.001`、NMS IoU `0.45`、`max_det=100` 各推理一次，AP 在 PC 端计算；固定工作点指标使用 `conf=0.25`、匹配 IoU `0.5`。

## GT 与匹配

官方 BDD 文档确认 Detection 2020 标签应为 `det_20/det_val.json`（约 53 MB，MD5 `b86a3e1b7edbcad421b7dad2b3987c94`）。本机访问官方 `dl.cv.ethz.ch` 时 DNS/SSL 均失败，因此没有把 2018 标签冒充官方 JSON。实际使用的可审计镜像是 [Feng1909/bdd100k_voc_labels](https://github.com/Feng1909/bdd100k_voc_labels)，仓库说明为 BDD100K 2020 detection labels 的 VOC 转换，抓取时 main 提交为 `1a1bb707b60955a5646329ea1fa9e5a81dca0279`。50 个准确文件名全部匹配，图片尺寸均为 `1280x720`；XML 不保留交通灯颜色属性。

| GT 类别（七类） | 框数 | 含该类图片数 |
| --- | ---: | ---: |
| car | 493 | 50 |
| truck | 13 | 9 |
| bus | 6 | 6 |
| person（`pedestrian` 映射） | 34 | 13 |
| motorcycle（`motor`/`motorcycle`） | 4 | 4 |
| bicycle（`bike`/`bicycle`） | 2 | 2 |
| traffic light | 153 | 29 |
| **合计** | **705** | - |

`rider`（4）和 `traffic sign`（175）保持独立并排除，未并入 `person`。同一批图的 2018 HF JSON 交叉核对为 704 个七类框（car 492、truck 12、traffic light 154），旧 B-02 CSV 的 753 个总数也不是本轮 GT；这解释了历史数字差异。本轮 2020 VOC 镜像没有 `trafficLightColor`，但 2018 JSON 交叉源确实含该字段（green 81、red 33、none 37、yellow 3），这些颜色统计不进入 AP。

## 核心结果

| 指标 | YOLO11n RKNN | BDD YOLOv8n RKNN INT8 |
| --- | ---: | ---: |
| AP50 | 14.37% | **23.34%** |
| AP50:95 | 6.74% | **11.79%** |
| Precision @0.25 | **76.47%** | 60.87% |
| Recall @0.25 | 23.97% | **43.69%** |
| F1 @0.25 | 36.50% | **50.87%** |
| Car Recall | 32.05% | **53.75%** |
| Person Recall | 8.82% | **20.59%** |
| Traffic Light Recall | 1.96% | **18.30%** |
| 总误检 FP | **52** | 198 |
| Pipeline FPS（B-03C-R 历史实测） | 21.616 | **26.317** |

### 分类别工作点

| 模型 / 类别 | TP | FP | FN | AP50 | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11n / car | 158 | 36 | 335 | 38.79% | 32.05% | 46.00% |
| YOLO11n / person | 3 | 0 | 31 | 22.35% | 8.82% | 16.22% |
| YOLO11n / traffic light | 3 | 7 | 150 | 1.92% | 1.96% | 3.68% |
| BDD YOLOv8n INT8 / car | 265 | 113 | 228 | 55.39% | 53.75% | 60.85% |
| BDD YOLOv8n INT8 / person | 7 | 10 | 27 | 22.37% | 20.59% | 27.45% |
| BDD YOLOv8n INT8 / traffic light | 28 | 68 | 125 | 10.17% | 18.30% | 22.49% |

BDD YOLOv8n INT8 相对正式 YOLO11n 的 AP50 提高 8.97 个百分点，Recall 提高 19.72 个百分点，F1 提高 14.37 个百分点；交通灯 Recall 提高 16.34 个百分点，但交通灯 FP 增加 61 个，总 FP 增加 146 个。它找回了更多有效目标，同时显著牺牲了精度。

## FP32 与 INT8

同一 50 张图、相同类别映射和 `conf=0.25` 下，BDD YOLOv8n FP32 参考与板端 INT8 的结果如下。FP32 参考来自 B-03C-R RTX 4060 缓存，不能和板端速度混用。

| 指标 | BDD YOLOv8n FP32 | BDD YOLOv8n INT8 | INT8 - FP32 |
| --- | ---: | ---: | ---: |
| AP50 | 29.59% | 23.34% | -6.25 pp |
| AP50:95 | 18.34% | 11.79% | -6.56 pp |
| Precision | 79.61% | 60.87% | -18.74 pp |
| Recall | 52.62% | 43.69% | -8.94 pp |
| F1 | 63.36% | 50.87% | -12.50 pp |
| Car Recall | 58.62% | 53.75% | -4.87 pp |
| Person Recall | 38.24% | 20.59% | -17.65 pp |
| Traffic Light Recall | 39.87% | 18.30% | -21.57 pp |
| 总 FP | 95 | 198 | +103 |

结论是 INT8 有明显量化退化，交通灯和行人最明显；但在这个小样本筛选集上，INT8 仍比正式 YOLO11n 的 AP50、三类 Recall 和 F1 高。这里的 FP 是相对 GT 的误检，不是 FP32/INT8 输出差异；后一差异已单独核算。

## 板端条件与结论

- SSH `lubancat` 已恢复，公钥认证成功；测试期间临时停止 bridge 以降低 NPU 竞争，测试后服务进程已重新出现。正式模型未覆盖，SHA256 仍为 `7a697ccc476954f6f36f7441b842257a2d7a019653cc4bad2fbf8de327c50591`。
- BDD YOLOv8n INT8 SHA256 为 `BDFAA64782EA1EBBDD8F23F18ABD91567C193A0C44C02D1B94D08661A227BD82`。两款输入均为 RGB `UINT8 [1,288,512,3]`；正式模型输出 `[1,4,3024]+[1,80,3024]`，候选输出 `[1,4,3024]+[1,10,3024]`，均为有限值。
- RKNN Lite `2.3.2`，运行库 `2.3.2`，驱动 `0.8.2`。Pipeline FPS 使用 B-03C-R 的独占短测历史值；本轮低阈值推理只为 AP，不重复声称新的速度 Benchmark。

明确回答：

1. **BDD YOLOv8n 在本轮 2020 标签镜像 GT 上更准确**，但以更多 FP 换取 Recall；这不是独立新测试集的最终排名。
2. **交通灯 Recall 相对正式模型 +16.34 pp，交通灯 FP +61**；INT8 相对自身 FP32 的 Recall -21.57 pp。
3. **INT8 有明显精度损害**，尤其是交通灯和行人，但仍保留相对 YOLO11n 的收益。
4. **不建议直接替换正式主检测器**。候选 Pipeline 历史 FPS 约快 21.8%，但当前量化退化和误检代价过大，先作为候选保留，需在更多未参与模型选择的带 GT 数据上复核。
5. **继续保留 Traffic-Light Specialist**；本轮没有验证灯色状态，也没有修改 Specialist。

评测脚本：`tools/evaluate_b03d_accuracy.py`；单元测试覆盖 VOC/JSON 标签解析和交通灯属性读取。原图、XML/JSON 标注缓存、模型权重和板端预测均未提交 Git。
