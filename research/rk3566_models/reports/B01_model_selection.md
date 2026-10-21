# B-01：RK3566 ADAS 候选模型选型与部署基准

日期：2026-10-21

研究分支：`research/rk3566-adas-models`

报告基线 commit：`e8a92bb734518edf4a420931f577724a5816b10f`

## A. 执行环境与基线

本轮完成了候选 ONNX 检查、RKNN INT8 离线转换和历史 BDD100K 诊断。**候选模型没有取得本轮 RK3566 实机推理数据**：`192.168.137.2:22` 可达，但 SSH 公钥认证失败。旧记录中的 LubanCat RK3566 环境为 Ubuntu 20.04.6、kernel 4.19.232、4 个 CPU 核、3.9 GB RAM，`rknn-toolkit-lite2/librknnrt 2.3.2`、RKNPU driver 0.8.2；这些版本不是本轮重新连板读取的结果。

| 环境 | 已确认配置 |
|---|---|
| PC | Windows 11 Pro 10.0.26200；指标评估使用 Conda `yolo` Python 3.10.20、PyTorch 2.5.1+cu121、Ultralytics 8.4.68、ONNX Runtime 1.23.2 CPU provider、pycocotools 2.0.11 |
| 离线转换 | WSL Ubuntu；`~/.venvs/adas-rknn310` Python 3.10.20、`rknn-toolkit2 2.3.2`、ONNX 1.16.1、protobuf 3.20.3；目标 `rk3566`，INT8、per-channel/default `normal` |
| 校准 | 三个候选构建使用同来源 25 张 COCO128 train 图（每 5 张取 1 张），RGB、UINT8、静态输入画布；未使用 BDD holdout。校准报告：`C:\Users\JingyuWang\Temp\B01-model-assets\calib_640_report.json`、`calib_512x288_report.json` |
| 质量诊断集 | BDD100K validation 中按文件名排序的前 1000 张；7 类映射与历史协议一致，IoU=0.5，工作点置信度 0.25，COCO `maxDets=100`。该切片曾被多轮分析使用，**只作为历史诊断，不能称为未触碰的独立测试集** |

正式系统基线仍是未微调 COCO `yolo11n.pt`（SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`）生成的 512×288 split INT8 模型（SHA-256 `7a697ccc476954f6f36f7441b842257a2d7a019653cc4bad2fbf8de327c50591`）。当前 Front-only 交通灯专模为 `tl_specialist_512x192_hybrid2.rknn`（SHA-256 `becfaada5da49c4dd6c143e815f8f957fb7b7ed759040d4e18c685a9c434e6f8`），输入为全宽顶部 60% ROI 裁剪后 letterbox 到 512×192。

部署输入约定：先保持宽高比缩放并以 114 padding letterbox；模型输入为 RGB。ONNX 接收 NCHW `float32`、像素除以 255；RKNN 接收 NHWC `uint8`（0–255），转换配置用 mean=0/std=255。YOLO11 split 输出为 boxes `[1,4,3024]` 与 scores `[1,80,3024]`，框按 xywh 解码为 xyxy，再按类别置信度过滤及 class-aware NMS；检测头仍是 80 类 COCO 语义。候选 Model Zoo 图为 640×640 静态输入，每个 stride 有三路输出：YOLOv6 为 `[1,4,H,W]`、`[1,80,H,W]`、`[1,1,H,W]`；YOLOv8 为 `[1,64,H,W]`、`[1,80,H,W]`、`[1,1,H,W]`。YOLOv8 512×288 split 图输出 `[1,4,3024]` 与 `[1,80,3024]`。本轮诊断使用 NMS IoU 0.45；正式 YOLO11 detector 默认也是置信度 0.25 / NMS IoU 0.45。

本轮下载/生成的模型和预测只保存在本机 `Temp`，不纳入 Git：

| 产物 | 来源 / 画布 | 大小 | SHA-256 |
|---|---|---:|---|
| YOLOv6n Model Zoo ONNX | `airockchip/rknn_model_zoo` v2.5.0，640×640 | 18,612,712 B | `fe2eed9e6f3e0f66acfd5a9a3eaf257664208e91df093ec6437eb2a94cd7fd95` |
| YOLOv6n RKNN INT8 | 上述 ONNX，640×640 | 5,750,306 B | `f95f4edc3240d0e57df0eac668fada31775e4808f615ccd98b97d619a5693024` |
| YOLOv8n Model Zoo ONNX | `airockchip/rknn_model_zoo` v2.5.0，640×640 | 12,650,184 B | `0c8716701f471067932b797eeb67c8e5db47c693c2557c881d7679ec12e21bc5` |
| YOLOv8n Model Zoo RKNN INT8 | 上述 ONNX，640×640 | 4,809,995 B | `8cd0c47eaa61bea61bf6d8f2cc6452b2ea0b49a7458e544bf34381d19a94d8a2` |
| YOLOv8n COCO checkpoint | Ultralytics export checkpoint `yolov8n_b01_export.pt` | 6,549,796 B | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| YOLOv8n split ONNX | 上述 checkpoint，512×288 | 12,743,746 B | `d047451aea73de20066e236c952d01b99cd728a5675050b6aea0f3e0d0cff4b1` |
| YOLOv8n split RKNN INT8 | 上述 ONNX，512×288 | 4,140,161 B | `58425dc3c1dc98561482589fc7f19aa40b98ae536feff93affb0fa4e6b948c26` |

Model Zoo 性能表基准固定于 commit `837db8e992815ff8647eadb709f07067f4bc3a71`（v2.5.0）。官方 RK3566/RK3568 合并列的 640×640 INT8 数字为 YOLOv6n 49.9 FPS、YOLOv8n 35.1 FPS、YOLO11n 22.6 FPS、YOLOv5n 41.7 FPS；它们是官方特定优化模型的单模型参考值，不是本项目测量。题面中的 YOLOv6n 48.8 实际对应 Model Zoo 表格里的 YOLOv5n RK3562 列。

## B. 候选可部署性

| 模型 | ONNX | RKNN INT8 离线转换 | RK3566 实机 | 备注 |
|---|---|---|---|---|
| YOLOv6n Model Zoo | 640×640 图已通过 PC CPU 前向与解码 | 成功，产物 5.75 MB | **BLOCKED** | 九输出、stride 8/16/32；本轮没有 512×288 或 640×352 产物，不能声称支持比赛矩形画布 |
| YOLOv8n Model Zoo | 640×640 图已通过 PC CPU 前向与解码 | 成功，产物 4.81 MB | **BLOCKED** | 九输出、包含 DFL 距离分布；性能表口径是 640×640 |
| YOLOv8n Ultralytics split | 512×288 图通过 PC CPU 前向；与 PyTorch 对照近似一致 | 成功，产物 4.14 MB | **BLOCKED** | 两个 split 输出；本轮为最贴近目标部署输入的候选 |
| RTMDet-Tiny | UNKNOWN；本轮未导出 | 未转换 | **BLOCKED** | 未安装 `mmdet`、`mmdeploy`、`mmcv`，Model Zoo v2.5.0 未找到对应 RKNN 示例；按约 90 分钟探索预算停止，未安装大型依赖 |
| YOLO11n 当前基线 | PASS，512×288 静态 ONNX | PASS，RK3566 INT8 | 历史实机 PASS | SHA 与环境见 A；它和交通灯专模是生产参考，不是本轮候选新测数据 |

转换成功只说明 Toolkit 能生成 `.rknn` 文件，不代表目标板能够加载、输出正确或没有算子回退。三款新 RKNN 的加载、NPU 频率/负载、CPU、RSS、输出有限性和推理时间均待实机验证。

## C. 速度对照

| 模型 | 输入尺寸 | NPU ms | 预处理 ms | 后处理 ms | 总延迟 ms | FPS |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv6n Model Zoo | 640×640 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| YOLOv8n Model Zoo | 640×640 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| YOLOv8n Ultralytics split | 512×288 | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN | UNKNOWN |
| YOLO11n 生产基线（历史板测） | 512×288 | 30.15 pipeline NPU；30.42 NPU-only 均值 | 7.74 | 11.82（解码后处理） | 58.51（含 8.80 ms 视频解码） | 17.09 |

生产基线的 NPU-only 40 帧测量为均值 30.420 ms、p50 30.049 ms、p95 32.392 ms；pipeline 使用 600 帧、20 次预热，表中分段可相加。历史监测记录进程 RSS 随输入画布约 110–160 MB；该数值不是候选模型 RSS。2026-10-19 的合成 UI/AI 并行记录（均值 42.08 ms、p50 41.76 ms、p95 46.44 ms、NPU 600 MHz）来自不同计时范围，不能与上述 pipeline 混为一项，也不能当成本轮候选结果。

官方性能表只可用作参考：640×640 时 YOLOv6n 49.9 FPS 高于 YOLOv8n 35.1 FPS。该表未包含项目预处理/解码/NMS，也不能预测不同 512×288 输入的 NPU 延迟。**本项目候选最快者尚 UNKNOWN**。新板测脚本 `deployment/rknn/benchmark_candidate.py` 支持板端 NPU-only 预热与 200 次计时、资源快照；YOLOv8 split 路径另有完整预处理 / 解码 / NMS / 端到端测量。候选数据尚未上传到板端执行。

## D. 道路识别质量

以下都是 PC FP32 路径的历史诊断，按同一 BDD100K 1000 图切片、7 类映射、置信度 0.25 计算；YOLOv6n/YOLOv8n 是 ONNX Runtime CPU，YOLO11n 基线为 PyTorch CPU。不是 RKNN INT8 指标，不是道路视频级 sealed test。COCO 预训练权重只有通用 `traffic light` 类，没有红/黄/绿状态或斑马线类别。

| 模型 / 后端 | 输入 | AP50 | AP50:95 | car Recall | person Recall | traffic light Recall |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv6n Model Zoo ONNX FP32 CPU | 640×640 | 0.2396 | 0.1411 | 0.5092 | 0.3825 | 0.1397 |
| YOLOv8n Model Zoo ONNX FP32 CPU | 640×640 | **0.2555** | **0.1463** | 0.3925 | 0.2857 | 0.0827 |
| YOLOv8n Ultralytics ONNX FP32 CPU | 512×288 | 0.2034 | 0.1217 | **0.3435** | 0.2029 | **0.0541** |
| YOLO11n 基线 PyTorch FP32 CPU | 512×288 | 0.1999 | 0.1184 | 0.3367 | **0.2156** | 0.0516 |

640×640 同尺寸下，YOLOv8n 的 mAP 较高，YOLOv6n 在固定置信度下的 car/person/traffic-light Recall 与 macro recall（0.3018 对 0.2159）较高；二者置信度排序表现不同，不能只挑单一指标宣布赢家。512×288 同尺寸下，YOLOv8n 的 AP 提升很小（AP50 +0.0035、AP50:95 +0.0034），Recall 呈混合变化，macro recall 0.1673 低于 YOLO11n 的 0.1795；当前证据不能说明它在整体道路质量上超过基线。YOLOv6n 640 与目标矩形输入的 YOLO11n 512 不是同像素预算比较。

小目标 Recall（COCO area `<32²`，同一固定置信度匹配）分别为：YOLOv6n 640 0.1358、YOLOv8n 640 0.0563、YOLOv8n 512 0.0266、YOLO11n 512 0.0235。交通灯这些数字只代表框召回，不代表灯色状态准确率。所有模型在行人、两轮车和交通灯上仍有明显漏检；BDD 历史切片不足以验证远距离目标的正式胜负。

## E. INT8 量化损失

- YOLOv8n Ultralytics split 的 PyTorch FP32 与 ONNX FP32 输出已在同类框间按 IoU≥0.95 对照：匹配 96,326 个框，平均 IoU 0.9991，支持“该导出路径差异很小”；这是框匹配诊断，不是完整的量化质量验收。
- YOLOv6n 和 YOLOv8n Model Zoo 图的上游 PyTorch 逐框对照没有完成，FP32→ONNX 误差未知。
- ONNX→RKNN INT8 在目标 RK3566 上没有候选输出，因此各候选的置信度漂移、坐标漂移、类别失效和小交通灯 Recall 损失全部为 **UNKNOWN / BLOCKED**。不能据离线 build 成功推定量化无损。
- 候选校准只用了 25 张 COCO128 通用图，未覆盖专门采集的道路分布或颜色状态标签。已有项目实验证明交通灯框对约 2 px 的 INT8 量化步长敏感：当前 YOLO11n 主模型曾出现框匹配率/平均 IoU退化，traffic-light specialist 普通 INT8 AP50 从 0.101 降至 0.029；混合量化把 25 层框链保留 FP16 后，板端 NPU 均值为 25.29 ms。此历史事实说明候选即使 PC FP32 表现较好，也必须做板端量化审计；本轮没有为新候选调混合量化。

## F. 综合判断

1. **本项目实机最快模型：UNKNOWN。** 官方 640×640 参考吞吐最高的是 YOLOv6n，但不能替代本机数据。
2. **标准 mAP 最好的候选：YOLOv8n Model Zoo 640×640 ONNX FP32。** AP50 与 AP50:95 领先，但 fixed-conf recall 不及 YOLOv6n，且不是目标矩形输入。
3. **固定工作点 Recall 最好的候选：YOLOv6n Model Zoo 640×640 ONNX FP32。** 需要验证矩形导出与 RK3566 延迟后才可评价工程收益。
4. **精度/速度综合最优：UNKNOWN。** 新候选没有 RK3566 延迟和 INT8 质量；当前 YOLO11n 512×288 是唯一同画布且已有真实系统基准的参照。
5. **下一轮优先小规模微调验证：YOLOv8n 512×288。** 它具备成熟 Ultralytics 训练流程和当前目标画布的静态 split 导出；本轮 AP 仅小幅领先 YOLO11n，应该用相同训练集/种子做试验，而非视为胜出。
6. **第二候选：YOLOv6n。** 它有较好的官方吞吐参考和 640×640 质量诊断，但先决条件是构建并验证 512×288 或 640×352 静态 RKNN 图。若矩形转换或板端表现不佳，应停止后续投入。
7. **RTMDet-Tiny 暂不值得继续本阶段投入。** 这是工程可行性/依赖成本判断，不是对其精度或架构的否定；缺少 ONNX/RKNN/实机证据。

**结论：没有证据证明新候选优于现有 YOLO11n + traffic-light specialist，也没有依据替换生产模型。** 本轮没有运行训练或付费 GPU。COCO 80 类检测头在架构上可以重配车辆、行人、两轮车、灯色与斑马线类别，但这些目标需要许可明确、按视频分组且对目标类别完整标注的数据；架构可扩展不等于当前已识别这些语义。

## G. 下一阶段建议

1. 恢复 LubanCat SSH 公钥访问后，先在同一 RK3566、runtime、输入样本和空闲系统状态下测候选 NPU-only；640×640 的 v6/v8 走官方九输出 decode，512×288 v8 split 走两输出 decoder。记录 mean/p50/p95、输入输出形状、RSS、CPU、NPU load/frequency，并核对是否有 CPU fallback。
2. 对 512×288 v8 split 做完整 pipeline；若保留 YOLOv6n，先构建同为 stride 合法的矩形输入图并重复 ONNX/INT8 质量检查。NPU-only FPS 不可充当完整帧率。
3. 用同一批非 holdout 道路校准图对比 FP32、ONNX、RKNN INT8 输出，重点统计小交通灯框、行人与远车；必要时再决定是否值得混合量化。
4. 建立按源视频/采集行程隔离的新 sealed test，并审计 11 个目标类别的完整标注覆盖后，再用完全相同的数据、画布与随机种子进行最多两款模型的 300–1000 张小规模训练试验。只有板端延迟、量化后质量和独立集结果均达到门槛，才另行申请更大训练预算。

当前 Git 不包含模型、校准图、BDD 预测或私人道路素材。此报告的指标 JSON、诊断 Markdown 和 RKNN 文件保存在本机 `C:\Users\JingyuWang\Temp\B01-model-assets`，可用上表的哈希核验。
