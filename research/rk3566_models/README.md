# 支线 B：RK3566 ADAS 专用模型研发

状态：**S0 已建立研究基线；尚未训练、未完成新候选板端测试。**
独立于 Qt/Tesla/PCIe 主线；禁止直接覆盖生产权重或修改正式部署分支。

## 研究问题与先验

目标不是证明“非 YOLO 必然优越”，而是在同一台 RK3566（1 TOPS）上验证**真实道路识别质量、INT8 量化损失、NPU/CPU 端到端延迟**的 Pareto 最优解。候选：YOLOv6n、YOLOv8n、RTMDet-Tiny；现有 YOLO11n + Traffic-Light Specialist 为**系统基线**。YOLOv5n只作纯速度补充。

已有研究结果：
- `docs/road_domain_pilot_results.md`：nuImages 领域微调后，冻结的1000张BDD图上 traffic light TP从141降至0；该数据源无traffic light标注，保持COCO头并不能防止共享特征漂移。
- `docs/competition_tl_specialist.md`：交通灯小框对INT8框回归量化敏感，混合量化才得到可用板端结果。单看PC FP32不够。
- `docs/competition_detection_metric_protocol.md`：项目历史macro recall@conf=.25不是COCO AP。不得混淆。固定输入需合法且保留精确letterbox。
- 历史1000图BDD holdout已被多轮调参/方法选择反复查看，**只能做历史可比诊断，不能再宣称是完全未触碰的最终测试集**。正式胜负判定必须新建封存测试集。

参考源：
- Rockchip 官方性能表 https://github.com/airockchip/rknn_model_zoo （官方640方画布INT8、仅模型推理：v6n 48.8 FPS、v8n 34.0 FPS、yolo11n 20.6 FPS；**非本项目实测**）
- Rockchip yolov6 example https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov6
- Rockchip yolov8 example https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov8
- RTMDet https://github.com/open-mmlab/mmdetection （转换/实测仍待验证）

## S1：不租 GPU，先审查 RKNN 可部署性

每个候选使用相同RKNN toolkit/runtime、相同线程和系统状态、相同图像、相同静态输入尺寸；先固定 **512×288**，可扩展 **640×352**（两维均为32倍数）。确保NPU算子没有静默CPU回退，记录原版与Rockchip优化版ONNX的导出方式/输出张量形状，不能拿不同量化/导出路径的官方FPS当本机数据。

记录：NPU-only ms p50/p95，预处理、推理、解码、NMS、总帧时延，CPU/NPU占用、RSS，模型版本、校准来源、分辨率、运行时版本；完整UI中的“不同源帧上传FPS”由主线A独立测量。板端不可访问时本步骤标记BLOCKED，不使用PC FPS代替。

## S2：数据获取与标注

以BDD100K train为优先道路训练来源；DTLD/Bosch补充小交通灯，Mapillary/A2D2等视标签格式与许可再纳入。比赛官方视频只有**规则明确允许训练**才可进入train，测试视频不可偷用于训练。

先按源视频/行程ID分train、val、sealed-test，禁止相邻帧跨split。训练类别初拟11个：
`person,bicycle,motorcycle,car,bus,truck,red_light,yellow_light,green_light,traffic_light_other,crosswalk`。
交通灯颜色来自可验证的状态标签；未知类不能把所有未标注目标当负例。斑马线来源通常是多边形/分割标注，转换为bbox只能用于“存在检测”，不能伪称道路几何分割。

**不得把缺少目标类别的图像当成“对所有11类都已完整标注”。** 混合部分标注数据必须先补齐对应实例，或明确实现并验证class-aware ignore loss；普通YOLO训练器不能假定支持按类别忽略。先运行 `python tools/audit_adas_dataset_manifest.py --manifest path/to/manifest.csv`，它仅检查元数据声明和视频分组泄漏，不能代替人工验标。

## S3：训练与模型比较

- V0：现有 YOLO11n + Specialist（保持不动）。
- V1：YOLOv6n 微调，保留6类交通参与者 + 正确带色状态的交通灯/斑马线标签。
- V2：YOLOv8n 用完全相同的数据/类别/随机种子公平对照。
- V3（预算许可才进行）：RTMDet-Tiny，优先验证导出是否真的能在RK3566运行。

4090仅在S1转换/基准和S2标注覆盖满足条件后再使用。首先进行300–1000张精选样本试验；不承诺直接大规模训练。保留COCO通用能力评估，防止灾难性遗忘。独立验证至少比较两种随机种子，关键差异不足时做第三种子。

## S4：量化与合并门槛

1. 类别逐项报告 AP50、AP50:95、Recall@固定conf，以及红黄绿混淆表、小目标按尺寸分层Recall。
2. 同一张量前/后处理对比PC FP32与RK3566 RKNN INT8；交通灯框回归单独检查量化退化。
3. 新模型只有在 **车辆/行人表现不出现不可接受的回退，交通灯状态准确率和召回不劣于当前两模型，端侧资源成本确实减少** 时才允许提出合并。
4. 如果一个统一检测器对远处交通灯依然差，保留Specialist是合格结果。绝不能为了让模型数量变少而牺牲红灯召回。
5. 不输出未经标定的距离、TTC或虚构导航/车道属性。

## 分工与当前阻塞

本分支只包含研究计划与数据审计工具；真实模型、视频、标注、标定集和RKNN模型均不提交到公开GitHub（文件体积和许可约束）。当前桌面设备离线，无法直接操作本机/鲁班猫，因此S1板端测试和S2素材核查仍需设备可访问后执行。

下次按序执行：先获取板端环境和目前的512×288导出设置 → 下载官方优化过的v6n/v8n权重并跑同条件模型基准 → 编制新独立sealed测试样本和类别覆盖审计 → 决定4090训练。
