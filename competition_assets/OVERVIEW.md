# ADAS Edge 比赛素材总览

采集整理：2026-11-02（Asia/Shanghai）。基于当前实际运行的 RK3566 C++/Qt + PCIe 版本，来源、状态及 SHA-256 随库提供。

原始候选 **361 张**，精品 **54 张**，核心推荐 **21 张**。另有 5 张合成 UI 参考，隔离保存，不计入推荐。

## 快速使用

双击 index.html 可按核心/精品/全部候选、主题和关键词筛选，每张图可打开原图及裁剪版本。contact_sheets 为 PNG 缩略图总览。

## 六类主题

| 主题 | 精品 | 推荐位置 |
|---|---:|---|
| [完整车机与系统总览](curated/01_system_overview/README.md) | 8 | 项目书封面、系统介绍、答辩开场 |
| [MTL 多任务分割与道路语义](curated/02_multitask_segmentation/README.md) | 10 | 多任务感知、检测与分割联合输出 |
| [FAST-Q4、FAST-MOSAIC 与目标检测](curated/03_multicamera_detection/README.md) | 12 | 单路/整图一次推理、模型方案对比 |
| [Tesla BEV 与空间感知](curated/04_tesla_bev/README.md) | 8 | 跟踪结果与相对空间可视化 |
| [真实危险预警与风险可视化](curated/05_hazard_warning/README.md) | 8 | 危险感知闭环、真实告警案例 |
| [FPGA 控制、模型菜单与性能证据](curated/06_control_models_performance/README.md) | 8 | FPGA 反馈、模型切换、软硬件联动 |

## 核心图片推荐

| 编号 | 图片 | 推荐位置 | 采集前日志摘要（异步） |
|---|---|---|---|
| 01 | [live_0068](core/01_system_overview_live_0068.png) | 项目书封面、系统介绍、答辩开场 | MTL-FAST；目标 12；事件 1 |
| 02 | [live_0060](core/02_system_overview_live_0060.png) | 项目书封面、系统介绍、答辩开场 | MTL-FAST；目标 11；事件 1 |
| 03 | [live_0009](core/03_system_overview_live_0009.png) | 项目书封面、系统介绍、答辩开场 | MTL-FAST；目标 12；事件 1 |
| 04 | [live_0061](core/04_multitask_segmentation_live_0061.png) | 多任务感知、检测与分割联合输出 | MTL-FAST；目标 7；事件 0；硬件刹车允许 |
| 05 | [live_0069](core/05_multitask_segmentation_live_0069.png) | 多任务感知、检测与分割联合输出 | MTL-FAST；目标 10；事件 3 |
| 06 | [live_0075](core/06_multitask_segmentation_live_0075.png) | 多任务感知、检测与分割联合输出 | MTL-FAST；目标 15；事件 0 |
| 07 | [live_0021](core/07_multitask_segmentation_live_0021.png) | 多任务感知、检测与分割联合输出 | MTL-FAST；目标 9；事件 3 |
| 08 | [live_0163](core/08_multicamera_detection_live_0163.png) | 单路/整图一次推理、模型方案对比 | FAST-MOSAIC；目标 16；事件 0 |
| 09 | [live_0121](core/09_multicamera_detection_live_0121.png) | 单路/整图一次推理、模型方案对比 | FAST-Q4；目标 13；事件 1 |
| 10 | [live_0228](core/10_multicamera_detection_live_0228.png) | 单路/整图一次推理、模型方案对比 | FAST-MOSAIC-LITE；目标 18；事件 0 |
| 11 | [live_0203](core/11_multicamera_detection_live_0203.png) | 单路/整图一次推理、模型方案对比 | FAST-MOSAIC；目标 9；事件 2 |
| 12 | [live_0252](core/12_tesla_bev_live_0252.png) | 跟踪结果与相对空间可视化 | MTL-FAST；目标 8；事件 2；硬件刹车允许 |
| 13 | [live_0250](core/13_tesla_bev_live_0250.png) | 跟踪结果与相对空间可视化 | MTL-FAST；目标 9；事件 0 |
| 14 | [live_0269](core/14_tesla_bev_live_0269.png) | 跟踪结果与相对空间可视化 | MTL-FAST；目标 14；事件 0 |
| 15 | [live_0299](core/15_hazard_warning_live_0299.png) | 危险感知闭环、真实告警案例 | FAST-Q4；目标 7；事件 2；硬件刹车允许 |
| 16 | [live_0302](core/16_hazard_warning_live_0302.png) | 危险感知闭环、真实告警案例 | FAST-Q4；目标 9；事件 1；硬件刹车允许 |
| 17 | [live_0094](core/17_hazard_warning_live_0094.png) | 危险感知闭环、真实告警案例 | MTL-FAST；目标 6；事件 2；硬件刹车允许 |
| 18 | [live_0001](core/18_control_models_performance_live_0001.png) | FPGA 反馈、模型切换、软硬件联动 | MTL-FAST；目标 2；事件 0；硬件刹车允许 |
| 19 | [live_0294](core/19_control_models_performance_live_0294.png) | FPGA 反馈、模型切换、软硬件联动 | FAST-Q4；目标 9；事件 2；硬件刹车允许 |
| 20 | [live_0078](core/20_control_models_performance_live_0078.png) | FPGA 反馈、模型切换、软硬件联动 | MTL-FAST；目标 9；事件 1；硬件刹车允许 |
| 21 | [live_0275](core/21_control_models_performance_live_0275.png) | FPGA 反馈、模型切换、软硬件联动 | MTL-FAST；目标 13；事件 0 |

## 推荐展示叙事

1. 系统总览：完整车机实际运行。
2. MTL：车辆、行人与道路语义联合输出。
3. FAST-Q4 / FAST-MOSAIC：单路与整图一次推理；Lite 要明确说明纯检测负载。
4. 真实预警与 Tesla BEV：从检测、跟踪到风险可视化。
5. FPGA 控制：用受控探测截图和 JSON 说明请求、ACK、允许反馈及结束闭环。
6. 模型菜单与性能：显示实际可切换方案；正式性能引用历史独立验收，不能用截图时瞬时 FPS 替代。

## 真实性边界

- 精品全部为本次 REAL_PCIE_LIVE，未添加或重绘框、分割、FPS、ACK 或控制状态。
- Q1 非黑但短采样未变化，未证明四个独立动态摄像头验活，也未定位旧上游问题根因。
- 受控探测为人工授权单次请求，不伪装成 AI 自动触发。FPGA 接收/允许不等于机械制动验收。
- BEV 是相对空间示意。原图页脚为板端时钟，日期与电脑不一致；采集日期以清单电脑端时间为准，关联采用单调时间。
- 旧回放、历史实测和合成演示独立标注，未混作本次硬件证据。

## 目录

- raw_candidates：全部原始候选 PNG。
- curated：六主题精品，提供 originals、ppt_16x9、project_book、focus_16x9。
- core：可直接使用的核心全图。
- manifest.json / catalogue.csv：说明、模式、推荐位置、筛选和 SHA-256。
- evidence：状态、硬件联动、象限数据、历史报告和版本来源。
- tools：可复用采集与整理脚本。
