# 轻量规则

所有快速规则仅使用最新实测目标、主 ID、类别、置信度、时间戳、框和短轨迹；每次完成执行，不增加神经网络推理或整图光流。目标置信度至少 0.4，最多处理 32 个目标，风险只消费完整处于配置前向通道的目标（比赛 Q4、历史回放 Q1）。基础图像路径是随脚点高度变宽的居中区域，证据中明确写作 ASSUMED_PATH。

| 类型 | 黄色条件 | 红色条件 |
|---|---|---|
| FRONT_COLLISION_RISK | 车辆在假定路径内、脚点较近且框高度/面积达到阈值，至少两次确认 | 短轨迹持续增长、脚点更近、足够框高，至少三次规则确认；静态大框单独不能升级 |
| PEDESTRIAN_IN_PATH | 行人靠近路径及较近脚点 | 实测多帧、处于路径内、脚点与框高达到迫近条件 |
| PEDESTRIAN_CROSSING_PATH | 实测横向速度朝向路径且靠近路径 | 已进入路径，并满足较近脚点及框高 |
| TWO_WHEELER_PATH_CONFLICT | bicycle / motorcycle 靠近路径 | 已进入路径且近距离，同时具有横向进入、接近趋势或非常近脚点；没有虚构电动自行车新类别 |
| VEHICLE_CUT_IN | 车辆从路径外向内横移 | 已进入路径并达到近距离条件 |
| ONCOMING_PATH_CONFLICT | 独立方向输入标记对向，但未确认 | 独立方向确认、持续接近且路径冲突；框增长本身不能证明对向 |
| SUSPECTED_BRAKING_WITH_CLOSING | 最新低频双红灯变化与路径内车辆对应 | 再满足快速层的实测接近及近距离条件；不声称测得车辆减速度 |
| PEDESTRIAN_NEAR_CROSSWALK | 新鲜 MTL 斑马线和实测行人脚点关联 | 本条不直接升级；横穿风险仍由快速实测轨迹产生 |
| CROSSWALK_BRAKE_PULSE | 不用于普通视觉预警 | 仅明确合成测试场景进入 DRY_RUN 点刹；实际推理不自动生成本事件 |

后台最多 2 Hz 复用 `road_geometry.crosswalk_polygons` 和既有 `paired_red_lamps`：读取 MTL 已产生的新掩码引用，最多处理三个车辆裁剪；没有完整 RoadScene、HazardFusion 或额外 NPU 推理。FAST-MOSAIC 没有 MTL 掩码，绝不生成或保留斑马线分割。

对向独立方向来源在当前录像/设备条件下不可用；真实回放不能证明对向识别精度。合成序列分别验证方向未知保持预警、已确认方向发生冲突时升级。真实制动灯/语义结果仅作补充，不能脱离实测路径目标单独发制动请求。

规则开关为 `ADAS_RULE_FRONT`、`ADAS_RULE_PEDESTRIAN`、`ADAS_RULE_TWO_WHEELER`、`ADAS_RULE_CUT_IN`、`ADAS_RULE_ONCOMING`、`ADAS_RULE_BRAKING`、`ADAS_RULE_CROSSWALK`，默认均为 1，设 0 可单独关闭。`risk_rules` 每秒报告启用状态、按目标执行次数、输出观测事件次数及滑动耗时均值/p95；事件次数不等同于唯一事件 ID 数。
