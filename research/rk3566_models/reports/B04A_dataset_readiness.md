# B-04A：比赛场景视频清点与训练前准备（启动记录）

**状态：IN_PROGRESS / 数据未就绪。** 本轮创建视频清点工具及测试，不宣称已读取本机视频、抽帧、标注或训练。研究分支外正式模型和RK3566不变。

## 为什么先清点
B-03E已证实BDD YOLOv8n KL在RK3566约26.109 FPS，但在反复筛选的50张BDD100K历史图上，交通灯Recall@0.25只有11.11%。下轮优先测试合法比赛开发/自采道路视频上的场景适配收益。

B-03A审计发现本机9000张BDD validation衍生交通灯ROI单类图，另有冻结验证名单的100张全帧图。这些不能直接当作完整十类训练集。暂无经核实的完整比赛训练图片和标注。

## 数据使用硬规则

1. 正式隐藏测试视频、竞赛禁止训练的视频不可用于训练、校准或阈值选择。
2. 团队自采/明确允许训练的比赛开发视频，保留来源、许可依据、采集场次、摄像头及视频ID。
3. 按原视频/采集场次划分train、val、sealed-test，禁止相邻帧跨集合。历史冻结BDD数据不可回流训练。
4. BDD YOLOv8n保留十类头：person/rider/car/truck/bus/train/motor/bike/traffic light/traffic sign。普通YOLO训练器要求完整标注；不能把交通灯或交通标志未标注当背景。若未来改为六类参与者检测头，须单独审计并继续保留Specialist。
5. 本阶段仅生成抽帧时间戳提案，不生成训练数据；后续需核实视频许可、人工校正标签完整性。

## 已添加的工具

- tools/b04a_video_inventory.py：只检查显式CSV登记的视频；按最小间隔5秒生成时间戳，默认每视频最多150张、总800条；轮询分配候选；不实际抽帧、不上传视频。
- tests/test_b04a_video_inventory.py：覆盖禁止隐藏测试、未知许可排除、视频级划分泄漏与稀疏抽帧。
- 不提交原始视频、图片、标注缓存、训练权重。

## 本机登记表格式

CSV列名：

video_path,source_id,source_kind,permission,split,license_note

source_kind取值：team_recorded、competition_development、official_test、third_party。

permission取值：allowed、unknown、forbidden。必须有真实授权依据，未知不可填allowed。

split取值：train、val、sealed-test、unassigned。source_id基于真实采集场次/序列而非不可靠的文件名前缀；同一source_id不得跨split。

在本机运行：

    python tools/b04a_video_inventory.py --registry D:/Datasets/adas_edge_training/B04A_sources.csv --output-dir D:/Datasets/adas_edge_training/b04a_inventory --gap-seconds 5 --max-total 800

输出B04A_inventory.json和B04A_frame_proposals.csv。所有提案均为NOT_ANNOTATED / training_ready=false；程序PARTIAL不意味着训练数据就绪。

## 当前阻塞与下步

本次ChatGPT连接的Windows Desktop Commander设备离线，未能枚举视频源与标签。本机实际可训练视频数、帧数和许可、标注覆盖仍为未知。

让本机Codex首先登记真实视频素材和规则许可，运行上述脚本，报告可授权提案数量、无法训练的来源、标签缺口和按视频的train/val/sealed-test划分。

确认来源许可后，再筛选300–800张具有代表性的比赛场景帧，覆盖光照、天气、拥挤程度和小目标。先自动预标注再人工修订十类完整目标框；审核通过后才进入B-04B短程微调，不在本阶段启动训练。
