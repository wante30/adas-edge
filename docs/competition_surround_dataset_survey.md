# 2026 FPGA 智慧驾驶比赛：四路环视数据集路线核验

调查日期：2026-09-18。范围严格限定为：公开资料核验、少量合法下载、样本检查、四路预览与 PC 端一次性几何可行性探针。本轮没有训练模型，没有修改 `CompetitionDetector`、调度器、论文实验或 FPGA 实现。

## 结论先行

- **PRIMARY SURROUND DATASET = Audi A2D2。** 它是本轮唯一同时满足“匿名逐文件下载可用、同车连续多相机、前/后/左/右方向可取、时间戳与内外参实测存在、总下载可控制在 2 GB 内”的真实道路数据。四路 10 秒样本已生成并验证。
- **SECONDARY DATASET = Argoverse 2 Sensor Dataset。** 七路环形相机覆盖更完整、20 fps、单个 log 可按相机逐文件取；适合多路吞吐和未来完整 360 研究。但它没有单独的正后相机，实际曝光也不是七路同一纳秒，不能把选四路说成完整 360。
- **TRAFFIC LIGHT DATASET = DriveU Traffic Light Dataset（DTLD）。** 它比 BSTLD 更适合当前“小目标交通灯 Gate FAIL”问题：超过 23 万个交通灯、2 MP、15 Hz、状态/图案/相关性/方向/track ID；但需要人工登记并接受仅限自身研究和教学、禁止商业使用和向第三方披露的条款。
- **FPGA AVM 几何开发的当前数据源 = A2D2 全六路，而不是当前四路子集。** 实测四路地面投影覆盖率约 79.08%，存在明显前对角盲区；六路约 91.43%。A2D2 能验证标定解析和地面映射，但不是“正好四个低位 180° 鱼眼”的最终硬件等价物。
- **正好四路、最接近 AVM 的条件候选 = AVM-SLAM 或 Surround-View Calibration Benchmark。** 前者有四路鱼眼和预生成 BEV，但下载走百度网盘，且公开页未确认标定文件是否随包提供；后者有四鱼眼、视频与外参真值，但未找到明确数据许可证。两者在完成下载包和授权核验前都不能替代主路线。
- **WoodScape 不作为主选。** 原始采集确为四个 190° 鱼眼，但公开版为了帧多样性并没有把四个相机在同一时刻全部标注/打包；公开清单表现为每个样本 ID 只有一个相机方向和其前一帧，不是同步四路连续视频。数据许可证为 Proprietary，比赛演示权也未确认。

## “360”术语边界

| 名称 | 本报告定义 | 本轮是否完成 |
|---|---|---|
| A. 2×2 mosaic | 四个独立画面按 `FRONT | RIGHT / LEFT | BACK` 排列，不做几何融合 | YES |
| B. panorama | 柱面、球面或 equirectangular 横向 360° 展开，需要完整视场、相机模型和重叠区 | NO |
| C. AVM / BEV | 以车辆为中心，将多相机地面区域映射到鸟瞰平面并处理接缝 | 只做了 PC 几何探针；不是无缝 AVM |

比赛目标优先级是 C。A 只能用于通道、同步和吞吐展示，不能命名为“360 无缝融合”。

## 最终推荐矩阵

`YES*` 表示不是原生 2D 检测框，而是可由 3D 框投影或由语义掩码转换；这种转换不等同于官方 2D GT。

| Dataset | 4路/多路同步 | 360覆盖 | Calibration | 2D GT | TL GT | Continuous | Download | License | Fusion | Train | Demo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A2D2** | 6 路；四路实测约 3 ms 偏差 | 6 路 YES；所选四路 NO | 内参、外参、畸变、时间延迟 | NO；前视 3D/语义可转换* | 语义 traffic signal；无灯色状态框 | YES，约 30 Hz 原始；本轮取 10 Hz | **匿名 S3，逐文件，已验证** | CC BY-ND 4.0 | 6 路 GOOD；四路 POOR | 语义 GOOD；通用 2D 检测 POOR | 四路回放 EXCELLENT；公开衍生视频需谨慎 |
| **Argoverse 2 Sensor** | 7 ring，20 fps；与 LiDAR 触发协调 | 7 路 YES；任选四路 NO | 内参、外参、畸变、pose、timestamp | 3D cuboid 投影*；非原生 2D | NO；只有 `TRAFFIC_LIGHT_TRAILER` | YES，15 s | **匿名 S3，可按 log/相机，已验证** | CC BY-NC-SA 4.0 | 7 路 GOOD；近车 AVM POOR | 3D GOOD；2D detector FAIR | 多路压力 EXCELLENT；四路 GOOD |
| nuScenes | 6 路；关键帧约 2 Hz 对齐，raw sweep 12 Hz | 6 路 YES；FRONT/BACK/FL/FR NO | 内参、外参、timestamp、pose | 3D box 投影*；非原生 2D | NO（地图层有交通灯，不是图像检测 GT） | 20 s scenes | 登录/接受条款；mini 约 4 GB | CC BY-NC-SA 4.0 | 6 路 GOOD；四路 POOR | 3D GOOD；2D FAIR | GOOD，预处理较重 |
| WoodScape | 采集硬件 4 路；**公开版不是同步四路序列** | 采集 rig YES；公开包不能重放 | 标定可用 | YES，5 类 | YES，bbox 类之一；无灯色状态 | 只有 `t` 与 `t-1` 单方向帧 | Google Drive；数据未完成匿名取样 | Proprietary | **公开包 NOT SUITABLE** | 鱼眼 2D GOOD | 四路 NOT SUITABLE |
| DDAD | 6 路同步，10 Hz | YES | 内参、外参 | NO，深度为主 | NO | 5/10 s | 257 GB 整包；有第三方 tiny 路线 | CC BY-NC-SA 4.0 | GOOD | detector POOR | GOOD，但下载形态不合适 |
| PandaSet | 6 路，10 Hz | 6 路 YES；四路有缺口 | 内参、逐帧 pose/timestamp | 3D cuboid；非原生 2D | NO | 8 s scenes | 官方链接本轮 404；HF 镜像 44.5 GB 整包 | CC BY 4.0 | 6 路 GOOD；近车 AVM POOR | 3D GOOD | 技术 GOOD，当前下载 POOR |
| Waymo Perception | 5 路，10 Hz | **无后视相机，不是相机 360** | YES | YES：vehicle/person/cyclist/sign | 标准 2D NO；panoptic 有 TL 类 | 20 s | Google 登录、条款、云端大数据 | Waymo Dataset License，非商业 | POOR | GOOD | 无 BACK，POOR |
| Waymo E2E Driving | 8 路 surround | YES | YES/以当前开发包为准 | 规划数据，不是本任务的检测 GT | NO/UNKNOWN | 20 s segments | 登录与条款；大型云数据 | Waymo Dataset License | GOOD | POOR | GOOD，但访问重 |
| KITTI-360 | 2 perspective + 2 fisheye | 全 rig 可 360，但不是前左后右四等向 | 完整内外参、timestamp | 语义/3D，不是通用 2D detector | 语义 traffic light | 11 连续序列 | 注册；服务器近期恢复过 | 非商业研究 | panorama GOOD；近车 AVM FAIR | FAIR | FAIR |
| Boden AVM | 公开数据主要是**已融合 AVM 图** | 已融合 BEV | 原始 rig 参数描述；公开 raw/calib 未确认 | parking/freespace/semantic | NO | 图像集，连续性 UNKNOWN | 2k sample + 18k 云端；本轮未下载 | CC BY 4.0 | raw fusion NOT SUITABLE；效果参考 EXCELLENT | detector POOR | `OFFICIAL_360_MODE` 参考 GOOD |
| AVM-SLAM | 正好 4 fisheye + BEV | YES（预生成 BEV） | 页面未确认随包标定参数 | NO | NO | 10 Hz sequence | 百度网盘，需人工处理 | 页面称 academic GPLv3；商业需联系 | 条件 EXCELLENT | NOT SUITABLE | 条件 EXCELLENT |
| SVS Calibration Benchmark | 正好 4 fisheye | YES，面向 homography/外参 | GT extrinsics；其他标定字段待包内核验 | 车道标注，不是 detector GT | NO | 40 videos / 32k frames | 百度网盘 | **UNKNOWN** | 技术 EXCELLENT；法律未清 | POOR | BLOCKED |
| BDD100K | 单前视 | NO | 不适用 | **YES**，道路 2D bbox | YES，且有灯色属性 | 100k×约 40 s | 门户/账号；本轮未取到原视频 | 当前数据条款需接受 | NOT SUITABLE | **EXCELLENT** | 单路 detector GOOD |
| DTLD | 单/双目交通灯专项 | NO | stereo/calib/vehicle data | **YES，仅 TL** | **YES：状态、图案、相关性、方向、track** | 15 Hz sequences | 人工注册 | 仅自身研究/教学；禁止商业和转交 | NOT SUITABLE | TL **EXCELLENT** | 条款内研究演示 CONDITIONAL |
| BSTLD | 单前视交通灯专项 | NO | 不适用 | **YES，仅 TL** | YES：red/yellow/green/off | test 连续约 15 fps；train 稀疏 | 表单/邮件链接 | 非商业 | NOT SUITABLE | TL GOOD | CONDITIONAL |

## 统一字段审计

### 访问、规模与法律状态

| Dataset | Official URL / paper | Research use | Competition demo use | Registration | Current download | Total size | Subset |
|---|---|---|---|---|---|---|---|
| WoodScape | [GitHub](https://github.com/valeoai/WoodScape) / [paper](https://arxiv.org/abs/1905.01489) | YES，按数据条款 | **UNKNOWN** | Google Drive 可能要求交互 | 代码下载已验证；数据页存在，数据包未验证 | 8.2k images + 8.2k previous images；字节数 UNKNOWN | 最小独立数据包 UNKNOWN |
| SynWoodScape | [paper](https://arxiv.org/abs/2203.05056) | UNKNOWN | UNKNOWN | UNKNOWN | 当前明确下载入口未找到 | 80k images | UNKNOWN |
| nuScenes / nuImages | [official](https://www.nuscenes.org/) / [paper](https://arxiv.org/abs/1903.11027) | YES，非商业 | CONDITIONAL | YES | 门户可用；本轮未绕过登录 | full 大型；mini 约 4 GB；nuImages 大型 | mini/官方 splits |
| Argoverse 2 | [official](https://www.argoverse.org/av2.html) / [sensor guide](https://argoverse.github.io/user-guide/datasets/sensor.html) | YES，非商业 | CONDITIONAL | NO for public S3 | **匿名 S3 已验证** | extracted sensor 约 1 TB；单 log 约 1 GB | YES，按 log、相机、文件 |
| A2D2 | [official](https://a2d2-dataset.github.io/) / [paper](https://arxiv.org/abs/2004.06320) / [AWS](https://registry.opendata.aws/aev-a2d2/) | YES | **CONDITIONAL**：BY-ND 对公开衍生视频需谨慎 | NO | **匿名 S3 已验证** | 原 tar bucket 实测约 2.486 TB；可逐文件 | **YES** |
| PandaSet | [official](https://pandaset.org/) / [paper](https://arxiv.org/abs/2112.12610) | YES | YES with attribution | 官方页面无账号要求 | 官方下载跳转本轮 404；HF 镜像可见 | HF zip 44,520,528,731 B | HF 镜像不便逐 scene |
| Waymo | [official](https://waymo.com/open/) / [terms](https://waymo.com/open/terms/) | YES，按自定义许可 | CONDITIONAL/需逐条核对公开展示 | YES | 登录、接受协议、云存储 | 大型，版本相关 | segment-level，仍需账号 |
| DDAD | [GitHub](https://github.com/TRI-ML/DDAD) / [paper](https://arxiv.org/abs/1905.02693) | YES，非商业 | CONDITIONAL | NO | 官方 S3 整包链接存在 | 257 GB train+val | 官方主包 NO；tiny 为附属路线 |
| KITTI-360 | [official](https://www.cvlibs.net/datasets/kitti-360/) / [paper](https://arxiv.org/abs/2109.13410) | YES，非商业 | CONDITIONAL | YES | 站点可用，2026-09-07 曾从备份恢复 | UNKNOWN | sequence/package |
| Boden AVM | [GitHub search entry](https://github.com/BodenAI/BodenAVMDataset) | YES | YES with attribution | UNKNOWN | 页面/云包本轮未实际取样 | 20k AVM images | 2k sample advertised |
| AVM-SLAM | [official](https://yale-cv.github.io/avm-slam_dataset/) | YES | CONDITIONAL | 百度网盘交互 | 链接/提取码页面存在，未下载 | UNKNOWN | UNKNOWN |
| SVS Calibration | [GitHub](https://github.com/Chunyu-Lin-bjtu/Camera-Calibration-for-the-Surround-View-System-A-Benchmark-and-Dataset) | UNKNOWN | **UNKNOWN** | 百度网盘交互 | 链接存在，未下载 | 40 videos / 32k frames | UNKNOWN |
| DAIR-V2X | [official](https://thudair.baai.ac.cn/) / [paper](https://arxiv.org/abs/2204.05575) | YES/按当前条款 | CONDITIONAL | YES/当前站点流程 | 未下载 | 大型 | split；不是 surround |
| Rope3D | [official](https://thudair.baai.ac.cn/rope) / [paper](https://arxiv.org/abs/2203.13608) | YES/按当前条款 | CONDITIONAL | YES/当前站点流程 | 未下载 | 50k images | image；不是 onboard surround |
| OpenLane / OpenLane-V2 | [OpenLane](https://github.com/OpenDriveLab/OpenLane) / [V2](https://github.com/OpenDriveLab/OpenLane-V2) | 代码与标签按各自许可；图像继承源数据条款 | CONDITIONAL | 继承 Waymo/nuScenes/AV2 | 标签可见；原图另取 | 版本相关 | YES；不是新 raw surround 源 |
| BDD100K | [GitHub](https://github.com/bdd100k/bdd100k) / [paper](https://arxiv.org/abs/1805.04687) | YES/接受数据条款 | CONDITIONAL | YES | 门户需人工操作 | 100k videos | split/文件能力需登录核验 |
| BSTLD | [official](https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset) / [GitHub](https://github.com/bosch-ros-pkg/bstld) | YES，非商业 | CONDITIONAL | 表单/邮件 | 人工下载链接 | 13,427 images | train/test |
| DTLD | [official](https://www.uni-ulm.de/en/in/institute-of-measurement-control-and-microtechnology/research/data-sets/driveu-traffic-light-dataset/) / [paper](https://doi.org/10.1109/ICRA.2018.8460737) | **YES，仅自身研究/教学** | CONDITIONAL | **YES** | 注册表单可用 | UNKNOWN | 下载包结构需登录后核验 |

法律判断不是法律意见。`CONDITIONAL` 表示只有在非商业比赛、署名/引用、数据不向第三方再分发，并满足原条款时才可能使用。A2D2 的 CC BY-ND 4.0 尤其需要区分：内部算法输入与公开发布经过裁剪、编码、拼接的衍生视频不是同一风险等级；正式公开展示前应得到主办方/数据方确认。

### 传感器、时序与标定

| Dataset | Video/images | Continuous | Cameras / directions | FPS | Resolution | Lens | Sync / timestamp | Intrinsics | Extrinsics |
|---|---|---|---|---|---|---|---|---|---|
| WoodScape release | raw fisheye images | 每样本只有当前帧+前一帧 | FV front, RV rear, MVL left mirror, MVR right mirror | 序列 FPS UNKNOWN | 1 MP；精确存储尺寸本轮未核验 | 190° fisheye | 原采集有 timestamp；公开表不是四路同刻 | YES | YES |
| SynWoodScape | images | synthetic sequence | 4 simultaneous surround | UNKNOWN | UNKNOWN | 190° fisheye | 论文称同时四路 | YES | YES |
| nuScenes | images/sweeps | 20 s | 6：front, front-left, front-right, back, back-left, back-right | cameras 12 Hz；annotated keyframe 2 Hz | 1600×900 | perspective/wide | keyframes time-aligned；raw sweeps 有 timestamp | YES | YES |
| nuImages | images | 6 s source clips，但发布目标是 image-level | 来自 nuScenes 相机；不是同步四路包 | 不作为同步视频 FPS | 1600×900 | perspective/wide | 有 image metadata；不提供本任务需要的同刻 quartet | YES/继承 | YES/继承 |
| AV2 Sensor | JPEG images | 15 s | 7 ring + 2 front stereo；ring 包含 front-center/FL/FR/rear-L/rear-R/side-L/side-R | 20 fps | front-center 1550×2048；其余 ring 2048×1550（实测） | perspective | 有 timestamp；实测首帧七路跨度约 42.50 ms | YES | YES |
| A2D2 | PNG images | 392,556 sequential frames | 6：front-center/FL/FR/rear-center/side-L/side-R | 实测相邻帧约 33.3 ms（约 30 Hz） | 1920×1208（实测） | front-center Telecam；side-L/R、rear-center Fisheye | timestamp；所选四路对齐最大 3.037 ms | YES | YES |
| PandaSet | JPEG images | 8 s | 6：front/left/right/back/front-left/front-right | 10 Hz | 1920×1080 | 5 wide + 1 long focus；逐相机型号需包内核验 | per-frame timestamp/pose | YES | YES |
| Waymo Perception | records/images | 20 s | 5：front/front-left/front-right/side-left/side-right；无 back | 10 Hz | camera-dependent | perspective | synchronized | YES | YES |
| Waymo E2E | records/video tensors | 20 s | 8 surround | 官方当前版本相关 | camera-dependent | perspective/wide | YES | YES | YES |
| DDAD | images | 5/10 s | 6，间隔 60° | 10 Hz | 1936×1216 | global-shutter perspective | 官方明确 time-synchronized | YES | YES |
| KITTI-360 | PNG | 11 continuous drives | 2 forward perspective stereo + 2 fisheye side/rear | camera FPS 本轮未确认 | camera-specific | mixed | per-camera timestamps | YES | YES |
| Boden AVM | 公开为 AVM images | UNKNOWN | 原采集 4×216° fisheye；raw 未确认公开 | UNKNOWN | raw description 1920×1080；AVM 1024×1024 | fisheye | 公开 AVM 已融合 | raw calib UNKNOWN | raw calib UNKNOWN |
| AVM-SLAM | images + BEV sequence | YES | 4 fisheye | 10 Hz | raw 1280×960；BEV 1354×1632 | fisheye | 页面呈现同步四路与 BEV | 包内 UNKNOWN | 包内 UNKNOWN |
| SVS Calibration | video/images | YES | 4 fisheye | UNKNOWN | UNKNOWN | fisheye | 同 rig；精确同步说明需包内核验 | UNKNOWN | GT extrinsics |

### 标签与场景覆盖

| Dataset | Native 2D box | Segmentation | person | car/vehicle | truck/bus | bicycle/motorcycle | traffic light | cone/obstacle | Night | Intersection | Dense traffic |
|---|---|---|---|---|---|---|---|---|---|---|---|
| WoodScape | YES，5 类 | YES，10 类 | YES | vehicle | 未细分 | bicycle box；semantic 有 motorcycle | **bbox YES**，无状态 | UNKNOWN | UNKNOWN | YES/场景存在 | YES |
| nuScenes | NO；3D 可投影 | lidar/地图，不是相机 dense semantic 主项 | YES | YES | bus/truck | bicycle/motorcycle | **NO object GT** | cone/barrier | YES | YES | YES |
| nuImages | YES，2D box + mask | YES | YES | YES | bus/truck | bicycle/motorcycle | NO | barrier/cone | YES | YES | YES |
| AV2 Sensor | NO；3D 可投影 | YES，任务相关 | pedestrian | regular vehicle | bus/truck | bicycle/motorcycle | NO；仅便携 `TRAFFIC_LIGHT_TRAILER` | construction cone | UNKNOWN | YES | YES |
| A2D2 | NO；前视 3D | **YES，41,277 标注帧** | YES | car | truck；bus 独立性需映射核验 | bicycle/small vehicles；moto 可合并 | **语义 traffic signal**；无灯色状态 | obstacles/trash、road blocks 等语义 | UNKNOWN | YES | YES |
| PandaSet | NO；3D cuboid | lidar semantic | pedestrian | car | bus/truck | bicycle/motorcycle | NO | cone/sign | UNKNOWN | YES | YES |
| Waymo Perception | **YES** | panoptic subset | YES | YES | vehicle 类可细分映射 | cyclist | standard 2D NO；panoptic YES | sign；cone 需版本映射 | YES | YES | YES |
| DDAD | NO，depth 为主 | 少量 panoptic/test GT 不公开 | 非主要 | 非主要 | 非主要 | 非主要 | NO | NO | UNKNOWN | YES | YES |
| KITTI-360 | 3D/semantic；非主流 2D bbox | YES 2D/3D semantic+instance | YES | YES | truck/bus 取决于标签映射 | bicycle/motorcycle | semantic YES | object/terrain 类 | NO/主要白天 | YES | YES |
| BDD100K | **YES** | YES | YES | car | bus/truck | bicycle/motorcycle | **YES + color attribute** | dedicated cone UNKNOWN | YES | YES | YES |
| BSTLD | TL only | NO | NO | NO | NO | NO | **24k 左右 TL，4 states** | NO | UNKNOWN | YES | 可有 |
| DTLD | TL only | NO | NO | NO | NO | NO | **>230k TL，344 属性组合** | NO | UNKNOWN | **YES，sequence 到通过路口** | 可有 |

## 重点审计结论

### WoodScape

已确认：原始车载 rig 是 FV、RV、MVL、MVR 四个 190° 鱼眼；有内外参与时间信息；公开检测框五类包括 pedestrian、vehicle、bicycle、traffic light、traffic sign；公开数据保留当前帧和前一帧。

否决主路线的原因不是“它没有四个相机”，而是**公开 release 不能恢复同一时刻的四路连续流**。官方 SynWoodScape 论文还明确解释：WoodScape 为增加场景多样性，没有对四个相机在同一时刻全部标注，因此不能直接支持 multi-camera BEV。再叠加 Proprietary 数据许可和比赛演示权 UNKNOWN，不应为了“正好四鱼眼”硬选。

### nuScenes

六个相机名称与题目一致。mini 有 10 scenes、404 个 annotated keyframes、约 4 GB；每 scene 20 s；相机 raw sweep 12 Hz，标注关键帧 2 Hz。它适合从同一 scene 生成同步多路 MP4，但若只取 `CAM_FRONT/CAM_BACK/CAM_FRONT_LEFT/CAM_FRONT_RIGHT`，会丢失后侧方向覆盖，不能声称完整 360。相机是车顶广角透视，适合 learned BEV/多路感知，不等同于近车低位鱼眼 AVM。

### Argoverse 2

七个 ring camera 名称已从真实 log 核验。官方 S3 支持匿名逐文件下载；探测到一个 log 的七路相机总计约 789 MiB，因此技术上可在 1 GB 量级获取完整 ring camera sequence。本轮只下载每路一帧和 metadata，没有下载整 log。

它比 nuScenes 更适合“高帧率多路压力测试”：20 fps、七路、匿名按文件访问。限制是没有 `ring_rear_center`；选 FRONT + SIDE_LEFT + SIDE_RIGHT + 某个 rear diagonal 仍有视场缺口。实测所选真实 log 的各相机首帧时间戳跨度约 42.50 ms，说明“与 LiDAR 触发协调”不等于所有相机同一曝光时刻。

### A2D2

匿名 S3 和逐文件下载已真实验证。全 rig 六相机覆盖 360；连续序列、每图 JSON、全局 `cams_lidars.json` 均存在。实际校准文件包含相机内参、畸变模型、车辆坐标外参、分辨率与 timestamp delay。

所选四个逻辑通道映射：

| Logical | A2D2 camera | Lens |
|---|---|---|
| FRONT | `cam_front_center` | Telecam |
| LEFT | `cam_side_left` | Fisheye |
| RIGHT | `cam_side_right` | Fisheye |
| BACK | `cam_rear_center` | Fisheye |

不能按相同文件序号直接对齐：该序列 front 同号帧比 side/rear 早约 0.77 s。本轮按图像 JSON 时间戳最近邻对齐，样本第 0 组为 front `1533906415488833`、left/right `1533906415491800`、back `1533906415491822` µs，跨相机偏差 2.989 ms；100 组的最小/最大/平均偏差为 2.939/3.037/2.9971 ms。

标签方面，A2D2 更适合语义分割与前视 3D 检测，而不是直接训练当前通用 YOLO：41,277 个标注帧是稀疏、非连续语义帧；12,497 帧有前视 3D box；traffic signal 是语义类，不是带灯色状态的 2D bbox。

### Boden AVM

公开内容的核心是 1024×1024 的**已融合 AVM 图像**及 parking-slot/freespace/semantic 标注。项目描述了四个 1920×1080、216° 鱼眼原始传感器，但本轮没有证据证明公开包同时提供可重放的四路 raw fisheye 和完整标定。因此它适合“融合效果参考”或模拟 `OFFICIAL_360_MODE`，不适合验证我们自己的四路 FPGA warp/blend。

## 实际下载与文件核验

数据放在仓库外：`D:\Datasets\competition_surround_eval\`。

| 内容 | 实际结果 |
|---|---|
| A2D2 sequence | `a2d2_20180810_150607`；400 PNG + 400 per-frame JSON；另有 probe/AVM 临时输入和全局标定 |
| A2D2 selected streams | front/left/right/back 各 100 帧；同车同序列；源时间跨度约 9.900 s；编码为 10 fps、10.000 s |
| A2D2 image check | 全部 1920×1208、RGB；四目录零字节文件数为 0 |
| A2D2 timing | 100 组；跨相机偏差 2.939–3.037 ms，平均 2.9971 ms |
| A2D2 calibration | 实际读取内参、vehicle-coordinate 外参、畸变、镜头类型、分辨率、时间延迟 |
| AV2 probe | 一个 val log 的 7 个 ring camera 各一帧 + intrinsics/extrinsics/annotations/ego pose；实际 metadata 约 3.02 MiB |
| AV2 labels in probed log | 5,337 rows；该场景实际出现 BICYCLE、BICYCLIST、PEDESTRIAN、REGULAR_VEHICLE、TRUCK、WHEELCHAIR |
| WoodScape | 官方代码 archive + train/val 列表；zip SHA-256 `92C5E1DB43986CD8FA481C8C78B98FCEABAFEC6CF7E3756A3CAC0354E035C005`；未下载数据包 |
| **总量** | **1,387,175,478 bytes = 1.292 GiB，909 files**；低于 2 GB 目标和 5 GB 硬上限 |

## 四路视频与预览

输出目录：`outputs/surround_dataset_preview/`。所有视频均为 H.264、1920×1208、10 fps、100 帧、10.000 s。

| File | Bytes | Result |
|---|---:|---|
| `front.mp4` | 21,829,623 | PASS |
| `left.mp4` | 21,037,994 | PASS |
| `right.mp4` | 29,592,696 | PASS |
| `back.mp4` | 25,841,453 | PASS |
| `mosaic_2x2.mp4` | 23,925,036 | PASS；布局为 `FRONT | RIGHT / LEFT | BACK` |
| `mosaic_2x2_first_frame.jpg` | 167,918 | 已视觉检查，同一地点/车辆时刻合理一致 |

这些文件是同步数据集预览，不是 FPGA 结果，也不是 panorama/AVM。

## PC 端 AVM 几何可行性探针

一次性探针将真实 A2D2 内外参和畸变模型用于 15 m×15 m 地面平面逆投影；没有做学习、接缝寻找、曝光补偿或多频段融合。

| Input | Valid ground-plane coverage | 观察 | 判断 |
|---|---:|---|---|
| FRONT + SIDE_LEFT + SIDE_RIGHT + REAR_CENTER | 0.790844 | 前方两个对角大缺口、车体盲区、接缝明显；高于地面的物体被拉伸 | **能做坐标映射，但不能生成合理完整 AVM；POOR** |
| 全六路 | 0.914253 | 两个前斜相机补上多数对角缺口；仍有车体盲区和简单 overwrite 接缝 | **具备粗 AVM 几何条件；GOOD，不是成品** |

这证明 A2D2 标定足以映射到统一地面，但也证明“所选四路”不具备无缝全周覆盖。缺口不是代码 bug，而是相机方向/FOV 组合造成的几何限制。

## 比赛具体用法

### 当前可执行路线

```text
A2D2 timestamp-aligned FRONT / LEFT / RIGHT / BACK
                    ├─> FPGA 四输入/采集链路模拟 ─> 2×2 mosaic / OSD 验证
                    └─> PCIe ─> RK3568 四路 AI/ADAS 吞吐与调度验证
```

- **FPGA 融合开发**：PC 阶段用 A2D2 **全六路**验证标定解析、逆映射、地面坐标和 blend 需求；最终四输入硬件算法必须换成真正四个低位宽鱼眼且有重叠区的 rig/受权数据再验收。
- **RK3568 四路压力测试**：直接用已生成的 A2D2 四个 10 秒 MP4。它们同车、同序列、时间戳对齐，不是四个无关视频。
- **detector 训练**：通用道路 2D 检测首选 BDD100K；A2D2 只作为语义/域覆盖辅助，不应把语义 mask 自动转换后伪称官方 2D GT。
- **traffic-light 专项**：优先 DTLD；若登记/条款不合适，再退到 BSTLD。
- **最终比赛演示**：数据许可确认前只做内部技术演示；公开发布 A2D2 派生 MP4/AVM 前单独确认 BY-ND 的适用性。

### `OFFICIAL_360_MODE`

若主办方只给一条已经融合好的 panorama/AVM：

```text
pre-stitched panorama / AVM
          ↓
FPGA input / playback
          ↓
按画面几何定义拆 FRONT / LEFT / RIGHT / BACK logical views
          ↓
复用四路 PCIe + RK3568 AI backend
```

可用 Boden AVM 或 AVM-SLAM 的预生成 BEV 提前模拟“预融合输入”，也可用本轮 A2D2 六路 PC 探针的后续成品作为内部输入。但这只验证后端逻辑复用，不验证 FPGA 从 raw camera 生成 AVM 的能力。裁切规则必须根据主办方视频的实际投影定义，不能假设任意 AVM 都能还原四个透视相机。

## 明确拒绝/降级的候选

- **DAIR-V2X**：车端与路端协同视角，不是同一车体前后左右 surround。
- **Rope3D**：路侧单目，非 onboard multi-camera。
- **OpenLane**：原版主要前视；OpenLane-V2 是基于 nuScenes/AV2 的拓扑标签层，不是新的 raw surround source。
- **nuImages**：优秀的 2D image-level 标签源，但不是同步多相机 quartet。
- **Waymo Perception**：没有 rear camera；无法构造真实 BACK。
- **Boden AVM**：公开核心是已经 stitch 的 BEV；不能验证 raw four-camera fusion。
- **四个独立 MP4**：无论内容多像，都不满足同车、同步、固定外参，禁止冒充环视。

## 需要人工处理的阻塞项

1. nuScenes/nuImages：登录并接受当前数据协议；本轮未用非官方方式绕过。
2. Waymo：Google 账号、数据许可和云端访问。
3. BDD100K：门户登录；需要确认原始 video split 与公开演示权。
4. DTLD：填写姓名、邮箱、单位并接受仅研究/教学条款；数据不得交给第三方。
5. BSTLD：表单/邮件获取下载链接。
6. AVM-SLAM：百度网盘提取；下载后必须确认 raw 四路、BEV、标定和 timestamp 是否同包。
7. SVS Calibration Benchmark：百度网盘；先向作者确认数据许可证/比赛演示权，再下载。
8. WoodScape：Google Drive 数据访问与 Proprietary 许可；即使获得也仍需确认是否存在未公开的同步四路序列。
9. PandaSet：官方 Scale 下载入口本轮返回 404；公开 HF 镜像为约 44.5 GB 单体包，超出本轮限额。

## 18 项最终答复索引

1. **找了哪些**：WoodScape/SynWoodScape、nuScenes/nuImages、AV2、A2D2、PandaSet、Waymo Perception/E2E、DDAD、KITTI-360、Boden、AVM-SLAM、SVS Calibration、DAIR、Rope3D、OpenLane/V2、BDD100K、BSTLD、DTLD。
2. **真正同步多摄像头**：A2D2、nuScenes、DDAD、PandaSet、Waymo、AV2（协调采集但不同相机曝光有毫秒级时差）；WoodScape 公开版不是同步四路序列。
3. **正好四路**：WoodScape rig、SynWoodScape、AVM-SLAM、SVS Calibration；KITTI-360 也有四相机但方向布局不是 FRONT/LEFT/RIGHT/BACK。
4. **能做真实 360/AVM**：A2D2 全六路、nuScenes 全六路、AV2 全七路、DDAD 全六路具备统一标定；精确近车 AVM 更匹配 AVM-SLAM/SVS，但仍受访问/授权阻塞。
5. **完整 calibration**：A2D2、AV2、nuScenes、DDAD、PandaSet、KITTI-360、WoodScape；AVM-SLAM 包内情况 UNKNOWN。
6. **适合 detector**：BDD100K 最佳通用 2D；nuImages 次选；A2D2 更偏语义/3D。
7. **有 TL 标注**：WoodScape 2D TL 类、A2D2 TL 语义、BDD100K TL bbox+颜色、DTLD/BSTLD 专项。nuScenes/AV2 标准 object taxonomy 不提供普通 TL 检测 GT。
8. **比赛四路数据源**：A2D2。
9. **FPGA 360 融合**：当前用 A2D2 全六路做几何开发；最终四输入验收等待授权后的 AVM-SLAM/SVS 或自采四鱼眼标定数据。
10. **AI 训练**：BDD100K；traffic-light 专项 DTLD。
11. **实际下载**：A2D2 400 个对齐候选图+JSON和标定、AV2 七路 probe+metadata、WoodScape 官方代码与列表。
12. **总大小**：1,387,175,478 bytes（1.292 GiB）。
13. **四路 MP4**：YES，全部 10 s/10 fps/100 frames/H.264/1920×1208。
14. **2×2 preview**：YES，正确布局；明确不是无缝 360。
15. **真正 AVM 几何条件**：A2D2 全六路 YES/GOOD；当前四路只有局部地面映射条件，完整性 POOR。
16. **许可**：研究使用多数可行；比赛公开演示均须逐数据条款确认，A2D2 BY-ND、AV2/nuScenes 非商业 SA、DTLD 禁止商业/转交尤其需要注意。
17. **人工阻塞**：见上一节九项。
18. **下一步建议**：由项目负责人在 A2D2 内部演示、AVM-SLAM/SVS 授权核验、DTLD 登记三个选择上作决定。本轮到此停止，不进入 FPGA 开发。

## 证据等级说明

- **实测**：本机下载文件、metadata、时间戳、图像、视频或标定解析所得。
- **官方确认**：官方主页、论文、官方 GitHub/文档或官方开放数据注册表。
- **UNKNOWN**：官方材料没有明确说明，或入口需要登录/网盘/表单而本轮没有取得数据包。
- **推断**：例如“比赛演示可能属于非商业研究展示”；所有此类判断都标为 CONDITIONAL，不当作授权结论。

