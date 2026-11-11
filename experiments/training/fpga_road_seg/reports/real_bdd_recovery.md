# BDD100K 真实数据恢复与标注核查

2026-11-12，PR #16。本记录是新模型训练的数据证据，不引用旧模型精度。

## 已完成的数据恢复

ETH 官方下载在当前网络不可用后，使用公开的
[Internet Archive BDD100K 原包副本](https://archive.org/details/bdd100k)。
HTTPS 证书验证保持启用，没有绕过登录或授权。下载器使用 HTTP Range 分块、
失败重试、块续传、完整长度与 MD5 校验；校验值来自镜像元数据，不能视为官方数字签名。

| 包 | 实际字节数 | MD5 |
|---|---:|---|
| bdd100k_images.zip | 6,941,520,264 | b538e3731a132e28dae37f18c442c51e |
| bdd100k_labels.zip | 154,084,208 | 035bd9e56c0c2dbdef2165e0bf1cf6a5 |

数据独立保存在 `C:/Users/JingyuWang/datasets/bdd100k_issue15/`。
100,000 张 JPEG 已解压，70,000 train + 10,000 val JSON 一一配对，无缺失。
20,000 官方 test 图像没有公开 JSON，保留为无监督索引，不能声称有 test 真值。
80,000 份原生分辨率 Mask 已生成，正式 manifest 校验通过。

复核后，3,533 帧缺失 drivable 对象、3,834 帧缺失 lane 对象；这些任务有效性置零，
其余可用任务继续监督。修正之前的试训权重已隔离，不续训、不列入最终指标。

## 复用 PR #4 的方法

已读取历史提交 `121f186b` 中 `research/b05n_p2_mtl/README.md`、
`scripts/prepare_data.py`、`mtl/label_render.py`、`mtl/data.py`。
适配同一套每图 JSON、frames/objects 包装、Scalabel poly2d L/C 曲线与 style 属性。
`area/drivable` 与 `area/alternative` 合并为可行驶区域；车道类别排除 road curb
和 crosswalk，再按 solid/dashed 分别渲染。使用原图 10 px 线宽，对应历史
512×288 画布 4 px；曲线采样密度和 OpenCV/PIL 光栅化有区别，不声称逐像素复刻。

日间、夜间、路口等 12 张真实图像的接触表已人工视觉抽查，区域和曲线与道路大体对齐。
BDD crosswalk 仍然是曲线标签：仅正像素参加独立弱监督实验；主训练排除该通道。
没有真正的密集斑马线真值，dense IoU/Recall 必须为空，不能将曲线 IoU 充当面域精度。

## 防止划分泄漏

train 按视频 ID 前缀分组随机留出内部 val（seed 15），官方 val 作为最终 held-out。
原始 split 的 `fe189115` 视频前缀同时出现于 train/val：隔离 4 张 train 帧，
保留官方 val。原数据与 Mask 未删除；隔离原因和样本见 `split-audit.json`。
最终使用 62,984 train、7,012 internal val、10,000 held-out 样本。

SHA-256 检查发现 3 对相同 JPEG，仅出现在同 split 内；没有跨 split 相同文件内容。
同 split 重复保留，跨 split 内容重复和视频组交叉仍会拒绝执行。

## 新模型执行约定

Candidate 8,508 参数，Candidate-plus 32,524 参数，TwinLiteNet+ Small 121,552 参数。
Small 保留原有 drivable/lane 两头，不能直接输出实线/虚线，相关指标应为空。
先分别训练 512 张真实样本/128 张内部验证的 3 epoch，再继续全量训练。
全量和小样本验证范围不同，最佳权重选择会重置，并在 history 中逐 epoch 记录样本量。

RTX 4060、AMP、batch 16、8 workers；最佳/last/每 5 epoch 权重独立保存。
训练会重试并恢复 last，报表逐 epoch 原子写入。原生 NPZ 单进程实测 128 张用时
14.48 秒（8.84 张/秒，包含当时并行校验负载）；因此增加精确等价的预处理缓存，
RGB 保留 uint8、二值 Mask 位打包，避免每 epoch 解压原生 1280×720 Mask。
缓存按输入文件路径、大小、mtime、预处理版本和尺寸隔离，不改变标注或几何变换。

对 16 张样本做 cProfile 后，二值检查 `np.isin` 占 0.626/1.204 秒；改为等价的 uint8
范围检查后，128 张原生样本读取实测 4.10 秒（31.23 张/秒）。这与缓存后的吞吐是不同测量。

独立持久进程入口为 `run_persistent.ps1`，可等待已有流水线结束后恢复/补齐后处理，
最多重试三轮。`-Publish` 只在三模型真实 10k held-out 评估和量化报告存在时，
提交有大小限制的轻量报告并推送到 PR #16 分支；数据、权重和完整日志不提交。

完整产物在 `outputs/fpga_road_seg_final/`，最终指标以对应新 checkpoint 的
`metrics.csv` 和 `README_RESULT.md` 为准；训练未结束前不填写估计精度。
FPGA 算子、定点、资源和时序仍需硬件团队资料，详见 `docs/fpga_road_seg_handoff.md`。
