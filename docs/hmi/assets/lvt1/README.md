# LVT1 升级截图说明

日期：2026-10-29  
分支：`feature/lvt1-ego-bev-20261028`  
环境：LubanCat-1 RK3566，Qt 5.12.8，独立测试目录，**未替换生产二进制**。

| 文件 | 内容 | 数据来源 |
|---|---|---|
| `single_live_targets_20261029.png` | 单路 + 车道感知 + 本车控制 + 真实 YOLO 目标 3D | 本地 60 帧 PGVF fixture + 第二 AI sidecar（A 模型 5Hz）+ LVT1 fixture `left-solid/right-dashed` |
| `quad_live_targets_20261029.png` | 四路 + 右侧新卡片 | 同上 |
| `spatial_live_targets_20261029.png` | 空间视图 + 真实 YOLO 目标 + LVT1 线型 | 同上；修复 camera X 镜像后 |
| `single_left_change_denied_20261029.png` | 左变道请求 + 左实线拒绝 | 同上；LVT1 fixture `left-change-denied` |

## 重要边界

- `SIM` 表示 LVT1 是 fixture，不是 FPGA 实板。
- 视频是本地 PGVF fixture，不是 PCIe 采集。
- AI 目标是真实 RKNN 推理结果，但输入是本地视频。
- 测试时同时运行生产 Qt/AI，测试 Qt 的 FPS 低于生产；不能作为生产性能。
- 生产二进制、服务、FPGA RTL、内核模块均未修改。

详细：[`docs/hmi/LVT1_REAL_DATA_HMI.md`](../../../hmi/LVT1_REAL_DATA_HMI.md)
