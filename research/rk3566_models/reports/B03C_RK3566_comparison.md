# B-03C-R：RK3566 双模型实机对比

**结论：PARTIAL。** SSH、模型加载、输出解码和短时测速完成。BDD YOLOv8n INT8 在本次板端测试中更快，且输出更多车辆、行人和交通灯框；但测试图没有可用 GT，不能据此判定检测精度或直接替换正式模型。

## 实机条件

- SSH 已恢复：`lubancat.local` 解析并连接到 `fe80::c0fb:d16f:cd8b:b5c%21:22`，ED25519 公钥认证成功；未修改 SSH 配置。
- 重连时板端 uptime 约 9 分钟，原因无法确认；本轮没有发出重启命令。原 `/tmp` 候选副本已不存在，重新从本机已验 SHA 的文件上传到独立临时目录。
- 测试期间暂时停止 `adas-ai-bridge` 和 `adas-hud-bridge`。无可见的 AI/RKNN 推理进程后依次测试，未同时运行两个模型；完成后两服务均已恢复为 `active`。
- RKNN Runtime `2.3.2`，NPU Driver `0.8.2`，频率 `600 MHz`。两模型输入均为 `512×288`、batch=1、RGB `UINT8 [1,288,512,3]`；阈值 `conf=0.25`、NMS IoU `0.45`。两模型输出分别为 `[1,4,3024] + [1,80,3024]` 和 `[1,4,3024] + [1,10,3024]`，数值均有限。
- 每模型 NPU-only 预热20次、计时100次；完整 Pipeline 另预热20次、计时100次。Pipeline 包含内存图像预处理、推理、解码/NMS和坐标映射，不含磁盘解码。
- 板端没有找到 B-02 原50张清单；本轮使用现有 `holdout100` manifest 的前50张。没有这些图对应的 GT 标注，因此 Recall、FP、Precision 均不能计算。
- `/sys/class/devfreq/fde40000.npu/load` 在 bridge 停止、未见推理进程时仍固定显示 `100@600MHz`，debugfs 负载接口不可读。测速是在 bridge 停止的短时环境中完成，但驱动负载计数无法独立校验，结果不应外推为长时独占性能。

## 实测结果

| 指标 | 原版 YOLO11n RKNN | BDD YOLOv8n RKNN INT8 |
| --- | ---: | ---: |
| 模型输入 / 输出 | 512×288 RGB UINT8 / 80类 | 512×288 RGB UINT8 / 10类 |
| NPU-only 平均 / p95 / FPS | 36.149 / 39.266 ms / 27.664 | 25.451 / 28.152 ms / 39.291 |
| Pipeline 预处理 / NPU / 后处理 | 7.504 / 31.610 / 7.148 ms | 7.624 / 24.317 / 6.057 ms |
| 完整 Pipeline 平均 / p95 / FPS | 46.262 / 49.020 ms / 21.616 | 37.999 / 42.740 ms / 26.317 |
| 车辆 Recall | N/A，无 GT | N/A，无 GT |
| 行人 Recall | N/A，无 GT | N/A，无 GT |
| 交通灯 Recall | N/A，无 GT | N/A，无 GT |
| 总 FP | N/A，无 GT | N/A，无 GT |
| 50图框数（car+truck+bus / person / traffic light） | 207 / 3 / 10 | 393 / 17 / 96 |
| 进程 CPU / RSS（测量末采样） | 73.3% / 266.81 MB | 75.3% / 263.40 MB |

进程 CPU 百分比是测量间隔的采样值；同期系统 CPU 末采样分别为25.8%和25.7%。Pipeline 分阶段数值相加为总时延口径；NPU-only 是独立的100次计时，故与 Pipeline 内的 NPU 阶段均值不完全相同。

## FP32 与 INT8 输出核对

BDD YOLOv8n FP32 参考在同一50张图上由 RTX 4060、PyTorch `2.5.1+cu121`、Ultralytics `8.4.68` 运行。模型原始类名 `motor`、`bike` 分别映射为 `motorcycle`、`bicycle`；`rider` 保持独立。下表是同类逐图框匹配的一致性检查，不是检测精度指标。

| 类别 | FP32 框数 | RKNN INT8 框数 | IoU≥0.5 匹配 | FP32 未匹配 / INT8 未匹配 |
| --- | ---: | ---: | ---: | ---: |
| car | 351 | 378 | 299（85.2%） | 52 / 79 |
| person | 18 | 17 | 15（83.3%） | 3 / 2 |
| traffic light | 83 | 96 | 48（57.8%） | 35 / 48 |
| truck | 10 | 11 | 9（90.0%） | 1 / 2 |
| bus | 4 | 4 | 4（100%） | 0 / 0 |

交通灯框数没有下降，代表图上 FP32 与 INT8 的灯位大体相似；但小灯框的位置、数量存在明显分歧。无 GT 时，额外的48个 INT8 未匹配框不能称为误检，也不能证明是找回目标。图像对照保存在本机 `D:\Datasets\adas_edge_training\b03c_rknn_validation_20261021\visual_*.png`，没有上传或提交 BDD 图片。

## 结论与状态

- **检测谁更好：** BDD YOLOv8n 在样本上输出了更多车辆、行人和交通灯框；框数不能代表敏感度或 Recall，也不能代替 FP。没有 GT，精度胜负未定。
- **4060/RK3566 上谁更快：** RK3566 上 BDD YOLOv8n INT8 更快；本轮 NPU-only 约快1.42倍，完整 Pipeline 约快1.22倍。PC 端未用于替代板端速度。
- **INT8 是否明显损害交通灯：** 没有看到整体少检的迹象，但框一致性仅48/83达到 IoU 0.5，不能据此判定量化无损。
- **是否替换主检测器：** 暂不建议。保留 YOLO11n 正式模型；待有带 GT 的独立道路样本后，再确认候选的交通灯误检和召回。
- **是否保留 Traffic-Light Specialist：** 是。主检测器的灯框质量尚未验证，且本任务未验证灯色状态；Specialist 未修改。
- **模型文件校验：** YOLO11n 前后 SHA256 均为 `7a697ccc476954f6f36f7441b842257a2d7a019653cc4bad2fbf8de327c50591`；正式 Specialist `tl_specialist_512x192_hybrid2.rknn` 前后 SHA256 均为 `becfaada5da49c4dd6c143e815f8f957fb7b7ed759040d4e18c685a9c434e6f8`。候选 SHA256 为 `BDFAA64782EA1EBBDD8F23F18ABD91567C193A0C44C02D1B94D08661A227BD82`。
- **ONNX / RKNN：** B-03C 离线静态 ONNX 导出和 RKNN INT8 构建此前已成功；本轮 RK3566 实际加载、推理和输出形状检查成功。
- **最终状态：** `PARTIAL`。短时性能已实测；精度排名、FP/Recall、长时独占性能仍未验证。
