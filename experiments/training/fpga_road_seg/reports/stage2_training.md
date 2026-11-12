# 第二阶段：Candidate-plus 混合数据微调

第二阶段独立写入 `outputs/fpga_road_seg_stage2/`；第一阶段目录、全部权重、量化报告均保留。
不重训 Candidate / TwinLiteNet+ Small，不修改网络结构。Candidate-plus 仍为 32,524 参数、
320×192 输入下 84,718,080 MACs；这些是软件统计，不是 FPGA 实测。

## 数据来源与核验

- 官方：[stu9113611/RLMD](https://github.com/stu9113611/RLMD)，README 指向公开 Google Drive。
- 下载：[固定版本公开副本](https://huggingface.co/datasets/veetinator/Road_Line_Marking_Dataset/tree/a1f0fdbcb3f56c0a6cc999c7dbf860d67246bbb0)。
- ZIP：1,356,808,912 bytes；SHA-256 `1269c85675a04fbb0f48cb48517d2fd8e8f8614b7364146f28355b19dee845c4`，
  与固定版本 LFS 摘要一致。支持 Range 续传，保留 TLS 校验。
- 2,137 对 JPEG / RGB PNG：原 train 1,737、val 400，全部尺寸匹配、未知颜色 0。
- 副本类别 CSV 与官方 25 类 CSV 的 ID、名称、颜色逐项相同。
- 官方 Drive train/val 各 50 个可见样本文件名在副本中全部匹配且划分一致；各下载一张官方图像，
  `10.jpg`、`45.jpg` 均与副本 SHA-256 相同。这是抽样来源核验，不是逐文件官方签名验证。
- 官方仓库无明确 LICENSE；副本声称 MIT，不能据此认定原作者授予 MIT。
  本地科研实验保留出处和引用；不向 Git 提交原数据、Mask 或数据包，不声称已解决再分发许可。

## 标签与独立验证

四通道顺序：drivable_area、solid_lane、dashed_lane、crosswalk。
RLMD：solid ID 4–8，dashed ID 9–10，crosswalk ID 2；所有其余已知密集类别为这些任务的负样本。
RLMD drivable 的 valid 恒为零；未知颜色屏蔽；BDD crosswalk 在混合 manifest 按行屏蔽，
缓存保留原有监督，因此不会污染旧实验。测试覆盖原生和缓存读取两条路径。

人工查看真实 Mask 覆盖图：crosswalk 是整个过街区域的密集区域，包含条纹间隙，
不是单独白色条纹像素。应按这个语义解释后续 IoU / Recall。

为减少原逐帧划分中的近邻泄漏，将数字文件名按相邻编号间隔 >500 分组，day 名称按序列前缀分组，
固定 SHA 排序选择约 20% 验证组。新 train 1,698、val 439；跨划分同 SHA 或 dHash 距离 ≤2 的隔离数为 0。
真正的录制序列 ID 未提供，不能声称已证明所有场景独立。原 split 保留在 metadata。
BDD 继续使用既有 62,984 train、7,012 internal-val、10,000 official-val held-out，不重复恢复。

## 可恢复的真实训练

`configs/stage2_candidate_plus.json`：从第一阶段 Candidate-plus `best.pt` 只加载权重，
新优化器学习率 1e-4（原 1e-3），AMP、batch16、4 workers，12 epoch 上限。
每轮有放回抽样 48,000 张，BDD/RLMD 概率 0.8/0.2，并使用水平翻转及轻度亮度增广。
轮次是混合采样轮次，不等同于遍历一次全部 BDD。

分别计算 BDD、RLMD 内部验证。选择分数 = 0.2×(BDD drivable IoU + solid F1 + dashed F1)
+ 0.4×RLMD crosswalk IoU；三项 BDD 指标任意一项相对同一内部验证基线下降超过 2.5 个百分点，
该 checkpoint 不具备成为 best 的资格。每两轮无有效改善降学习率，至少 10 轮后 patience4 早停。
最佳/最后权重原子保存，5 轮保存周期权重；优化器、scaler、随机状态随 checkpoint 保存。

启动（使用现有 GPU Python 环境）：

```powershell
python -u -m experiments.training.fpga_road_seg.run_stage2
```

控制器以独立进程运行，日志在 `outputs/fpga_road_seg_stage2/run.log`，状态在 `status.json`；
训练失败自动从 last.pt 重试，重新启动控制器也会继续而非从零训练。
睡眠抑制只持续到控制器退出，允许关闭屏幕。当前阶段的实际指标以 report.json / 最终结果报告为准，
此文不预填训练或量化成功结论。
