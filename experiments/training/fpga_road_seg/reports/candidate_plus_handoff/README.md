# Candidate-plus FPGA 交接生成与实测记录

正式文档：`docs/FPGA_CandidatePlus_技术交接文档.md`。完整交接包在本机 `outputs/fpga_candidate_plus_handoff/`，ZIP/RAR 路径和 SHA256 见 DELIVERY.json。

在仓库根目录，现有第二阶段最终产物和提供的视频均已存在时：

```powershell
python -u -m experiments.training.fpga_road_seg.build_handoff
python -u -m experiments.training.fpga_road_seg.handoff_package --root outputs/fpga_candidate_plus_handoff --model-root outputs/fpga_road_seg_stage2_qat --rar-executable PATH/TO/Rar.exe
python -m pytest experiments/training/fpga_road_seg/tests/test_handoff_integer.py experiments/training/fpga_road_seg/tests/test_data.py experiments/training/fpga_road_seg/tests/test_stage2.py -q
```

19 项回归测试通过。包在归档前、ZIP 解压后、RAR 解压后分别完成完整软件自检；FX QAT/FP32 权重严格加载且 TorchScript 可运行。6 个真实视频样本的 20 个 UINT8 stage 和 14 个 INT32 acc 均重新运行并与对应 golden 逐位一致。工具还通过单字节损坏的负控制。

`integer_stages.csv`、architecture_summary.json 为真实图提取，不包含权重数据。整个整数规范相对生产 QDQ 的全验证集精度尚未重新测量；六样本一致率不代表精度。没有 FPGA 实测结果。

归档命令默认拒绝覆盖已有 ZIP/RAR。重新生成时选择新的输出目录；文件均使用 UTF-8。RAR 需要真正的 RAR 编码器，不能用 7-Zip 的 ZIP 冒充。
