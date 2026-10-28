# 640×640 四路 AVM RTL

本目录是按现有 A2D2 冻结标定生成的四路 640×640 AVM 核，目标是先验证固定点数据通路和资源下限。

## RTL 接口

`avm_4ch_640_stream.v` 不在片上保存 LUT 或图像帧：

- LUT 外部读口：19-bit entry address、96-bit data；
- 四路 RGB565 外部统一读口：2-bit camera slot、18-bit pixel address、16-bit data；
- 一次只发一个读请求，外部存储器返回一个 `*_rd_valid`；
- 每个输出像素最多读取两路相机的四邻域，然后做 Q0.4 双线性和 Q0.6 融合。

这是省片上资源的参考核，不是 DDR burst/cache 完整集成核。实际系统应在外部存储器接口前增加 burst/cache 层。

## 已验证结果

使用 `outputs/avm_fpga_640x640_20261011` 的 640×640 LUT、4 个 RGB565 输入和 expected 输出：

- RTL：409,600/409,600 像素通过，0 mismatch；
- RTL：8,019,193 个仿真时钟周期；
- Yosys generic LUT4 mapping：2,353 LUT、260 FF、0 memory；
- Yosys generic LUT6 mapping：1,698 LUT、260 FF、0 memory。

Yosys 数字不是 PG2L100H 的最终 LUT 数。PG2L100H 专用 PDS 当前不在本机，必须在安装 PDS 的环境中把本模块加入原工程后重新跑 synth/device-map 才能得到器件报告。

## 运行

RTL testbench 使用 Icarus Verilog，输入路径通过 `+LUT/+SRC0/+SRC1/+SRC2/+SRC3/+EXPECTED` 传入。完整命令和原始日志保存在 `build/` 下。
