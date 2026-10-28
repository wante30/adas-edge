# 640x640 四路 AVM RTL 计划

## 目标

把现有四路 640×320 RGB565 输入映射到 640×640 RGB565 输出，并保持片上资源优先约束。输入帧、映射表和四邻域像素由外部存储器提供，RTL 不实例化整帧缓存。

## 冻结接口

- 四个输入槽：local camera slot 0..3。
- 输入像素：RGB565 little-endian，640×320，row-major。
- 输出像素：RGB565，640×640，row-major。
- 每个输出像素一条 96-bit LUT entry：两个相机、两个 18-bit top-left address、Q0.4 坐标小数、Q0.6 alpha、valid/blend。
- source memory：单共享读端口，RTL 依次读取 A/B 两个相机的四邻域像素。
- LUT memory：19-bit entry address，96-bit read data。

## 省资源架构

1. 单个 LUT 读取器；
2. 单个四邻域 RGB565→RGB888 双线性采样器；
3. 单个 Q0.6 融合器；
4. 不在 FPGA 内保存 640×320 或 640×640 全帧；
5. 不做四路并行插值，接受较高外部 DDR 访问次数换取 LUT/FF 节省。

## 验收条件

- MATLAB 生成 640×640 LUT 和至少一个真实 4CH 输入向量；
- RTL testbench 逐像素比较输出 RGB565；
- RTL 仿真无 X/Z 输出，frame_done 正确；
- 通用综合报告 LUT/FF/存储器统计；
- 若 Pango PDS 可用，再补 PG2L100H 专用 device-map 资源报告；否则明确标注为未完成 Pango 资源映射。

## 已知限制

当前工程包没有 AVM 顶层 RTL、没有顶层 testbench，且本机暂未发现 Pango PDS 可执行文件。因此本计划不会把 MATLAB 或通用 LUT 统计冒充成 PG2L100H 最终布局结果。
