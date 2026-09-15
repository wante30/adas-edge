# LubanCat RK3566 ↔ Pango FPGA PCIe 现场记录

日期：2026-09-15（鲁班猫系统时钟显示为 2024-06-18，以下保留设备原始日志时间）

## 结论

PCIe Root Complex、FPGA Endpoint 枚举、链路训练、BAR 分配和 Pango 内核驱动加载均已通过。当前链路实际协商为 **PCIe Gen2 5 GT/s x1**；FPGA Endpoint 能力声明为 x4，因此存在 lane 降级。DMA 和视频帧传输尚未测试，不能据此宣称视频链路完成。

## 主机与设备

```text
Board: EmbedFire LubanCat-1
OS: Ubuntu 20.04.6 LTS
Kernel: 4.19.232
Arch: aarch64 / ARM64
BDF: 01:00.0
VID:DID: 0755:0755
Class: Memory controller [0580]
```

## PCIe 链路

```text
LnkCap: Port #0, Speed 5GT/s, Width x4
LnkSta: Speed 5GT/s (ok), Width x1 (downgraded)
```

Linux 枚举日志：

```text
rk-pcie 3c0000000.pcie: PCIe Link up, LTSSM is 0x130011
pci 0000:01:00.0: [0755:0755] type 00 class 0x058000
pci 0000:01:00.0: 4.000 Gb/s available PCIe bandwidth, limited by 5 GT/s x1 link
```

BAR：

```text
BAR0: 0xf4200000 - 0xf4201fff (8 KB, 32-bit)
BAR1: 0xf4204000 - 0xf4204fff (4 KB, 32-bit)
BAR2: 0xf4202000 - 0xf4203fff (8 KB, 64-bit)
```

AER 能力状态中 `UESta` 和 `CESta` 均为全清除；现场检查未见 AER 错误。

## Pango 驱动

驱动文件：`~/pcie_test/pango_pcie_dma_alloc/driver/pango_pci_driver.ko`

```text
insmod: PASS
lsmod: pango_pci_driver
device node: /dev/pango_pci_driver
Kernel driver in use: pango_pci_driver
/sys/bus/pci/devices/0000:01:00.0/enable: 1
```

驱动日志关键行：

```text
pci_enable_device result : 0
sizeof(dma_addr_t) == 8
Using a 64-bit DMA mask.
pci_request_region result : 0
ioremap result : 0
create device.
```

## 尚未完成

```text
PIO: NOT TESTED
DMA: NOT TESTED
Video frame transfer: NOT TESTED
GTK2 userspace app: blocked by unavailable libgtk2.0-dev on board
```

下一步应以 BAR 小读写、128-byte DMA、连续 DMA buffer、帧号/CRC/丢帧统计的顺序验证视频传输。x1 链路的理论有效带宽足以覆盖 1280x720@25 FPS 的 NV12/YUYV 数据量，但实际持续 DMA 吞吐仍需测量。

## 证据边界

本记录只描述本次鲁班猫现场 PCIe 枚举和驱动加载结果。它不把软件模拟、历史 RK3566 NPU 测试或未执行的 DMA/视频测试当作 FPGA 视频通信证据。
