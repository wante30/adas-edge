# MTL / C API 板端联调脚本

这些脚本在 RK3566 板端运行，使用真实的 RKNN Runtime、AI Bridge 和
`/dev/shm` 分割通道。路径按本机实验目录写死，复现时按实际目录调整。

| 脚本 | 用途 |
|---|---|
| `tl_check.py` | 对比 base `RknnConsoleDetector`、桥接适配器、B-CAPI 的类别输出，定位交通灯/类别归一化差异 |
| `test_capi_standalone.py` | 不经过 AI Bridge，直接对比 B Python 与 B-CAPI 的检测结果和耗时 |
| `test_capi_switch.py` | B → B-CAPI → B 连续切换与性能冒烟 |
| `test_mtl_switch.py` | B → B-CAPI → MTL-FULL → MTL-FAST → B 切换、mask 尺寸、overlay 文件检查 |
| `switch_cli.py` | 向运行中的 AI Bridge 发送一条 MVM1 切换请求并打印 MMS1 结果 |
| `capture_overlay.py` | 单帧运行 MTL-FULL / MTL-FAST，保存 `/dev/shm/adas_mtl_overlay.png` 供检查 |

依赖：

- `/home/cat/competition_cockpit_port_20261018`（competition_ai / pipeline）
- `/home/cat/qt_gpu_probe_ai_20261019`（ai_bridge.py、adapter、libp2capi.so）
- `/home/cat/models/b05n_p2_mtl_b2a/` 与 `_seg256/`
- 本地回放 `/home/cat/b05n_mtl_b2a_bench_20261031/replay60/`

运行前停止正式 AI 服务，避免占用 NPU 和 UDP 端口：

```bash
sudo systemctl stop adas-ai-bridge-live.service
python3 test_mtl_switch.py
sudo systemctl start adas-ai-bridge-live.service
```

测试结束必须恢复服务并确认默认模型仍为 B。
