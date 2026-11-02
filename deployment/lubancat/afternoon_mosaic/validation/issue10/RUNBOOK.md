# Issue #10 运行与回退

当前板端独立安装：`/home/cat/adas_issue10_20261102/`。默认 Q4、FAST-Q4、本地回放、空间视图；原系统自启动与文件未替换。已有模型以及 `/home/cat/qt_gpu_probe_ai_20261019/local_quad_full.rgb` 由原板端保留，不重复打包。

下列命令在 RK3566 Linux shell 执行。先确认当前没有另一个本次回放会话；不要同时启动两个播放器占用相同 UDP 端口。

## 启动独立修复版

```sh
python3 /home/cat/adas_issue10_20261102/candidate/tools/start_bev_local.py
```

默认把录像前 600 帧原样缓存到 `/dev/shm/adas_bev_runtime/replay_600.rgb`，需要约 1.1GB RAM，首次准备可能需要数十秒。内容为 10 秒循环，不是新道路采集。代码和二进制在独立安装目录，帧槽及运行日志在 RAM；不发送硬件制动，不伪造 ACK。

内存不足时可使用 `--no-cache` 直接读原录像，但 eMMC 会显著限制回放推进速度。

## 运行旧版对照

先结束当前回放，再运行：

```sh
python3 /home/cat/adas_issue10_20261102/candidate/tools/start_bev_local.py --baseline
```

包内 baseline 保留 PR #8 的原运动策略，仅含验证诊断与共同 QML 激活修复。它用于本地对照，不冒充原发布二进制的逐字节副本。

## 恢复开发前会话

```sh
python3 /home/cat/adas_issue10_20261102/candidate/tools/restore_bev_previous.py
```

脚本停止本次回放并核对原进程命令行后恢复保留的旧会话。原 PID 仅对同一次启动有效；跨重启时不会复用旧 PID，原自启动配置本来就未修改。当前没有真实 PCIe，恢复旧会话不代表 PCIe 或 FPGA 硬件已恢复验收。

## 从运行包安装

解压 `adas-issue10-rk3566-runtime-20261103.tar.gz` 到一个新目录即可，包内顶层为 `adas_issue10_rk3566/`，含 candidate、baseline_src、runtime_manifest.json 与恢复状态快照。不要覆盖现有目录。可在解压目录执行 `python3 candidate/tools/start_bev_local.py`；若 RAM workspace 已指向另一份安装，先停止旧回放，使用原安装入口或在重启后的空 RAM workspace 启动新安装。

模型、RKNN Runtime、Mali/Qt 动态库沿用当前板端安装；包不是系统镜像，也不包含模型权重。恢复状态快照绑定原板端 boot_id，移到其他设备不会向旧 PID 发送信号。

## 重现验证

源码工具：`tools/instrument_bev.py` 给 PR #8 源码加入可选诊断；`run_bev_ab.py` 跑真实 RKNN/Qt；`bev_replay_main.cpp` 播放记录的真实测量；`run_bev_player.py` 录屏；`analyze_bev_ab.py` 统计轨迹；`summarize_bev_evidence.py` 从 archive 还原性能和原始 AI epoch/CSV。

真实轨迹与 CONTROLLED METADATA FIXTURE 分开验收，不能把 fixture 写成道路检测或 FPGA 测试。