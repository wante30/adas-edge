# RK3566 独立调试运行与回退

## 当前状态

基线已从实机正在运行的 AI issue9 和 Qt issue10 导出。新增程序位于独立目录，不替换自启动。新多灯功能默认关闭，使用 `--multi-signals` 开启。最新箭头后端优先验证功能；暂未达到整机性能和全部比赛场景验收要求。

## 打包与离线运行

在仓库根目录运行：

```powershell
python tools/package_multi_signals.py --output outputs/arm-bundle
scp outputs/arm-bundle/adas-multi-signals-arm-debug.tgz cat@192.168.137.2:/dev/shm/
```

板上根分区空间不足，优先在 `/dev/shm` 解压临时调试包，不向原部署目录覆盖。

```sh
cd /dev/shm
tar -xzf adas-multi-signals-arm-debug.tgz
cd adas-multi-signals-debug
python3 tools/check_signal_backend.py --fixtures tests/fixtures/real_signals --output /dev/shm/signal-regressions.json
```

该检查只使用真实照片裁剪，既不加载检测模型，也不读取或写入硬件。报告中的通过率是回归测试结果，不是独立数据集准确率。文件以独占创建方式写出；再次运行时选择新的输出文件。

本次交付的运行包额外包含实机已编译 Qt 和已有 C API 库，其 SHA-256 纳入 bundle_manifest.json；原机 RKNN Runtime、系统 Qt/Mali 库和 RKNN 模型仍为外部依赖。运行包中 Qt 可直接使用 `build/qt_gpu_probe`。重新制作相同类别运行包时，可给打包工具添加 `--qt-binary` 和 `--capi-library` 路径；工具检查为 AArch64 ELF。

## Qt 编译

板上已有 Qt5、qmake、OpenCV、NumPy。Python 3.8.10 和 OpenCV 4.10.0 的真实 ROI 回归已运行。模型推理另需原机安装的 RKNN Runtime 和模型文件，包内不重新分发。

```sh
mkdir build
cd build
qmake ../hmi/qt_gpu_probe.pro
make -j2
```

运行完整 AI 时，C API 库需要来自已验证部署的 `fast_mosaic/capi/libp2capi_generic.so`，或按该目录 Makefile 构建；先检查目标文件，勿覆盖现有部署。模型路径由 `ai/model_profiles.py` 使用原机安装路径。

## 接入独立回放

`tools/board_multi_signal_ab.py` 已用于同视频 A/B。旧版 SIGSTOP 恢复会让 shell 的 wait 返回并额外拉起 Qt，已移除此机制。当前脚本发现任何现存 HMI/AI 就在写出回放或创建测试目录前失败；必须由维护者通过正常生命周期流程准备空闲台架，脚本不暂停、不恢复正式进程。源码中的 DRY_RUN、硬件授权 0、自动制动 0、关闭自动 HUD 发送和独立 UDP 端口必须保留。

```sh
python3 tools/board_multi_signal_ab.py --root /home/cat/adas_multi_signals_20261111 --video /home/cat/adas_multi_signals_20261111/real_red_112454.mp4 --model FAST-Q4 --label manual_replay --seconds 30
```

最新用户要求先完成功能，因此未继续进行新版形态模板的整机性能优化或验收。早期 A/B 与新版算法不可混为同一版本性能结果。

桥接器扩展参数：

```sh
--multi-signals --signal-attribute-hz 2 --signal-jsonl /dev/shm/signals.jsonl --signal-failure-dir /dev/shm/signal-failures
```

这些参数添加到独立 AI 进程，其他实机参数保持已核对的捕获布局和模型配置。不要启动第二个进程占用正式进程正在使用的 UDP 端口；隔离回放使用脚本中的 38360/38361。物理前向必须 Q4，支持 front/quad 输入；full 布局禁用新增信号路径。

Qt 的 SIGNAL DEBUG 按钮打开逐灯 ID、颜色、形状、分数、结果年龄和原因。VISIBLE 表示画面箭头描述，EGO 表示已有本车道路证据的关联结果。没有证据时 EGO 为 UNKNOWN。普通圆灯不能按画面位置分配方向。

## 回退

当前正式启动脚本及 FPGA 控制路径未修改，停止独立调试程序即可回到原有运行。当前 A/B 脚本只结束本次创建的测试子进程，并生成清理收据 `restored.json`。旧收据不能证明单实例恢复；本次最后检查发现 6 个正式 Qt 实例，已结束这些实例并由原启动器重新拉起一个，详见 `evidence/lifecycle_recovery.json`，正式文件哈希未变。

如需源码回退，创建新工作目录读取已核对基线标签，不在脏工作区 reset：

```sh
git worktree add ../adas-signals-baseline rollback/multi-signals-board-20261111
```

实机基线清单见 `board_baseline_20261111.json`。原 Qt 二进制及 AI 源码目录保留，不需要用本分支重建旧版本才能回退。基线标签冻结的是 AI/HMI 源码；RKNN 模型、系统库和原 Qt 二进制依赖原机现有资产，不应称为完整磁盘恢复镜像。

## 最后现场状态

最终核对时 PCIe 采集服务返回 `/dev/pango_video0: No such device`，设备 lspci 显示 rev ff，PGV 不再更新。正式 Qt 已恢复单实例，AI 服务存在，但不能据此宣称实时 PCIe 视频链路正常。没有重置 FPGA、重绑驱动或修改控制寄存器；这一硬件链路状态仍需现场排查。
