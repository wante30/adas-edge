# ADAS 下午稳定版快照（2026-11-01）

本目录保存 2026-11-01 下午在 LubaNcat 板端验证通过的 ADAS 运行版本，用于永久归档和恢复。快照来源于以下运行时：

- HMI：`/home/cat/hmi_recover_20261101/qt_gpu_probe_build_20261030_time`
- AI：`/home/cat/adas_afternoon_20261101/adas_hazard_scene_20261031`
- 备份：`/home/cat/backups/pre_mode_fix_20261101/{ai,hmi}.tgz`

## 验证结果

截图 `adas_afternoon_hmi_20261101.png` 对应本次归档时的实际运行画面：

- AI 推理：约 `7.7 FPS`
- PCIe 输入：约 `55 FPS`
- 导航 OCR：有效，约 `1.0-1.2 s` 一次；画面示例为 `2.2 km 右转`
- HMI：MTL 分割、目标框、LVT1、Tesla BEV、危险事件面板正常
- LVT1：真实数据，`simulated:false`
- AI 环境：`ADAS_DISABLE_ROAD_SCENE=1`、`ADAS_LEGACY_HAZARD=1`、`ADAS_AUTO_BRAKE=1`
- OCR 参数：`--ocr-hz 2 --navigation-ocr`

## 归档内容

```text
archives/adas_afternoon_ai_source_20261101.tgz
archives/adas_afternoon_hmi_runtime_20261101.tgz
adas_afternoon_hmi_20261101.png
scripts/run_afternoon_ai.sh
scripts/vehicle_status_mmio.py
scripts/start-pcie-hmi-gui.sh.reference
autostart/adas-pcie-hmi.desktop.reference
systemd/adas-hazard-scene-live.service.reference
MANIFEST.json
SHA256SUMS
```

`systemd/adas-hazard-scene-live.service.reference` 是当时板上的服务文件，但它的 OCR 频率为 `0.1 Hz`，不是本次验证版本。恢复此快照时使用 `scripts/run_afternoon_ai.sh`，它固定为 `2 Hz`。

## 恢复

以下路径以板端恢复为例。

```sh
mkdir -p /home/cat/adas_afternoon_20261101/adas_hazard_scene_20261031
tar -xzf archives/adas_afternoon_ai_source_20261101.tgz \
  -C /home/cat/adas_afternoon_20261101/adas_hazard_scene_20261031

mkdir -p /home/cat/hmi_afternoon_20261101
tar -xzf archives/adas_afternoon_hmi_runtime_20261101.tgz \
  -C /home/cat/hmi_afternoon_20261101
chmod +x /home/cat/hmi_afternoon_20261101/qt_gpu_probe

# 先启动 LVT1 真实状态生产器
nohup python3 scripts/vehicle_status_mmio.py >/tmp/vehicle_status_mmio.log 2>&1 &

# 启动 AI
ADAS_ROOT=/home/cat/adas_afternoon_20261101/adas_hazard_scene_20261031 \
  sh scripts/run_afternoon_ai.sh

# 启动 HMI（参考 scripts/start-pcie-hmi-gui.sh.reference，
# 将其中的 RUN 改为 /home/cat/hmi_afternoon_20261101）
```

模型文件不在仓库中。默认路径为：

```text
/home/cat/models/b05n_p2_mtl_b2a_seg256/b05n_p2_mtl_b2a_seg256_int8.rknn
```

模型存在后才能启动 AI；可用 `ADAS_MAIN_MODEL` 覆盖。

## 完整性

```sh
sha256sum -c SHA256SUMS
```

原始板端备份哈希：

```text
ec1b270514ea4709d943adb42aea800ebf11d7e7751dcfffeacb578a017537c1  ai.tgz
f6db5e296be24b1b03934a0af6b59cf502a839270c127f77aef43312a1644c99  hmi.tgz
```

## 注意

- 本快照保留自动刹车配置 `ADAS_AUTO_BRAKE=1`。恢复前确认车辆处于安全测试环境。
- `ADAS_LEGACY_HAZARD=1` 使用旧 HazardEvents，避免 None road 触发误报。
- HMI 二进制依赖板端 Qt/Mali 环境；不要在其他架构直接运行。
- 本快照没有压缩模型、日志和实验产物；它们不属于运行时源码。
