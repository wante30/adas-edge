# P2 C API 接入现有 AI Bridge（B-CAPI）

日期：2026-10-31
板端：鲁班猫 RK3566 / RKNN 2.3.2
默认模型：B（未改变）

## 1. 接入结果

- 新增模型注册项 `B-capi`：
  - label：`B-CAPI · B05N P2 Raw-head INT8 (C API)`
  - model：`b05n_p2_multiclass_capi.rknn`（指向原 B 模型的符号链接）
  - backend：`rknn_capi`
- 原有 A/B/C/D 和 19 个历史方案全部保留，`DEFAULT_MODEL_ID` 仍为 `B`。
- Qt 模型列表由 `model_selection.options` 动态生成；状态中已出现 `B-capi available=True`，Qt 无需改 QML。
- 模型热切换复用原 `MVM1`/`MMS1` 协议；B → B-CAPI → B 连续 3 轮均为 `APPLIED`。

## 2. 热切换实测

| 切换 | 结果 | 耗时 |
|---|---|---|
| B → B-CAPI 第 1 次 | APPLIED | 113.7 ms |
| B-CAPI → B 第 1 次 | APPLIED | 1050.9 ms |
| B → B-CAPI 第 2 次 | APPLIED | 604.4 ms |
| B-CAPI → B 第 2 次 | APPLIED | 1027.0 ms |
| B → B-CAPI 第 3 次 | APPLIED | 293.1 ms |
| B-CAPI → B 第 3 次 | APPLIED | 1390.1 ms |

切到 B-CAPI 后 `main_detector.capi is not None == True`，确认实际走 C API，不是仅改界面标签。切回 B 后恢复 Python RKNN Lite 检测器。

## 3. AI Bridge 本地 60 帧回放

同一批 60 帧、同一 `b05p_competition_bridge` 链路、10 Hz 请求：

| 指标 | B（Python RKNN Lite） | B-CAPI（C API） |
|---|---|---|
| 实际结果 FPS | 8.50 | **9.22** |
| 分析帧 / 请求 | 153 / 180 | 166 / 180 |
| latest-only 替换 | 33 | **14** |
| 主模型总耗时 p50 / p95 | 61.0 / 203.0 ms | **41.5 / 162.6 ms** |
| RKNN 调用 p50 / p95 | 46.9 / 166.1 ms | **25.4 / 52.8 ms** |
| 检测解码/NMS p50 / p95 | 10.9 / 19.4 ms | **2.3 / 3.4 ms** |
| 完整 pipeline p50 / p95 | 62.7 / 207.4 ms | **42.3 / 163.4 ms** |
| 推理异常 | 0 | 0 |

说明：AI Bridge 回放 FPS 受 10 Hz 请求和 Qt/HUD 负载限制，不能拿独立 C API 的 14.33 FPS 当联调结果。C API 把主模型 p50 从 61.0 ms 降到 41.5 ms，解码从 10.9 ms 降到 2.3 ms。

## 4. 检测与业务逻辑

- B-CAPI 使用与原 B 完全相同的 RKNN 模型、512×288 输入、BDD10 类别和 conf/NMS 阈值（0.25 / 0.45）。
- 单帧对比：B-CAPI 返回 11 个框，包含 car/person/rider/truck/traffic light；原 B 返回 8 个框（base 检测器跳过 traffic light）。高置信度 car 分数基本一致（0.92 vs 0.924）。
- C API 的 NMS 实现与原 Python 解码略有差异，会多出少量小框；没有类别错位或坐标错位。
- 切换后原有交通灯、危险预警逻辑继续消费同一 items 结构；3 轮切换和帧请求均无异常，`inference_errors=0`。
- 未修改结果包格式。

## 5. 修改文件

板端 `/home/cat/qt_gpu_probe_ai_20261019/`：

- `model_profiles.py`：新增 `backend` 字段和 `B-capi` profile。
- `b05p_competition_bridge.py`：新增 C API 分支、按模型文件名选择后端、修正 P2 class profile 判断。
- `ai_bridge.py`：仅把热切换的 `same_main` 从 `resolve()` 比较改为 `absolute()` 比较，避免符号链接被误判为同一模型。
- 新增 `capi_detector.py`：ctypes 持久化 C API 检测器。
- 新增 `libp2capi.so`：P2 Raw-head + NMS 的 C 共享库。
- 新增符号链接 `/home/cat/cityscapes_arch_compare_20261027/models/b05n_p2_multiclass_capi.rknn`。

本地记录：`deployment/capi_bridge/`，板端备份：`/home/cat/b05n_p2_capi_integration_20261031/backup/`。

## 6. 回退

```bash
cp /home/cat/b05n_p2_capi_integration_20261031/backup/ai_bridge.py \
   /home/cat/b05n_p2_capi_integration_20261031/backup/model_profiles.py \
   /home/cat/b05n_p2_capi_integration_20261031/backup/b05p_competition_bridge.py \
   /home/cat/qt_gpu_probe_ai_20261019/
rm -f /home/cat/qt_gpu_probe_ai_20261019/capi_detector.py \
      /home/cat/qt_gpu_probe_ai_20261019/libp2capi.so \
      /home/cat/cityscapes_arch_compare_20261027/models/b05n_p2_multiclass_capi.rknn
sudo systemctl restart adas-ai-bridge-live.service
```

## 7. 安全

- 未覆盖原 B05N P2 RKNN 文件；B-CAPI 使用符号链接。
- 未修改 Qt、FPGA/PCIe、HUD、内核或系统组件。
- 未升级 RKNN Runtime。
- 正式服务启动参数仍是原始 B 模型，默认 B 未改变；测试结束已恢复服务并确认 `active`。
- 没有接入 B-2a/Seg-256/车道分割，未新增其它功能。
