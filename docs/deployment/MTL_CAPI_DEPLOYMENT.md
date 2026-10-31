# B-2a / Seg-256 / C API 板端部署

更新日期：2026-10-31
板端：LubanCat RK3566，RKNN Runtime / Lite2 2.3.2，RKNPU driver 0.8.2
默认模型：B05N P2 Raw-head INT8（B），未改变
范围：C API 共享库、模型注册、AI Bridge 热切换、Qt 分割通道、回放测速。

> 本文件是部署说明；研究背景、训练与消融数据见
> [`docs/research/MTL_RESEARCH_SUMMARY.md`](../research/MTL_RESEARCH_SUMMARY.md)。

---

## 1. 产物与目录

| 产物 | 仓库路径 | 板端路径 |
|---|---|---|
| C API 源码 | `deployment/lubancat/capi/p2capi.c` | 同左 |
| C API 头文件 | `deployment/lubancat/capi/include/rknn_api.h` | 同左 |
| 编译脚本 | `deployment/lubancat/capi/Makefile` | 同左 |
| ctypes 封装 | `deployment/lubancat/qt_gpu_probe/capi_detector.py` | `/home/cat/qt_gpu_probe_ai_20261019/capi_detector.py` |
| 模型注册 | `deployment/lubancat/qt_gpu_probe/model_profiles.py` | 同左 |
| AI Bridge | `deployment/lubancat/qt_gpu_probe/ai_bridge.py` | 同左 |
| 桥接适配器 | `deployment/lubancat/qt_gpu_probe/b05p_competition_bridge.py` | 同左 |
| 共享库 | 编译产物，不入 Git | `/home/cat/qt_gpu_probe_ai_20261019/libp2capi.so` |
| MTL-FULL | `b05n_p2_mtl_b2a_int8.rknn` | `/home/cat/models/b05n_p2_mtl_b2a/` |
| MTL-FAST | `b05n_p2_mtl_b2a_seg256_int8.rknn` | `/home/cat/models/b05n_p2_mtl_b2a_seg256/` |
| B-CAPI | `b05n_p2_multiclass_capi.rknn`（符号链接） | `/home/cat/cityscapes_arch_compare_20261027/models/` |

模型权重、ONNX、RKNN 不进入普通 Git；SHA256 见
`artifacts/manifests/model_manifest.json`。

## 2. 编译

```bash
cd deployment/lubancat/capi
make CC=gcc
# gcc -O3 -fPIC -Wall -Wextra -Iinclude -shared p2capi.c -lrknnrt -lm -o libp2capi.so
```

部署：

```bash
cp libp2capi.so /home/cat/qt_gpu_probe_ai_20261019/
cp capi_detector.py model_profiles.py ai_bridge.py \
   b05p_competition_bridge.py /home/cat/qt_gpu_probe_ai_20261019/
sudo systemctl restart adas-ai-bridge-live.service
```

依赖：板端 `librknnrt.so`（2.3.2）、`libm`。不要替换系统 RKNN Runtime。

## 3. 模型注册

`model_profiles.py` 中的关键字段：

```python
ModelProfile(
    id="MTL-FULL",
    label="B-2a · Full Resolution MTL (C API)",
    main_model=Path("/home/cat/models/b05n_p2_mtl_b2a/b05n_p2_mtl_b2a_int8.rknn"),
    specialist_model=None,
    input_size=(512, 288),
    class_profile="bdd10",
    backend="rknn_capi",
    task="mtl",
)
```

- `backend`：`rknn_lite` / `rknn_capi`
- `task`：`detect` / `mtl`
- 当前共 21 个方案；`DEFAULT_MODEL_ID = "B"`。
- 选择检测器由 profile 的 backend/task 决定，不依赖文件名。
- `same_main` 同时比较 `absolute()` 路径、backend、task；符号链接不会被误判为同一个模型。

## 4. C API 接口

```c
void *p2_create(const char *model_path);
int   p2_detect(void *handle, const uint8_t *rgb, P2Result *out);
int   p2_detect_mtl(void *handle, const uint8_t *rgb, P2Result *out,
                    uint8_t *masks);
void  p2_seg_size(void *handle, int *h, int *w);
void  p2_destroy(void *handle);
```

- 输入：512×288×3 uint8 NHWC RGB，已 letterbox。
- 输出：最多 1000 框；`masks` 为 3×H×W uint8。
- 每帧不启动新进程；模型只加载一次。
- `want_float=0`，直接读取 INT8。
- 分割三头为 per-tensor affine；同像素跨通道 int8 argmax 与反量化 argmax 等价（60 帧、8.85M 像素一致率 1.000000）。

完整接口和类别顺序见 `deployment/lubancat/capi/README.md`。

## 5. AI Bridge 选择逻辑

```text
MVM1 请求 → 查 profile
  → backend == rknn_lite   → Python RknnConsoleDetector
  → backend == rknn_capi
       task == detect     → CapiDetector
       task == mtl        → CapiMtlDetector
```

- 切换时先创建新 detector；失败则关闭新资源，保留旧 detector，返回 `FAILED`，服务不崩溃。
- 成功后交换 `pipeline.main_detector`，调用 `pipeline.reset()`，再关闭旧 detector。
- 同一时刻只保留一个主模型。
- `single_model` 由 profile 是否带 specialist 决定，不再沿用启动时的 scheme。

### 5.1 类别归一化

C API 结果与 Python 适配器使用同一套业务类别：

- `pedestrian → person`
- `traffic sign` 丢弃
- `motor/motorcycle → motorcycle`
- `bike/bicycle → bicycle`
- motorcycle/bicycle 置信度 < 0.85 丢弃
- 仅保留 `COMPETITION_CLASSES`

这保证交通灯、行人危险预警继续消费同一 items 结构。

### 5.2 MVA2 状态包

模型选项列表不能每帧原样放进 4096 字节 JSON。实现：

- `ModelProfile.wire()` 只发送 Qt 需要的 6 个字段。
- 初始/脏标记时发送 options；之后每 50 个结果包重发一次。
- 编码超限时自动回退到无 options 的紧凑状态，保证推理不中断。

## 6. 分割通道

- 写入 `/dev/shm/adas_mtl_overlay.png`（RGBA，native mask 尺寸）。
- 元数据 `/dev/shm/adas_mtl_overlay.json`：`frame_id`、`timestamp_ns`、`mask_width/height`、`display_width/height`、`generation`、`model_path`。
- 先写 `.tmp`，再 `os.replace` 原子替换。
- PNG 编码在后台线程；检测主路径增加 `overlay_ms=0`。
- 关闭 MTL detector 时删除两个 shm 文件；切回 B/B-CAPI 后 Qt 覆盖层立即隐藏。

## 7. Qt 显示

`deployment/lubancat/qt_gpu_probe/main.qml`：

- `Image` 仅在 `video.aiModelId` 为 `MTL-FULL` / `MTL-FAST` 且 `video.aiOnline` 时可见。
- 160 ms Timer，`file:///dev/shm/adas_mtl_overlay.png#g=<time>`。
- **必须用 URL fragment**；Qt 对带 `?query` 的 file URL 会显示空白。
- 四路布局覆盖层跟随 `root.roadQuadrant`；viewMode=1 时只占对应象限。
- 颜色：斑马线洋红、direct 绿、alternative 蓝、实线红、虚线青。

详见 `docs/hmi/MTL_SEGMENTATION_OVERLAY.md`。

## 8. 构建 Qt

板端源码目录 `deployment/lubancat/qt_gpu_probe/`；构建目录示例：

```bash
cd /home/cat/qt_gpu_probe_build_20261030_time
cp /path/to/main.qml .
/usr/lib/qt5/bin/rcc -name qml qml.qrc -o qrc_qml.cpp
make -j2 qt_gpu_probe
```

生成物 `qrc_qml.cpp`、`.o`、`qt_gpu_probe` 不进入普通 Git；源码中的
`main.qml` 与 `qml.qrc` 足以重建。

## 9. 运行与切换

```bash
# 默认 B
sudo systemctl start adas-ai-bridge-live.service

# 发一条切换请求
python3 deployment/lubancat/qt_gpu_probe/tools/mtl/switch_cli.py MTL-FAST 7001
```

Qt 模型菜单与 `switch_cli.py` 使用同一 `MVM1/MMS1` 协议。

## 10. 实测

### 10.1 完整 AI Bridge 回放

条件：本地 60 帧、10 Hz 请求、Qt/HUD 同时运行、正式安装文件。

| 模型 | 结果 FPS | 主模型 p50 / p95 | RKNN p50 / p95 | 解码 p50 / p95 |
|---|---:|---:|---:|---:|
| B | 8.56 | 66.2 / 209.3 ms | 50.2 / 167.4 ms | 9.5 / 16.2 ms |
| B-CAPI | **9.39** | **44.0 / 138.7 ms** | **23.0 / 53.2 ms** | **2.3 / 3.1 ms** |
| MTL-FULL | 5.00 | 152.9 / 412.4 ms | 78.3 / 170.5 ms | 23.5 / 59.5 ms |
| MTL-FAST | 6.78 | 104.4 / 248.5 ms | 72.3 / 173.1 ms | 6.3 / 7.2 ms |

`inference_errors=0`。不要把该表写成 PCIe 实时整机性能。

### 10.2 独立 C API

| 模型 | RKNN p50 | 单帧 p50 | FPS |
|---|---:|---:|---:|
| B05N P2 | 23.9 ms | 49.0 ms | 14.33 |
| B-2a 原版 | 87.6 ms | 161.2 ms | 4.95 |
| B-2a Seg-256 | 80.5 ms | 122.3 ms | 5.97 |

### 10.3 切换

| 切换 | 结果 | 耗时 |
|---|---|---:|
| B → B-CAPI | APPLIED | 204.7 ms |
| B-CAPI → MTL-FULL | APPLIED | 296.0 ms |
| MTL-FULL → MTL-FAST | APPLIED | 288.0 ms |
| MTL-FAST → B | APPLIED | 1028.3 ms |

### 10.4 Qt

本地 PGV 回放 + llvmpipe：scene_graph_fps p50 4.2（B）/ 3.9（MTL-FAST）；覆盖层开销在噪声范围内。截图见 `docs/hmi/assets/mtl/`。

## 11. 回退

```bash
# 停服务，恢复备份的板端文件，再启动
sudo systemctl stop adas-ai-bridge-live.service
# 备份目录由部署时创建，例如
# /home/cat/b05n_p2_capi_integration_20261031/backup/mtl_stage/
sudo systemctl start adas-ai-bridge-live.service
systemctl is-active adas-ai-bridge-live.service
```

正式默认 B 不因回退/同步改变。

## 12. 证据

- `research/b05n_p2_mtl/deployment/stageB2a/capi/capi_report.md`
- `research/b05n_p2_mtl/deployment/stageB2a/capi/capi_bc_report.md`
- `research/b05n_p2_mtl/deployment/stageB2a/board_bench_20261031/`
- `docs/deployment/evidence/MTL/bench_prod_*.json`
- `docs/deployment/evidence/MTL/prod_switch.log`
- `docs/deployment/evidence/MTL/mtl_switch3.log`
- `docs/deployment/evidence/MTL/qt_metrics_mtl.jsonl`
