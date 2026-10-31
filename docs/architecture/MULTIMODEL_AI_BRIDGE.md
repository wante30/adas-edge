# 多模型 AI Bridge 架构

更新日期：2026-10-31
代码：`deployment/lubancat/qt_gpu_probe/`
默认模型：`B`（B05N P2 Raw-head INT8）

---

## 1. 目标

同一套 Qt 车机 / AI sidecar 中，运行时切换：

- 纯检测：A/B/C/D、B-CAPI、历史 YOLO11n / BDD；
- 多任务：MTL-FULL、MTL-FAST；
- 交通灯专模：C/D 与历史 specialist 方案；
- 回滚：legacy 模型。

不重启服务、不重启板、不替换模型文件、不改 MVA2 检测框协议。

## 2. 组件

```text
Qt HMI ──MVM1/MMS1(UDP 38260/38261)── AI Bridge
   │                                      │
   └── MFR1 共享帧 / PGVF ────────────────┘
                                          │
                          model_profiles.py (21 schemes)
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
     RknnConsoleDetector          CapiDetector                 CapiMtlDetector
     (Python RKNN Lite)           (C API, 12 outputs)          (C API, 15 outputs)
              │                           │                           │
         检测/专模/历史              B-CAPI                  MTL-FULL / MTL-FAST
                                                                      │
                                                          /dev/shm overlay PNG+JSON
                                                                      │
                                                                    Qt Image
```

## 3. 模型方案（21 项）

| id | 角色 | backend | task | specialist |
|---|---|---|---|---|
| B | B05N P2 十类 Raw-head INT8（默认） | rknn_lite | detect | - |
| B-capi | 同模型，C API | rknn_capi | detect | - |
| MTL-FULL | B-2a 原版 512×288 分割 | rknn_capi | mtl | - |
| MTL-FAST | B-2a Seg-256 144×256 分割 | rknn_capi | mtl | - |
| A | BDD-YOLOv8n Raw-head INT8 | rknn_lite | detect | - |
| C | BDD + B05L Specialist | rknn_lite | detect | ✓ |
| D | B05N P2 + B05L Specialist | rknn_lite | detect | ✓ |
| legacy-bdd-kl | 旧版 BDD KL INT8 | rknn_lite | detect | - |
| legacy-bdd-int8 | 旧版 BDD INT8 | rknn_lite | detect | - |
| legacy-yolo11n-320 | 旧版 YOLO11n 320×192 | rknn_lite | detect | - |
| legacy-yolo11n-416 | 旧版 YOLO11n 416×256 | rknn_lite | detect | - |
| legacy-yolo11n | 旧版 YOLO11n 512×288 | rknn_lite | detect | - |
| legacy-yolo11n-mmse | 旧版 YOLO11n MMSE | rknn_lite | detect | - |
| legacy-yolo11n-640 | 旧版 YOLO11n 640×352 | rknn_lite | detect | - |
| legacy-yolo11n-specialist | 旧版 YOLO11n + Specialist | rknn_lite | detect | ✓ |
| legacy-bdd-kl-specialist | 旧版 BDD KL + Specialist | rknn_lite | detect | ✓ |
| legacy-bdd-specialist-hybrid2 | 旧专模 hybrid2 | rknn_lite | detect | ✓ |
| legacy-bdd-specialist-logits | 旧专模 logits | rknn_lite | detect | ✓ |
| legacy-bdd-specialist-hybrid | 旧专模 hybrid | rknn_lite | detect | ✓ |
| legacy-bdd-specialist-int8-split | 旧专模 int8-split | rknn_lite | detect | ✓ |
| legacy-bdd-specialist-traincalib | 旧专模 traincalib | rknn_lite | detect | ✓ |

`DEFAULT_MODEL_ID = "B"`，`B05N_P2_FILENAME = b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn`。

## 4. profile 字段

```python
@dataclass(frozen=True)
class ModelProfile:
    id: str
    label: str
    main_model: Path
    specialist_model: Path | None
    input_size: tuple[int, int]
    class_profile: str
    legacy: bool = False
    backend: str = "rknn_lite"   # rknn_lite | rknn_capi
    task: str = "detect"        # detect | mtl
```

- `available`：模型文件存在才可选。
- `wire()`：发给 Qt 的紧凑选项（id/label/available/legacy/has_specialist/reason）。
- `public()`：本地/诊断用，额外含 backend/task。
- 后端选择按 `backend`/`task`，**不按文件名**；文件名只作为没有显式 backend 时的兼容回退。

## 5. 热切换协议

### 5.1 MVM1 请求

```text
magic  "MVM1"       4 B
request_id          uint32
id_len              uint8
profile_id          ASCII (1..48)
```

### 5.2 MMS1 回复

```text
magic  "MMS1"       4 B
request_id          uint32
state               uint8  (READY/SWITCHING/APPLIED/FAILED/REJECTED)
profile_id          变长
message             变长 UTF-8
```

### 5.3 状态机

```text
READY --MVM1--> SWITCHING
SWITCHING --成功--> APPLIED
SWITCHING --异常--> FAILED（原检测器继续可用）
未知/不可用方案 --> REJECTED
```

- `SWITCHING` 先回包，避免 Qt 等待超时。
- 切换在 worker 的安全点执行，不在检测中途销毁 context。
- 成功后 `pipeline.reset()`，旧模型 `close()`。
- 新模型加载失败时不交换 detector，服务继续用旧模型。

## 6. 检测器选择

`b05p_competition_bridge.py` 的 `CompetitionConsoleDetector`：

```python
if not specialist and backend == "rknn_capi":
    backend_class = CapiMtlDetector if task == "mtl" else CapiDetector
    self.capi = backend_class(...)
    self.detector = self.capi
    return
super().__init__(...)  # Python RKNN Lite
```

- `single_model` 由 profile 是否带 specialist 决定（`profile.specialist_model is None`）。
- B/B-CAPI/MTL 保留交通灯；C/D 继续用专模结果替换主模型交通灯。
- 类别归一化在 Python 适配器/C 封装中完成，业务层无感。

## 7. 结果包

MVA2 保持不变：

```text
header (MVA2, frame_id, timestamp_ns, count, flags, nav_dir, nav_dist, status_len)
boxes  (7 float + class_id + confidence) × count
status (UTF-8 JSON, <= 4096 B)
```

`model_selection` 字段：

```json
{
  "active_id": "MTL-FAST",
  "active_label": "B-2a · Seg-256 MTL (C API)",
  "requested_id": "",
  "state": "APPLIED",
  "message": "模型已切换",
  "options": [ ... 21 项，周期发送 ... ]
}
```

分割 mask **不进入 MVA2**；走 `/dev/shm`。

## 8. 调度与线程

- 请求 listener：UDP 38260，处理 MFR1/MVR1/MVR2/MVS1/MVM1。
- 推理 worker：latest-only 队列，10 Hz 主检测；hazard 5 Hz；specialist 1–3 Hz。
- 结果 publish：UDP 38261。
- MTL overlay writer：独立 daemon 线程，合并最新 mask，PNG 编码不阻塞检测。
- Qt 视频纹理线程与 AI worker 分离；AI 结果不会阻塞 EGL 提交。

## 9. 资源与故障

- 一个主模型 + 可选一个 specialist；切换成功后才释放旧模型。
- `same_main` 比较路径 + backend + task；符号链接/同文件不同后端不会复用。
- 切换失败返回 `FAILED`，`last_error` 记录异常，服务不退出。
- MTL detector `close()` 删除 `/dev/shm/adas_mtl_overlay.*`，避免旧分割残留。
- 4096 字节状态上限：紧凑 options + 周期重发 + 超限回退，保证推理不中断。

## 10. 代码索引

| 文件 | 作用 |
|---|---|
| `deployment/lubancat/qt_gpu_probe/model_profiles.py` | 21 个方案、MVM1/MMS1 编解码 |
| `deployment/lubancat/qt_gpu_probe/ai_bridge.py` | 初始化、热切换、状态包、调度 |
| `deployment/lubancat/qt_gpu_probe/b05p_competition_bridge.py` | 桥接适配器、后端选择、类别归一化 |
| `deployment/lubancat/qt_gpu_probe/capi_detector.py` | C API ctypes、MTL mask、overlay writer |
| `deployment/lubancat/capi/p2capi.c` | RKNN C API 推理 |
| `deployment/lubancat/qt_gpu_probe/main.qml` | Qt 模型菜单/分割覆盖层 |
| `deployment/lubancat/qt_gpu_probe/tools/mtl/` | 板端切换/回归脚本 |

## 11. 实测

完整回放（60 帧，10 Hz，Qt/HUD 同时运行）：

| 模型 | FPS | 主模型 p50/p95 |
|---|---:|---:|
| B | 8.56 | 66.2 / 209.3 ms |
| B-CAPI | 9.39 | 44.0 / 138.7 ms |
| MTL-FULL | 5.00 | 152.9 / 412.4 ms |
| MTL-FAST | 6.78 | 104.4 / 248.5 ms |

切换：B→B-CAPI 204.7 ms，B-CAPI→MTL-FULL 296.0 ms，MTL-FULL→MTL-FAST 288.0 ms，MTL-FAST→B 1028.3 ms；全部 `APPLIED`。

详见 `docs/deployment/MTL_CAPI_DEPLOYMENT.md`。
