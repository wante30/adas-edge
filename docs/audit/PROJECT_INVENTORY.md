# ADAS Edge 项目全量盘点

审计日期：2026-10-28  
审计范围：Windows 开发机、GitHub 仓库、RTX 4060 训练目录、LubanCat-1 RK3566 实板、当前工作区  
结论先行：**当前真正运行的是 Windows `live/stage2` 工作树 + RK3566 板端 `/home/cat/qt_gpu_probe_ai_20261019` 部署；GitHub 默认分支 `main` 在审计开始时仍停在旧的 `c321a89`，落后 77 个提交。** 本次已把 `live/stage2`、研究报告和审计文档整理为可同步状态，详见 `docs/audit/GIT_SYNC_REPORT.md`。

---

## 1. 可访问环境

| 环境 | 标识 | 访问方式 | 状态 | 说明 |
|---|---|---|---|---|
| macOS 工作区 | 本机 | 本地 shell | 可用 | 本次审计、文档整理和 Git 提交的主工作区 |
| Windows 开发机 | `DESKTOP-2HNBT2L` / `ssh rog` | 免密 SSH | 可用 | 主代码工作树、训练产物、模型转换环境 |
| RK3566 开发板 | `lubancat` / EmbedFire LubanCat-1 | 经 Windows 跳板 `ssh lubancat` | 可用 | 4×Cortex-A55、3.8 GiB、Ubuntu 20.04.6、内核 4.19.232 |
| RTX 4060 训练目录 | `D:\Datasets\adas_edge_training` | 经 Windows 访问 | 可用 | B05L/B05N/B05P 训练、评估、RKNN 产物 |
| FPGA 完整工程 | `D:/QiansaiFPGA/source_code_3/PCIE_timig_fix` | 未找到 | **不可用** | 交接文档记录的工程路径在本机不存在；无法核对 RTL/时序/资源 |
| 完整 BDD100K 70k 训练集 | 训练机/远端 | 未完整在本机 | 部分可用 | 本机主要是 validation 10k 镜像和派生 split |

> 板端系统时钟未同步：2026-10-28 实际审计时板端 `date` 显示 `2024-06-18`。本文所有“板端时间”均不代表真实日历时间；以文件写入时间、Git 提交时间和审计日期为准。

---

## 2. Git 仓库与工作树

### 2.1 仓库

- 远程：`https://github.com/wante30/adas-edge`
- 默认分支：`main`
- 审计开始时远程 HEAD：`c321a89f26e5fa05a222ad29659e4148ec6c62d6`（`Publish FPGA competition software baseline`）
- 本地最新工作分支：`live/stage2`，HEAD `19f050e9cb2fffd248edcefc25449e954d42d007`
- `live/stage2` 相对 `origin/main`：**领先 77 个提交，落后 0 个提交**（可快进）

### 2.2 Windows 工作树

Windows 上不是多个独立仓库，而是**同一个 Git 仓库的 4 个 worktree**：

```text
C:/Users/JingyuWang/adas-edge       19f050e [live/stage2]                  ← 当前主工作树
C:/Users/JingyuWang/adas-edge-b01    6fd12b9 [research/rk3566-adas-models]
C:/Users/JingyuWang/adas-edge-b05n   6a2933b [research/rk3566-p2-multiclass]
C:/Users/JingyuWang/adas-edge-b05o   891978a [research/rk3566-b05o-video-scheduling]
```

`adas-edge` 工作树约 20.8 GB，其中 `outputs/` 11.6 GB、`.venv/` 5.9 GB、`.rknn-wsl-venv/` 1.3 GB、`tmp/` 617 MB、`deployment/` 571 MB（其中 Qt `results/` 482 MB）、`data/` 565 MB、`.deps/` 475 MB。**这些都不是应进入普通 Git 的开发缓存/实验结果。**

### 2.3 分支清单（审计开始时）

| 分支 | 本地/远程 | 指向 | 说明 |
|---|---|---|---|
| `main` | 远程默认 | `c321a89` | 旧基线，仅一个公开提交 |
| `live/stage2` | 仅本地 | `19f050e` | **最新完整工作线**，77 commits ahead of main |
| `codex/competition-software-closeout` | 本地/远程 | `2d96f13` / `b261e33` | 收口分支，本地与远程不同步 |
| `competition-adas-hmi-sim` | 仅本地 | `ba5e42f` | 3D ADAS cockpit 早期实验 |
| `competition-avm-matlab-fpga-freeze` | 仅本地 | `cdfb204` | AVM/Matlab/FPGA 冻结实验 |
| `competition-front-accuracy-sprint` | 本地=远程 | `0e1231a` | 前视强模型冲刺，结论保留 YOLO11n |
| `competition-rectangular-rk3566` | 本地=远程 | `13ad93f` | 矩形输入 + 交通灯专模 |
| `local-round2-full-history` | 仅本地 | `4d19014` | 含 575 MB zip、322 MB npy 等大文件历史，**不应推送** |
| `master` | 仅本地 | `fcc10b1` | 旧主线快照 |
| `research/rk3566-adas-models` | 本地=远程 | `6fd12b9` | B05L 之前研究线 |
| `research/rk3566-b05o-video-scheduling` | 本地领先/远程较旧 | `891978a` / `857ece9` | B05P 研究线，含未提交报告 |
| `research/rk3566-p2-multiclass` | 本地领先/远程较旧 | `6a2933b` / `82f2d57` | B05N P2 研究线，含未提交报告 |
| `restore/tesla-original` | 仅本地 | `b61309b` | Tesla 原版恢复实验 |
| `rk3566-hmi-high-fps` | 仅本地 | `d362da2` | HMI 高帧率实验 |
| `road-domain-pilot-round2` | 本地=远程 | `1ca14c2` | 道路域 pilot 第二轮 |

### 2.4 未提交修改（Windows `adas-edge` / `live/stage2`）

审计时 `git status` 显示 **约 42 个已跟踪文件修改 + 300 余个未跟踪文件**。已跟踪修改约 3,532 insertions / 548 deletions，核心包括：

- `deployment/lubancat/qt_gpu_probe/`：模型方案选择、AI bridge、HUD/PCIe、Qt 视频项、Tesla 场景。
- `deployment/lubancat/mali_video/`：旧 Mali 视频桥接和说明。
- `competition_ai/`、`src/adas_edge/`：检测、绑定信号、危险事件、Ultralytics 适配。
- `tools/adas_hmi_sim/`、`tools/competition_console/`：PC 仿真车机和原生测试台。
- `tests/`：新增约 25 个测试文件。
- `docs/`：2026-10-28 的 HMI 状态 checkpoint、Qt handoff、AVM 文档等。

**这些修改是本项目当前最新成果，已先做本地备份，再在本次审计中提交。**

### 2.5 备份

在修改前已生成以下备份（Windows `D:\Backup`，并复制关键件到 macOS `_backup/`）：

| 备份 | 大小 | 用途 |
|---|---:|---|
| `adas-edge-all-20261028.bundle` | 2,629,237,497 B | 全部本地分支历史（含大文件历史，仅本地保存） |
| `live-stage2-20261028.bundle` | 53,240,114 B | `live/stage2` 单分支历史，便于 Mac 审计 |
| `adas-src-20261028.tgz` | 153,357,121 B | 工作树源码快照，排除 outputs/venv/data/results |
| `adas-edge-tracked-20261028.patch` | 685,902 B | 审计时已跟踪文件的未提交 diff |

> 没有执行 `git reset --hard`、`git clean -fd`、`git push --force`。Windows 工作树保持原状，Mac 上的整理不会回写破坏板端服务。

---

## 3. 代码模块清单

| 模块 | 实际路径 | 主要功能 | 版本/状态 | 验证 |
|---|---|---|---|---|
| Qt ADAS HMI | `deployment/lubancat/qt_gpu_probe/` | 1280×720 Qt Quick 车机、PGVF 视频、Mali-G52 纹理、AI 框、Tesla BEV、HUD/PCIe 控制 | 当前板端运行中；源码与板端部分文件有漂移 | 559 tests pass；实板截图/日志 |
| AI sidecar | `deployment/lubancat/qt_gpu_probe/ai_bridge.py` | 读取 MFR1 共享帧、RKNN 推理、Raw-head 解码/NMS、危险事件、导航 OCR、MVA2 结果 | 板端运行中；hash `cf29b48e...` 与仓库一致 | 实板 summary、timing JSONL |
| 模型方案选择 | `model_profiles.py`、`b05p_competition_bridge.py`、`b05p_ai_bridge_launcher.py` | A/B/C/D 方案、MVM1 切换、Raw-head adapter | 板端运行 A；UI 截图曾显示 B | 板端进程参数、summary、代码 |
| 旧 Mali HMI | `deployment/lubancat/mali_video/` | Mali 视频 + 轻量 3D 车机、导航/危险/车道 | 历史/备选，不是当前主入口 | 单元测试/历史报告 |
| 比赛感知/决策 | `competition_ai/` | 检测、交通灯 HSV、危险事件、图像跟踪、场景策略 | 软件逻辑完成；部分实验性 | 559 tests pass |
| PCIe/协议 | `competition_io/`、`deployment/lubancat/qt_gpu_probe/pcie_*.py` | ADSF/ADSR 软件契约、HUD V1/V2、ADST v1、PGVF/MFR1/MVA2 | ADSF/ADSR 为 mock；HUD/PGVF/HSST 实板有证据 | 单元测试 + 实板读回 |
| RKNN 工具 | `deployment/rknn/`、`deployment/rknn_work/` | ONNX 导出、RKNN 转换、split/raw-head、校准、评测 | 历史/研究 | 报告 + 板端模型 |
| 训练工具 | `tools/b05*.py`、`b05l_work/`、`b05n_work/` | 数据构建、训练、评估、视频验证、计时 | 研究完成 | 报告 + JSON |
| PC 原生车机 | `tools/competition_console/` | Windows 1600×900 测试台、视频+ONNX、场景按键、HUD 软件接收 | 开发/验收工具，不是板端生产入口 | 127 项相关测试（历史） |
| PC 浏览器仿真 | `tools/adas_hmi_sim/` | WebSocket + Three.js/Maplibre 3D 仿真、四路 BEV | 研究/展示 | 单元测试 |
| AVM RTL | `rtl/avm_640/` | 四路 640×640 AVM 固定点 LUT 流式核 | 模块级仿真通过 | 409,600/409,600 像素，0 mismatch |
| FPGA 交接资料 | `deployment/lubancat/qt_gpu_probe/RK3566_PCIe_protocol_handoff.md` | PG2L100H 寄存器/时序/HSST/HUD/ISP 交接 | 文档 1.0.1 | 静态核对，非综合/时序 |

---

## 4. 模型与训练产物清单

| 资产 | 路径 | SHA256 | 状态 |
|---|---|---|---|
| BDD 主模型 FP32 | `D:\Datasets\adas_edge_training\b03b_pretrained_cache_20261021\bdd100k-yolov8n-best.pt` | `acd17780...72fb2` | PC 源权重 |
| BDD Raw-head ONNX | `...\rknn_b05o\bdd_yolov8n_512x288_raw_heads.onnx` | `d6ea1651...528adb` | 转换源 |
| BDD Raw-head INT8 RKNN | `...\rknn_b05o\bdd_yolov8n_512x288_raw_heads_int8.rknn` | `a86ab053...041afb` | **当前板端运行（A）** |
| B05N P2 FP32 | `...\b05n_final\b05n_p2_multiclass.pt` | `c249f7ff...e278ed` | PC 研究模型 |
| B05N P2 ONNX | `...\b05n_final\b05n_p2_multiclass.onnx` | `c30ca4d9...f22e75` | PC 参考 |
| B05N P2 Raw-head INT8 | `...\rknn_b05o\b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn` | `65089ef0...b2a5f` | 板端 B 方案，已验证 |
| B05L Specialist FP32 | `...\b05l_runs\v8n_p2\weights\best.pt` | `c2d8dd92...8776a7` | PC 专模 |
| B05L Raw-head INT8 | `...\rknn_b05o\b05l_v8n_p2_raw_heads_int8_500_20261026.rknn` | `c11b95b4...38b6f` | 板端 C/D 专模 |
| 旧 YOLO11n 320 split | `deployment/rknn/yolo11n_rk3566_int8_split_320.rknn` | `60f17e9b...c977da` | 历史 smoke，勿混用 |
| 旧 YOLO11n 416 split | `deployment/rknn/yolo11n_rk3566_int8_split_416.rknn` | `87dfd6d4...93bb43` | 历史 smoke，勿混用 |
| 旧生产 YOLO11n 512×288 | `/home/cat/adas-edge-rect/rect_models/yolo11n_rk3566_int8_split_512x288.rknn` | `7a697ccc...c50591` | 回滚候选 |
| 旧交通灯专模 | `/home/cat/adas-edge-rect/tl_check/tl_specialist_512x192_hybrid2.rknn` | `becfaada...34e6f8` | 回滚候选 |

完整清单见 `docs/models/MODEL_MANIFEST.md` 与 `artifacts/manifests/model_manifest.json`。

---

## 5. 当前板端运行状态（2026-10-28）

### 5.1 进程

| PID | 进程 | 说明 |
|---:|---|---|
| 784 | `/home/cat/pcie_test/pango_video_t05/pcie_video/app/pango_video_capture --continuous --fps 60 --publish-dir /dev/shm/pango-video` | FPGA PCIe 视频采集，约 49.5 FPS |
| 1054 | `python3 .../b05p_ai_bridge_launcher.py ... --main-model .../bdd_yolov8n_512x288_raw_heads_int8.rknn --main-hz 10 --hazard-hz 5` | AI sidecar，实际 A 方案 |
| 2875 | `.../qt_gpu_probe --fps 60 --duration 0 ... --pgv-file /dev/shm/pango-video/latest.pgv --ai-hz 10` | Qt HMI |
| 1332 | `python3 .../hud_bridge.py` | HUD V2 BAR1 bridge |

### 5.2 服务

| 服务 | 状态 | 说明 |
|---|---|---|
| `pango-video-capture.service` | active | 自动加载 `pango_video.ko`，启动采集 |
| `adas-ai-bridge-live.service` | active | drop-in `99-b05p-model-A.conf` + `zz-b05p-model-A-single.conf` |
| `adas-hud-bridge.service` | active | root，BAR1 HUD V2 |
| `qt_gpu_probe` | 非 systemd | 由 XFCE autostart `~/.config/autostart/adas-pcie-hmi.desktop` → `~/.local/bin/start-pcie-hmi-gui.sh` 启动 |

### 5.3 实板证据

```text
/dev/shm/pango-video/status.json:
  state=running, frames=445008, fps=49.492, error=""

lspci 01:00.0:
  0755:0755, Kernel driver in use: pango_video
  LnkCap: 5GT/s x4
  LnkSta: 5GT/s x1 (downgraded)

HUD BAR1:
  0x600 = 0x31445548 (HUD1)
  0x604 = 0x00010000
  0x630 = 0x000001ff
  0x64c applied seq = 0x00004b9d
  0x620 reject = 0, 0x624 last_error = 0

HSST lane mailbox:
  0x680 sequence 递增（约 30 Hz）
  0x684 payload 变化，最近样本 0xf50101 / 0xfa0102 / 0xfa0202
  0x688 = 0x1 (valid, not stale)
  0x68c = 0x4c4d0001 (protocol id)
```

`0x684` 按交接文档解码为左右边界数量与最近类型（0 未知、1 实线、2 虚线）；这说明 FPGA 的 HSST 车道邮箱**确实在实时更新**。但 HSST 背后的算法是否为 MobileNet、输出坐标/稳定性如何，当前可访问资料没有说明，**不能据此宣称“MobileNet 车道识别已验证”**。

### 5.4 当前 HMI 实测

截图：`docs/hmi/assets/current_live_20261028.png`（1280×720，真实板端 X11 根窗口）。

- HMI 自报：**56 FPS**，PCIe 在线，AI 10.0 FPS，NPU 100%，GPU 25%。
- AI summary（`adas_A_live_default_summary.json`）：
  - 主模型结果率：9.9756 Hz
  - 主 RKNN 调用 p50：29.15 ms，p95：34.41 ms
  - 主模型总耗时 p50：38.77 ms，p95：47.84 ms
  - 全 AI 工作 p50：49.79 ms，p95：67.34 ms
  - 进程 CPU：69.48% of one core / 17.37% of machine
  - RSS：144.06 MB，峰值 174.02 MB
  - NPU：100% @ 600 MHz；GPU：mean 33.9%，p50 35%
  - SoC 温度：35.6–40.0 °C
- 视频采集：49.492 FPS。
- 当前 HMI 显示：信号灯“未识别”、导航“暂不可读”、危险事件“未检测到”、车道状态“车道视觉未接入”。

---

## 6. 已同步 / 未同步判断

| 内容 | 审计开始状态 | 本次处理 |
|---|---|---|
| `live/stage2` 77 个提交 | 仅 Windows 本地 | 推送到 GitHub 分支并合入 `main`（见 sync report） |
| 当前工作树未提交源码 | 仅 Windows | 提交到 `live/stage2` |
| B05N/B05L/B05P 报告 | 分散在 worktree/本地 | 归档到 `docs/models/` 与 evidence |
| 模型权重 | 本地大文件 | 不入普通 Git；只提交 manifest/SHA256 |
| 板端运行配置 | 板端 `/home/cat/...` | 文档化到 `docs/deployment/`；不改板端 |
| FPGA 完整工程 | 另一台机器，不可访问 | 标记 BLOCKED，未编造 |
| 四路独立 PCIe 流 | 未验证 | 明确写成“单路 2×2 mosaic”，不冒充四路独立 DMA |
| Windows 上位机 | PC 测试台/网页调试台 | 文档化，不冒充板端生产 HMI |

---

## 7. 缺失与风险

1. **FPGA 完整工程不可访问**：无法生成 PG2L100H 的资源、时序、bitstream hash；只能引用交接文档和板端寄存器证据。
2. **MobileNet 车道识别未证实**：只有 HSST 邮箱活数据；算法名、模型文件、坐标输出、训练数据均无证据。
3. **四路输入边界**：FPGA 给 RK 的是一路 1280×720 2×2 mosaic；无法从 RK 侧证明四路独立采集/DMA。
4. **训练集不完整**：BDD 主模型训练配置/完整 70k 数据不在本机；B05N/B05L 主要用 8,513 张 train + 482 val + 1,000 test。
5. **量化精度仍有差距**：BDD Raw-head INT8 在 800 张上 mAP50 0.3606 vs ONNX 0.3688；Cityscapes 独立集 A 仅 0.2822。
6. **当前 HMI 模型标签不一致**：进程参数与 summary 为 A，截图右上角显示“方案 B”；需修 UI 状态回读。
7. **板端源码快照有漂移**：`main.qml`、`video_item.cpp` 的板端磁盘文件与仓库不同；运行二进制 hash 已记录，但不能声称字节级可复现。
8. **导航 OCR 不可用**：当前 OCR p50 约 1.2 s，结果 `valid=0`，HMI 显示“导航暂不可读”。
9. **红绿灯颜色/归属未验证**：当前画面“未识别”；HSV 规则与专模存在，但未在比赛域完成稳定验收。
10. **板端时钟未同步**：所有板端时间戳需换算，不能直接当真实日期。

---

## 8. 相关文档

- `docs/architecture/SYSTEM_ARCHITECTURE.md`
- `docs/architecture/PCIE_DATA_FLOW.md`
- `docs/architecture/FPGA_RK3566_INTERFACE.md`
- `docs/models/TRAINING_REPORT.md`
- `docs/models/MODEL_COMPARISON.md`
- `docs/models/DATASET_AND_EVALUATION.md`
- `docs/models/MODEL_MANIFEST.md`
- `docs/deployment/RKNN_QUANTIZATION_REPORT.md`
- `docs/deployment/RK3566_BENCHMARK.md`
- `docs/deployment/DEPLOYMENT_GUIDE.md`
- `docs/hmi/TESLA_ADAS_HMI.md`
- `docs/hmi/HMI_PERFORMANCE.md`
- `docs/roadmap/ADAS_EVENT_ENGINE_PLAN.md`
- `docs/competition/REQUIREMENT_MATRIX.md`
