# GitHub 同步报告

日期：2026-10-28  
仓库：<https://github.com/wante30/adas-edge>  
执行人：OpenCode / Codex 审计会话  
原则：非破坏性同步；未使用 `git push --force`、`git reset --hard`、`git clean -fd`。

---

## 1. 同步结果

| 项目 | 值 |
|---|---|
| 远程仓库 | `https://github.com/wante30/adas-edge` |
| 审计开始时远程 `main` | `c321a89f26e5fa05a222ad29659e4148ec6c62d6` |
| 同步分支 | `live/stage2` |
| 审计快照 commit | `659223faf7088f1f88a82f65fa29769f59b3b3d4` |
| 合并前分支 HEAD | `4fb4d0c6f8d5b2f7c51a2058b57618949b04ca92`（快照 + sync report） |
| 分支相对 main | 领先 78 个提交（77 个历史 + 2 个审计/文档提交） |
| PR | <https://github.com/wante30/adas-edge/pull/1>（已合并） |
| PR 合并时间 | 2026-10-28T12:35:05Z |
| PR merge commit | `dc378d3a0d7d71e2ff1c42e4c42eb7d97e1d589d` |
| 默认分支 | `main` 已更新到 PR #1 merge commit（后续 docs-only 提交可能再前进） |
| 强制推送 | 无 |
| 硬重置/清理 | 无 |

### 1.1 已推送内容

- `live/stage2` 的 77 个历史提交；
- 2026-10-28 工作树未提交源码（Qt/AI/PCIe/HMI/测试/工具）；
- 本次审计新增文档：
  - `docs/audit/PROJECT_INVENTORY.md`
  - `docs/architecture/*`
  - `docs/models/*` + evidence
  - `docs/deployment/*` + evidence
  - `docs/hmi/*` + 实板截图
  - `docs/roadmap/ADAS_EVENT_ENGINE_PLAN.md`
  - `docs/competition/REQUIREMENT_MATRIX.md`
  - `artifacts/manifests/model_manifest.json`
- `archive/oneoff_202610/` 一次性 HMI 脚本归档；
- `rtl/avm_640/`、`matlab/avm_fpga_sim/` 源码与仿真说明（不含 build 产物）。

### 1.2 未推送内容

| 内容 | 原因 |
|---|---|
| BDD/B05N/B05L 权重 `.pt`、ONNX、RKNN 大文件 | 仓库策略/体积；只保留 manifest/SHA256 |
| BDD100K/Cityscapes/MOT17 数据集 | 体积/许可；不进入 Git |
| `outputs/` 11.6 GB、`.venv/` 5.9 GB、`tmp/`、`.deps/`、`data/`、`benchmarks/` | 开发缓存/实验结果 |
| `deployment/lubancat/qt_gpu_probe/results/` 482 MB | 截图/日志/测试产物；只挑关键证据归档 |
| 根目录调试图片 `new.png`/`ref.png`/`*.jpg` | 一次性调试素材；保留在 Windows |
| `rtl/avm_640/build/` | 生成物；已加入 `.gitignore` |
| FPGA 完整工程 | 本机不可访问；未编造 |

---

## 2. 分支/工作树

审计开始时：

```text
C:/Users/JingyuWang/adas-edge       19f050e [live/stage2]
C:/Users/JingyuWang/adas-edge-b01    6fd12b9 [research/rk3566-adas-models]
C:/Users/JingyuWang/adas-edge-b05n   6a2933b [research/rk3566-p2-multiclass]
C:/Users/JingyuWang/adas-edge-b05o   891978a [research/rk3566-b05o-video-scheduling]
```

`live/stage2` 是当前最新完整工作线；其他 worktree 是研究分支。  
`local-round2-full-history` 含 575 MB zip/322 MB npy 等大文件历史，**未推送**，仅本地/备份保留。

---

## 3. 验证

### 3.1 本地测试

Windows `adas-edge` `.venv`：

```text
559 passed, 9 warnings in 78.01s
```

### 3.2 远程验证

```text
git ls-remote origin live/stage2
659223faf7088f1f88a82f65fa29769f59b3b3d4	refs/heads/live/stage2

git ls-remote origin main
c321a89f26e5fa05a222ad29659e4148ec6c62d6	refs/heads/main   ← 合并前

# PR #1 合并后
dc378d3a0d7d71e2ff1c42e4c42eb7d97e1d589d	refs/heads/main
4fb4d0c6f8d5b2f7c51a2058b57618949b04ca92	refs/heads/live/stage2

# 研究分支（审计后）
b8f13f881d866107e7e6834b2fa5fd0ae2774fc6	refs/heads/research/rk3566-adas-models
82f2d57534628f7e3920ede4b2293408ddfd7c89	refs/heads/research/rk3566-p2-multiclass
857ece9d0ba4a749978fb12ac33b282028931e76	refs/heads/research/rk3566-b05o-video-scheduling
```

> `research/rk3566-p2-multiclass` 与 `research/rk3566-b05o-video-scheduling` 的远程分支保留原 tip；最新的 B05N/B05P 报告与证据已归档到 `main` 的 `docs/models/`，未重写这两个研究分支的历史。

### 3.3 安全检查

- 未发现私钥/Token/密码入库；
- 已把示例中的内网地址/调试 token 替换为 `<board-ip>` / `<redacted>`；
- 未新增 >2 MB 二进制（最大为 1.89 MB `car.obj`、1.06 MB 截图）；
- 未把 `*.pt`/`*.onnx`/`*.rknn` 大文件加入普通 Git 历史。

---

## 4. 备份

| 备份 | 大小 | 位置 |
|---|---:|---|
| 全部分支 bundle | 2,629,237,497 B | `D:\Backup\adas-edge-all-20261028.bundle` |
| `live/stage2` bundle | 53,240,114 B | `D:\Backup\live-stage2-20261028.bundle`；macOS `_backup/` |
| 工作树源码 tar | 153,357,121 B | `D:\Backup\adas-src-20261028.tgz`；macOS `_backup/` |
| 已跟踪修改 patch | 685,902 B | `D:\Backup\adas-edge-tracked-20261028.patch`；macOS `_backup/` |

---

## 5. 风险与未同步

1. **FPGA 工程**：`D:/QiansaiFPGA/source_code_3/PCIE_timig_fix` 在本机不存在，无法同步 RTL/时序/资源/bitstream。
2. **板端源码漂移**：板端 `main.qml`/`video_item.cpp` 与仓库不同；运行二进制 hash 已记录，但不能保证字节级可复现。
3. **模型权重**：未上传；需按 `artifacts/manifests/model_manifest.json` 从训练机/板端取回。
4. **研究分支**：B05N/B05O 的未提交报告/工具仍在 Windows worktree；本次已把核心内容归档到 `docs/models/`，但分支级历史未全部合并到 `main`。
5. **默认分支**：合并前 `main` 仍是旧基线；PR #1 合并后更新。

---

## 6. 最终状态

- PR #1 已合并：`2026-10-28T12:35:05Z`
- merge commit：`dc378d3a0d7d71e2ff1c42e4c42eb7d97e1d589d`
- `main` 已从旧基线 `c321a89` 前进 80+ 个提交（77 个历史 + 审计/文档提交 + merge commit；准确数量以 `git rev-list --count c321a89..origin/main` 为准）。
- 默认分支已更新；`live/stage2` 分支保留。
- 本报告在合并后做了一次 docs-only 更新；若 `main` 再前进一个提交，以 `git ls-remote origin main` 为准。
- 本次同步不删除任何旧分支/旧提交；`main` 只做非强制合并。

### 6.1 默认分支内容抽查

合并后 `origin/main` 应包含：

```text
README.md
docs/audit/PROJECT_INVENTORY.md
docs/audit/GIT_SYNC_REPORT.md
docs/architecture/SYSTEM_ARCHITECTURE.md
docs/architecture/PCIE_DATA_FLOW.md
docs/architecture/FPGA_RK3566_INTERFACE.md
docs/models/TRAINING_REPORT.md
docs/models/MODEL_COMPARISON.md
docs/models/DATASET_AND_EVALUATION.md
docs/models/MODEL_MANIFEST.md
docs/deployment/RKNN_QUANTIZATION_REPORT.md
docs/deployment/RK3566_BENCHMARK.md
docs/deployment/DEPLOYMENT_GUIDE.md
docs/hmi/TESLA_ADAS_HMI.md
docs/hmi/HMI_PERFORMANCE.md
docs/roadmap/ADAS_EVENT_ENGINE_PLAN.md
docs/competition/REQUIREMENT_MATRIX.md
artifacts/manifests/model_manifest.json
```
