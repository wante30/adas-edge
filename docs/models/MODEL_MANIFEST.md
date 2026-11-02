# 模型清单与 SHA256

更新日期：2026-10-28  
机器可读版本：`artifacts/manifests/model_manifest.json`

> 权重、ONNX、RKNN 文件默认不进入普通 Git 历史（`.gitignore` 已排除 `*.pt`、`*.onnx`、`*.rknn`）。本清单用于把“文件名”与“真实文件”绑定，避免旧模型/新模型混淆。需要复现时，从训练机或板端按路径取回并校验 SHA256。

---

## 1. 比赛选型模型 B（生产 service 尚未切换）

| 项目 | 值 |
|---|---|
| 方案 | B · B05N P2 十类 Raw-head INT8 单模型 |
| 文件 | `b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn` |
| 板端路径 | `/home/cat/cityscapes_arch_compare_20261027/models/b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn` |
| 大小 | 4,393,465 B |
| SHA256 | `65089ef0697bd7378071290baa218905cdcaed768452678da440b8eb116b2a5f` |
| 输入 | `[1,288,512,3]` uint8 NHWC |
| 类别 | 10（person/rider/car/truck/bus/train/motor/bike/traffic light/traffic sign） |
| 状态 | 板端已有该正式 RKNN 文件；本分支正在做独立真实 PCIe 验证；生产 service 仍为 A |
| 来源 | B05N P2 checkpoint → raw-head ONNX → RKNN INT8 |

---

## 2. 全部模型资产

| ID | 角色 | 格式 | 大小 (B) | SHA256 | 状态 |
|---|---|---|---:|---|---|
| `bdd-main-fp32` | BDD 主模型 FP32 | .pt | 6,268,010 | `acd17780b331c3970e22faf3c6f123eae8b8c4abcc736ad5ef38ff28ecd72fb2` | PC 源权重 |
| `bdd-main-raw-head-onnx` | BDD Raw-head ONNX | .onnx | 12,081,311 | `d6ea1651a1bdcef1868bcb1651191e53d4568668df9e7f50cd5f2e9abe528adb` | 转换源 |
| `bdd-main-raw-head-int8` | BDD Raw-head INT8 | .rknn | 4,236,613 | `a86ab05338b49be54f95aa4d0cb937f1471924d917840d0f907480e29e041afb` | **当前运行 A** |
| `b05n-p2-fp32` | B05N P2 10 类 | .pt | 6,148,944 | `c249f7fffc828c892c8308c0f5a4768bb4c0a3c3ab28434d2e70e843b5e278ed` | PC 研究 |
| `b05n-p2-onnx` | B05N P2 标准 ONNX | .onnx | 12,024,733 | `c30ca4d99ccdd87a5c80f05dec1994fd256e38208955d191e8a55f9205f22e75` | PC 参考 |
| `b05n-p2-raw-head-onnx` | B05N P2 Raw-head ONNX | .onnx | 11,752,800 | `a1f3c20c3d4cd30e1690d75df4bf56be03059c874906c2536fecd1c6ad55cdd6` | 转换源 |
| `b05n-p2-raw-head-int8` | B05N P2 Raw-head INT8 | .rknn | 4,393,465 | `65089ef0697bd7378071290baa218905cdcaed768452678da440b8eb116b2a5f` | 板端 B 验证 |
| `b05l-specialist-fp32` | B05L Specialist | .pt | 6,120,464 | `c2d8dd9250393af9eae838f147e9fcaff8bae5c37b05252ea1c1286e9c8776a7` | PC 专模 |
| `b05l-specialist-raw-head-onnx` | B05L Raw-head ONNX | .onnx | 11,747,830 | `74b7776606c07195ae6d3cc86369dff08a13b46728089859f798c140bf20897c` | 转换源 |
| `b05l-specialist-raw-head-int8` | B05L Raw-head INT8 | .rknn | 4,717,877 | `c11b95b481b25ee34a093390001e821af8f2503f923a52159b8576c6a6f38b6f` | 板端 C/D 验证 |
| `yolo11n-split-320` | 历史 YOLO11n COCO 320 | .rknn | 3,893,761 | `60f17e9bfe20b27dc7d0be068bc9852024e4a941847f881e1890bf96b2c977da` | 历史 smoke |
| `yolo11n-split-416` | 历史 YOLO11n COCO 416 | .rknn | 3,954,625 | `87dfd6d480760d105e0abd7d04829d5059c8c5721bd7e3d6c53784de9793bb43` | 历史 smoke |
| `legacy-yolo11n-split-512x288` | 旧生产 YOLO11n 512×288 | .rknn | 3,927,489 | `7a697ccc476954f6f36f7441b842257a2d7a019653cc4bad2fbf8de327c50591` | 回滚候选 |
| `legacy-tl-specialist-hybrid2` | 旧交通灯专模 | .rknn | 4,246,080 | `becfaada5da49c4dd6c143e815f8f957fb7b7ed759040d4e18c685a9c434e6f8` | 回滚候选 |

---

## 3. 路径对照

### 3.1 Windows 训练机

```text
D:\Datasets\adas_edge_training\
  b03b_pretrained_cache_20261021\bdd100k-yolov8n-best.pt
  b05n_final\b05n_p2_multiclass.pt
  b05n_final\b05n_p2_multiclass.onnx
  b05n_runs\b05n_p2_bdd_full\weights\best.pt
  b05l_runs\v8n_p2\weights\best.pt
  rknn_b05o\bdd_yolov8n_512x288_raw_heads.onnx
  rknn_b05o\bdd_yolov8n_512x288_raw_heads_int8.rknn
  rknn_b05o\b05n_p2_multiclass_512x288_raw_heads_20261026.onnx
  rknn_b05o\b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn
  rknn_b05o\b05l_v8n_p2_512x192_raw_heads_20261026.onnx
  rknn_b05o\b05l_v8n_p2_raw_heads_int8_500_20261026.rknn
```

### 3.2 RK3566 板端

```text
/home/cat/cityscapes_arch_compare_20261027/models/
  bdd_yolov8n_512x288_raw_heads_int8.rknn          ← 当前 A
  b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn
  b05l_v8n_p2_raw_heads_int8_500_20261026.rknn

/home/cat/adas-edge-rect/rect_models/
  yolo11n_rk3566_int8_split_512x288.rknn            ← 旧生产回滚
  bdd_yolov8n_512x288_int8_kl.rknn
  ...

/home/cat/adas-edge-rect/tl_check/
  tl_specialist_512x192_hybrid2.rknn                ← 旧专模回滚
```

### 3.3 仓库内（小文件）

```text
deployment/rknn/yolo11n_rk3566_int8_split_320.rknn
deployment/rknn/yolo11n_rk3566_int8_split_416.rknn
deployment/rknn/yolo11n_rk3566_fp16.rknn
deployment/rknn/yolo11n_rk3566_int8.rknn
deployment/rknn/yolo11n_rk3566_int8_split.rknn
```

> 仓库内这些是历史 COCO YOLO11n 产物，**不是**当前 B05N/P2/BDD Raw-head 模型。

---

## 4. 复现/校验命令

Windows PowerShell：

```powershell
Get-FileHash 'D:\Datasets\adas_edge_training\rknn_b05o\bdd_yolov8n_512x288_raw_heads_int8.rknn' -Algorithm SHA256
```

Linux/RK3566：

```sh
sha256sum /home/cat/cityscapes_arch_compare_20261027/models/bdd_yolov8n_512x288_raw_heads_int8.rknn
```

---

## 5. 大文件策略

- 不把 `*.pt`、`*.onnx`、`*.rknn` 直接提交到普通 Git 历史。
- 不在本仓库上传 BDD100K、Cityscapes、MOT17 或训练缓存。
- 若后续需要公开权重，优先使用 GitHub Release，并附本 manifest 的 SHA256。
- 当前 GitHub 仓库只保留代码、报告、证据 JSON、截图和小型 RKNN 历史文件。
