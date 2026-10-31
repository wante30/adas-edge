# 模型清单与 SHA256

更新日期：2026-10-31
机器可读版本：`artifacts/manifests/model_manifest.json`

> 权重、ONNX、RKNN 文件默认不进入普通 Git 历史（`.gitignore` 已排除 `*.pt`、`*.onnx`、`*.rknn`）。本清单用于把“文件名”与“真实文件”绑定，避免旧模型/新模型混淆。需要复现时，从训练机或板端按路径取回并校验 SHA256。

---

## 1. 正式默认模型 B（2026-10-31）

| 项目 | 值 |
|---|---|
| 方案 | B · B05N P2 十类 Raw-head INT8 单模型 |
| 文件 | `b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn` |
| 板端路径 | `/home/cat/cityscapes_arch_compare_20261027/models/b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn` |
| 大小 | 4,393,465 B |
| SHA256 | `65089ef0697bd7378071290baa218905cdcaed768452678da440b8eb116b2a5f` |
| 输入 | `[1,288,512,3]` uint8 NHWC |
| 类别 | 10（person/rider/car/truck/bus/train/motor/bike/traffic light/traffic sign） |
| 状态 | 板端正式 RKNN；`adas-ai-bridge-live` 默认 B；B-CAPI/MTL 为可切换扩展 |
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
| `b05n-p2-mtl-b1-ckpt` | B-1 MTL checkpoint | .pt | 12,809,989 | `6c22062a90e810452f98420fae235be8cfc7159b503a15fa3e2e72dfef96b4c6` | 研究阶段 |
| `b05n-p2-mtl-b2a-ckpt` | B-2a MTL checkpoint | .pt | 12,810,053 | `34fd31c3470b7435a764c3ef3623685a3c8f60c38537fa6c38b7866d4c3df8fb` | MTL 源权重 |
| `b05n-p2-mtl-b2b-ckpt` | B-2b 对照 checkpoint | .pt | 12,810,053 | `3f087fa7dc02c6b485a941f5cd96bfae0b525b59862dec11aff9ade026bc2c75` | 对照，不部署 |
| `b05n-p2-mtl-b2a-onnx` | B-2a ONNX 转换源 | .onnx | 12,630,960 | `303e67511383c95638f51c4e6a0b53e5e3d904df141f548ee22561dff84de6d7` | 转换源 |
| `b05n-p2-mtl-b2a-fp16` | B-2a FP16 | .rknn | 8,360,418 | `e92b71e9c9dfc52befa5738741c12340517b2993fa7ea11c28599b0bbb8b85d6` | 精度参考 |
| `b05n-p2-mtl-b2a-int8` | B-2a INT8（MTL-FULL） | .rknn | 4,954,226 | `16b267ab5c2823a08e4dd86b77544dcdd09038c9e37c04a65140fd00dc44a68b` | **已部署/实测** |
| `b05n-p2-mtl-b2a-seg256-onnx` | Seg-256 ONNX | .onnx | 12,631,431 | `9d1e95ffb691c3a17f8ec3b93dfac30fa7005e378da9f6a06b19eea765a7a2cd` | 转换源 |
| `b05n-p2-mtl-b2a-seg256-int8` | Seg-256 INT8（MTL-FAST） | .rknn | 4,933,694 | `22f910ed5a40b0db32158c1175fb96b7ae1495565bb1e5d6d5d3d834b9e14732` | **已部署/实测** |
| `b05n-p2-mtl-b1-int8` | B-1 INT8 | .rknn | 4,954,290 | `29e232a3f1d3439dadde82f02613c6c929b35cd5f7e9a55fd48fecccfae9b413` | B-1 证据 |

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
  bdd_yolov8n_512x288_raw_heads_int8.rknn          ← A
  b05n_p2_multiclass_raw_heads_int8_500_20261026.rknn  ← 默认 B
  b05n_p2_multiclass_capi.rknn                     ← B-CAPI 符号链接
  b05l_v8n_p2_raw_heads_int8_500_20261026.rknn

/home/cat/models/b05n_p2_mtl_b2a/
  b05n_p2_mtl_b2a_int8.rknn                        ← MTL-FULL
  b05n_p2_mtl_b2a_fp16.rknn
/home/cat/models/b05n_p2_mtl_b2a_seg256/
  b05n_p2_mtl_b2a_seg256_int8.rknn                 ← MTL-FAST

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
