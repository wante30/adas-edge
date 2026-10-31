# B-2a C API 实验程序

本目录保存 2026-10-31 板端 C API 实验的源码和原始日志。生产共享库源码在
`deployment/lubancat/capi/`；这里是测速、输出 dump 和一致性比较程序。

| 文件 | 用途 |
|---|---|
| `rknn_bench.c` | P2 / B-2a / Seg-256 连续测速；`want_float=0/1`；检测 NMS + 分割 argmax |
| `capi_seg256_dump.c` | 导出 Seg-256 int8/float 分割结果做一致性对比 |
| `run_with_sampler.py` | 运行 benchmark 并采样 CPU/NPU/温度 |
| `seg256_c_vs_b.json` | C API vs Python Seg-256 一致性 |
| `seg256_ref_meta.json` | 参考输出元数据 |
| `b2a_int8_resource.json` / `seg256_int8_resource.json` | NPU/CPU 采样 |
| `*.log` | 原始运行日志 |

## 编译

```bash
RKNN_INC=<repo>/deployment/lubancat/capi/include
gcc -O3 -I"$RKNN_INC" rknn_bench.c -lrknnrt -lm -o rknn_bench
gcc -O3 -I"$RKNN_INC" capi_seg256_dump.c -lrknnrt -lm -o capi_seg256_dump
```

依赖板端 `librknnrt.so` 2.3.2；不要用其它版本替换。

## 运行

```bash
./rknn_bench <model.rknn> <result.json> 0   # 0 = want_float=0
python3 run_with_sampler.py -- ./rknn_bench ...
```

完整结果与结论见 `capi_report.md`、`capi_bc_report.md` 和
`docs/deployment/MTL_CAPI_DEPLOYMENT.md`。
