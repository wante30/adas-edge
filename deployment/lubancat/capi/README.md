# P2 / B-2a RKNN C API 推理库

`libp2capi.so` 是 AI Bridge 的持久化 C API 后端。它把 RKNN 调用、Raw-head
解码和 NMS 放在共享库里，Python 只负责 letterbox、结果归一化和业务逻辑。
每帧不会启动新进程，也不会重复加载模型。

## 1. 文件

| 文件 | 说明 |
|---|---|
| `p2capi.c` | 共享库源码：P2 检测 + B-2a 多任务检测/分割 |
| `include/rknn_api.h` | RKNN Runtime 2.3.2 头文件（编译用） |
| `Makefile` | 板端编译脚本 |
| `../../qt_gpu_probe/capi_detector.py` | ctypes 调用方 |

编译产物 `libp2capi.so` 不进入普通 Git 历史；板端编译后复制到
`/home/cat/qt_gpu_probe_ai_20261019/libp2capi.so`。

## 2. 编译

板端：

```bash
cd deployment/lubancat/capi
make CC=gcc
# 等价命令：
# gcc -O3 -fPIC -Wall -Wextra -Iinclude -shared p2capi.c -lrknnrt -lm -o libp2capi.so
```

依赖：

- RK3566 板端 `librknnrt.so`（RKNN Runtime 2.3.2，`-lrknnrt`）
- `libm`
- 头文件 `include/rknn_api.h`

不要用其它版本的 RKNN Runtime 替换板端库。

## 3. 导出接口

```c
void *p2_create(const char *model_path);
int   p2_detect(void *handle, const uint8_t *rgb, P2Result *out);
int   p2_detect_mtl(void *handle, const uint8_t *rgb, P2Result *out,
                    uint8_t *masks);
void  p2_seg_size(void *handle, int *h, int *w);
void  p2_destroy(void *handle);
```

`P2Result`：

```c
typedef struct {
    int count;
    float rknn_ms;
    float decode_ms;
    float total_ms;
    float data[1000][6];   /* cls, score, x1, y1, x2, y2 */
} P2Result;
```

`masks` 由调用方分配，大小为 `3 * h * w`，依次为
`lane_cat`、`lane_style`、`drivable` 的 uint8 argmax 类别图。

### 3.1 输入

- `rgb`：`512x288x3` uint8，NHWC，RGB，已做 letterbox。
- 检测模型（12 输出）调用 `p2_detect`。
- B-2a MTL 模型（15 输出）调用 `p2_detect_mtl`；`p2_seg_size` 返回
  `288x512`（原版）或 `144x256`（Seg-256）。

### 3.2 输出

- 检测框坐标为 512×288 模型空间，Python 侧按 letterbox 的
  `scale/pad_x/pad_y` 映射回原图。
- `want_float=0`，直接读取 INT8 输出。
- 分割输出为 per-tensor affine 量化；同一像素的 9/3/3 个通道共享
  scale/zero-point，因此直接对 int8 取 argmax 与反量化后取 argmax 等价。
  60 帧、8,847,360 像素实测一致率 1.000000。

## 4. 类别顺序

检测（与 BDD10 / B05N P2 一致）：

```text
0 person, 1 rider, 2 car, 3 truck, 4 bus, 5 train,
6 motor, 7 bike, 8 traffic light, 9 traffic sign
```

分割（严格按 `stageB2a/output_contract.json`）：

```text
lane_cat:  0 crosswalk, 1 double other, 2 double white, 3 double yellow,
           4 road curb, 5 single other, 6 single white, 7 single yellow,
           8 background
lane_style: 0 solid, 1 dashed, 2 background
drivable:   0 direct, 1 alternative, 2 background
```

## 5. 后处理

- 4 个尺度：P2/P3/P4/P5；每尺度 3 路 Raw-head。
- DFL softmax/期望 → ltrb → xyxy。
- 类别 argmax，conf 0.25，NMS IoU 0.45。
- MTL 的 3 个分割头在同一 `rknn_run` 内取回，argmax 后写入 `masks`。

## 6. 实测

独立 C API 连续 60 帧（板端 Qt/HUD 同时运行）：

| 模型 | RKNN p50 | 单帧 p50 | FPS |
|---|---:|---:|---:|
| B05N P2 INT8 | 23.9 ms | 49.0 ms | 14.33 |
| B-2a 原版 INT8 | 87.6 ms | 161.2 ms | 4.95 |
| B-2a Seg-256 INT8 | 80.5 ms | 122.3 ms | 5.97 |

完整 AI Bridge 回放数据见 `docs/deployment/MTL_CAPI_DEPLOYMENT.md`。

## 7. 生命周期与线程

- `p2_create` 每个模型调用一次；`p2_destroy` 释放 RKNN context。
- 一个 handle 不应被多个线程并发调用。
- AI Bridge 在模型切换时先创建新 detector，成功后再关闭旧 detector；
  失败时保留旧 detector。
