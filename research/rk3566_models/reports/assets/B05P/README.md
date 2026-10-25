# B-05P 证据索引

评估集：`b05n_bdd_multiclass` test 第 200–999 张（800 张），校准与评估无重叠。

阈值：AP 候选 conf=0.001，NMS IoU=0.7，匹配 IoU=0.5；operating point conf=0.25。

| 文件 | 内容 |
|---|---|
| `p2_int8_split_map.json` | P2 INT8，logits split，test 200 校准 |
| `p2_fp16_split_map.json` | P2 FP16，logits split |
| `p2_clip_int8_map.json` | P2 INT8 + 输出裁剪，train 200 校准 |
| `p2_clip_int8_500_map.json` | P2 INT8 + 输出裁剪，train 500 校准 |
| `p2_hybrid_out_map.json` | 输出层 FP16 hybrid |
| `p2_hybrid_head_map.json` | 检测头 FP16 hybrid |
| `bdd_fp16_map.json` | BDD 主模型 FP16 |
| `bdd_int8_map.json` | BDD 主模型 INT8（裁剪，train 500） |
| `error_analysis_image{0,1}.txt` | `rknn.accuracy_analysis` 逐层 cosine / 欧氏距离 |
| `p2_quantization.cfg` | hybrid step1 生成的量化配置，含 boxes/scores 的 scale、zero-point |

未收录：

- `spec_fp16_tlmap.json` / `spec_int8_tlmap.json`：评估脚本预处理/坐标映射有 bug，ONNX 参考同样 AP 0，结果无效；
- 逐事件原始 JSON 和 MP4 体积过大，保留在 `D:` 盘；
- 转换日志含大量进度条，保留在 `deployment/rknn_work/*.log`。
