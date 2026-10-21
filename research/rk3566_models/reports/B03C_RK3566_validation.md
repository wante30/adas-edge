# B-03C：BDD YOLOv8n RKNN 转换与 RK3566 验证

**状态：PARTIAL**。静态 ONNX 与 RKNN INT8 离线构建成功；RK3566 实机验证因既有 SSH 连接在 SSH banner 阶段中断而未完成。

## 离线结果

| 检查项 | 结果 |
| --- | --- |
| ONNX 导出 | 成功。Ultralytics 8.4.68，静态输入 `[1,3,288,512]`，原始输出 `[1,14,3024]`；拆分后 `boxes [1,4,3024]`、`scores [1,10,3024]`。 |
| BDD 类别 | 按模型元数据读取：`0 person, 1 rider, 2 car, 3 truck, 4 bus, 5 train, 6 motor, 7 bike, 8 traffic light, 9 traffic sign`。没有映射到COCO class ID。 |
| PyTorch / ONNX 对照 | BDD100K validation目录按文件名排序的前10张，512×288、RGB、conf 0.25、NMS IoU 0.45：两端均113框，113/113一对一匹配，平均IoU 0.9988，平均置信度差0.0001。类别计数一致：person 5、car 81、bus 1、traffic light 10、traffic sign 16。 |
| 低阈值对照 | conf 0.05、同一NMS：PyTorch 215框、ONNX 216框，匹配215框、无漏配、ONNX多1框；平均IoU 0.9991，平均置信度差0.0001。 |
| RKNN INT8 构建 | 成功。`rknn-toolkit2 2.3.2`、target `rk3566`、`normal` + `channel`量化，转换配置为`mean=0`、`std=255`。校准使用既有76张非冻结道路视频样本，不含BDD评测图。 |

第一次对照出现的框数差异来自参考脚本遗漏`iou`参数：PyTorch默认NMS IoU为0.7，而ONNX端使用0.45。已修正[rect_source_dump.py](../../../tools/rect_source_dump.py)，并在[test_rect_backends_and_hetero.py](../../../tests/test_rect_backends_and_hetero.py)添加回归测试。该差异不是模型导出损坏。

## RK3566 实机

未能连接到板端。RK3566的既有SSH别名`lubancat`解析到`198.18.0.84`；TCP连接建立后返回`kex_exchange_identification: read: Unknown error`，认证尚未开始。因此没有登录、上传或运行任何板端命令，也没有读取比赛服务/NPU负载。未修改RK3566或比赛服务。

RKNN Toolkit在PC端可加载生成的`.rknn`文件，但`init_runtime()`明确拒绝对已构建RKNN使用PC simulator；这不等于RK3566加载成功。板端模型加载、RGB UINT8输入、输出张量形状/有限值、10类解码、INT8相对FP32退化均**未验证**。NPU-only和完整Pipeline FPS均未测量；没有用PC速度代替板端速度。

ONNX与split ONNX、RKNN文件、10张图的预测JSON及对照结果均留在本机独立目录`D:\Datasets\adas_edge_training\b03c_rknn_validation_20261021`，未提交模型、BDD图片或视频。RKNN SHA-256：`BDFAA64782EA1EBBDD8F23F18ABD91567C193A0C44C02D1B94D08661A227BD82`。

## 结论

这款BDD YOLOv8n目前是**值得继续实机验证的离线候选**，但还不能作为现有YOLO11n的替换候选：缺少RK3566加载、INT8检测对照和速度证据。建议继续保留Traffic-Light Specialist；10张图上的框数不能证明交通灯召回完整，且INT8板端退化未知。

对照图片来自此前已使用的BDD100K validation集合，仅验证导出一致性，不代表独立精度或实机效果。恢复既有SSH连接后，最小后续步骤是在独立临时目录加载该模型、用同批图片对照，并完成短时NPU与Pipeline计时。
