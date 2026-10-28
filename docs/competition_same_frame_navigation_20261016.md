# 官方 112739 同帧导航与实景接入

本机原生车机使用 `video_20260929_112739.mp4` 的同一源帧生成道路画面和导航屏幕画面。道路送入现有 YOLO 主模型与红绿灯专项模型；导航顶部指令箭头采用图像形状分析，距离采用 RapidOCR 识别。方向、距离不来自播放时间表，也不借用另一段道路视频。

## 来源与实现范围

- 视频：`D:/BaiduNetdiskDownload/紫光同创杯赛题三参考素材/video_20260929_112739.mp4`。
- 视频 SHA-256：`1644f0230bcd72f77a31a846fed16e9965d38fa6d644659dc39b1f044992e587`。
- 视频为 1920×1080、30 FPS，共 2776 帧。
- 几何配置：`configs/competition_native_navigation_112739.json`。使用该视频的几何关键帧标定和插值，导航透视矫正到 960×540。配置不包含识别答案。
- 此素材是拍摄挡风玻璃与中控屏的视频，道路覆盖范围比独立前视相机有限。道路使用中央挡风玻璃裁剪，并遮黑中控屏及其边框，保留其他仍可见的近车和道路；黑区不是路面识别结果。出画的屏幕区域保留黑色，不生成缺失画面。
- 只增加原生车机适配，没有改变冻结的主模型/专项模型权重，也没有改旧网页。

## 导航识别及决策边界

识别区域限定为导航屏左上角主指令，不使用地图路线或下方车道箭头推断主指令。箭头与距离必须同时有效；距离采用 [RapidOCR 官方调用接口](https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/usage/) 的行识别模式，不运行全屏文字检测。依赖见 `requirements-navigation.txt`。

导航按源视频时钟 3 Hz 识别，750 ms 失效；暂停不重复推理、不刷新时间戳。轨迹中记录源帧、结果龄期、OCR 原文和裁剪矩阵。侧向影像、模糊或无法确认的截图输出未知；未识别道路名称保持空白。直行图标有合成测试，但该素材的真实样本主要为右转、左转、掉头。

导航是路线提示，不是立即转向命令。既有灯控优先级保持；未关联控制灯时仍不能据导航给出通行许可。掉头可以显示，但现有控制协议没有掉头动作，因此不能冒充左转发送。车机中的真实车辆速度、挡位、姿态均未测量。

## 运行与录制

默认双击 `run_competition_console.cmd`，打开导航模式原生窗口，初始暂停。空格播放，R 重播，F 查看诊断，V 切换检测标记。切换另一段视频会清除当前视频专用标定与导航解析。

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-navigation.txt
.venv\Scripts\python.exe tools/run_competition_console.py --video 'D:\BaiduNetdiskDownload\紫光同创杯赛题三参考素材\video_20260929_112739.mp4' --profile configs/competition_native_navigation_112739.json --split-navigation --mode navigation --mock-hud --record --headless --output outputs/navigation_112739_20261016/recording
```

输出目录必须是新的目录。`native_mp4v.mp4` 是实际原生车机渲染结果，`trace.jsonl` 保存每帧识别和软件 HUD 回读，`summary.json` 保存推理调用数、耗时和证据边界。导出 MP4 的帧率遵循源视频，不代表处理硬件实时达到同样帧率。

## 验证与限制

本轮证据位于 `outputs/navigation_112739_20261016/`。几何、导航识别、UI、同帧时间约束分别有独立测试；录制前后需检查真实输出帧，不能仅凭测试通过认定效果。

这次是 PC ONNX/CPU 实测和软件 HUD 寄存器验证，没有向真实 FPGA 写数据。RK3566 上的 OCR 部署、联合吞吐和 PCIe 联调尚未验证。道路名称、自动灯组关联、车道模型、自车姿态均未由本次接入补齐。
