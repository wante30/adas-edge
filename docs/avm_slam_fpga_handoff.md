# AVM-SLAM 四路全景融合与 FPGA LUT 交接

## 1. 交付范围

输入为本次提供的四段 AVM-SLAM 鱼眼 MP4、`stitch.json` 和 `param.yaml`。生成 640×640 地面鸟瞰融合、完整视频及兼容本项目参考 RTL 的 96 位映射表。完整机器校验结果见交付包的 `manifest.json`、`validation.json` 和 `logs/`。

本次是 PC 生成与 RTL 仿真交付。尚未进行 PG2L100H 布局布线、DDR 控制器集成、PCIe/HDMI 联调或真实相机标定，不代表板端已跑通。

## 2. 文件入口

| 文件 | 用途 |
| --- | --- |
| `lut/avm_slam_96bit_le.bin` | 409600 条、每条 12 字节，共 4915200 字节，无文件头；给外部 LUT 存储器加载 |
| `lut/avm_slam_96bit.hex` | 相同数据，每行 24 个十六进制字符，可供 Verilog `$readmemh` 使用 |
| `avm_slam_full_640_lut.mp4` | 整段真实输入经该 LUT 的整数回放结果；H.264 编码仅供观看 |
| `preview/frame_*_lut.png` | 无损 RGB565 展开预览 |
| `golden/frame_*/slot_*` | 四路工作分辨率 RGB565 输入，BIN/HEX 两种格式 |
| `golden/frame_*/expected_rgb565.*` | 同帧定点期望输出，硬件校验请用它而非压缩视频 |
| `source/stitch.json`、`param.yaml` | 本组原始参数副本 |
| `source/float_inverse_maps.npz` | 独立检查/重新量化用的浮点映射，不需要装入 FPGA |
| `tools/`、`tests/`、`rtl/` | 生成、回放、格式测试和现有参考 RTL |

## 3. 输入槽位必须对应

本包按 `stitch.json` 的 camId 定义槽位，**不可沿用其他数据集的视频排列**。

| FPGA 本地槽位 | 方向 | 本次视频文件 |
| --- | --- | --- |
| 0 | 左 | `47b43f60e2085fac67d6346fb54de39f_raw.mp4` |
| 1 | 前 | `b106ba02a43a7ce790f378bf10533d36_raw.mp4` |
| 2 | 右 | `863e9d946a1dc77f6dd1169da2ae28e1_raw.mp4` |
| 3 | 后 | `f42b4a943a47a4a6fb1d6b8a3914cc8d_raw.mp4` |

原始输入 1280×960。工作输入固定为 640×320 RGB565、小端、逐行，无行尾 padding。先执行 OpenCV INTER_LINEAR 像素中心缩放，再按 R5G6B5 截断打包。四路读同帧号，不得混用不同帧的摄像头缓存。MP4 的 2982 个逐帧时间戳一致，但本轮未独立验证传感器曝光同步。

缩放坐标为 `u_work=(u_native+0.5)/2-0.5`、`v_work=(v_native+0.5)/3-0.5`。输入虽在两轴使用不同缩放比例，投影已分别补偿；不是将方形地面坐标拉伸。若 FPGA 实际输入改变了分辨率、crop、旋转、镜像、行 stride 或缩放采样中心，必须重新导出或重新验证 LUT。

## 4. 几何和融合

车辆坐标 X 向右、Y 向前、Z 向上，地面 Z=0，单位米；输出上方为车前。640×640 对应 X、Y 各 [-7.13,7.13] 米，像素中心采样，每像素 0.02228125 米。相对 `param.yaml` 的 1354×1632 原始 BEV，这保留横向范围，裁去较远的前后区域，以保持输出平面等比例。

中心遮挡由 `VisualStart` 定义：半宽约 1.13 米、半长约 2.586 米，黑色表示车身遮挡，不填充不存在的地面。没有使用 YAML/JSON 中相互冲突的后悬值来调整标定。

外参按 row-major 3×4 的 world-to-camera `[R|t]` 解释；四个 R 均满足旋转矩阵约束，`C=-Rᵀt` 得到合理的左右/前后安装位置和正高度。原始像素用 `oriIntrinsics`；焦距为 100 的 `Intrinsics` 属于另一个成像平面，未混用。

畸变采用 KB4 鱼眼公式。配置没有明确写出模型名称，因此先比较 KB4/radtan4 两种解释及所有 24 种视频排列，并检查实际接缝、相机中心和独立 OpenCV 投影。选定排列的 KB4 在三组检查帧上重叠误差最低。这个结果支持该解释，但不能冒充原作者对配置字段的正式定义。`skewingH/W` 等未说明的应用层字段没有凭名称套用。

每个输出像素逆投影到四相机，剔除地面背向、越界及车身区域，按光轴夹角余弦平方和有效区边缘渐变打分，选前两路并归一化融合。属于静态几何 LUT；无模型推理、运行时 SLAM 或逐帧重新标定。

## 5. 96 位契约

| 位范围 | 字段 | 解释 |
| --- | --- | --- |
| 17:0 | addr_a | A 路左上角像素地址 `y*640+x`，单位是像素而不是字节 |
| 35:18 | addr_b | B 路左上角像素地址 |
| 43:36 / 51:44 | fx_a / fy_a | Q0.4 小数，有效值 0–15 |
| 59:52 / 67:60 | fx_b / fy_b | Q0.4 小数，有效值 0–15 |
| 78:68 | alpha_q | A 路权重 Q0.6，值 0–64，B 路为 `64-alpha_q` |
| 81:79 / 84:82 | cam_a / cam_b | 有效槽位 0–3 |
| 85 | valid | 0 输出黑色，不读取图像 |
| 86 | blend | 1 启用 B；0 时只读 A 且 alpha_q=64 |
| 95:87 | reserved | 0 |

BIN 为每条 96 位数的小端字节序；HEX 为数值高位在左的一行 24 位十六进制。不能将 BIN 的字节序直接照抄为 HEX 行。LUT 本身无版本头，必须连同 manifest 使用。

源图四邻域地址为 `addr`、`addr+1`、`addr+640`、`addr+641`；量化后也须保证 x≤638、y≤318。RGB565 先通过位复制展开到 RGB888，再以 Q0.4 双线性权重累加，加 128 后右移 8 位；A/B 融合加 32 后右移 6 位；最后截断打包 RGB565。参考实现始终使用整数，不以 OpenCV 插值的舍入代替 RTL 舍入。

字节地址可在外部适配器中计算为 `frame_base[cam]+2*pixel_address` 和 `lut_base+12*output_pixel_index`。本包不指定 BAR、DDR 基地址或 ioctl；这些取决于正式工程。输入和 LUT 放外部内存，不能把数 MB 全部展开成片上 ROM。

## 6. 复现与验证

安装 Python、NumPy、OpenCV、pytest 和 FFmpeg。交付包中的运行版本记录在 `validation.json`。导出脚本拒绝覆盖已有输出目录。审计目录应含 `input_audit.json` 和 `video_pts.npz`，可用本包 `source/` 的对应文件；其中旧审计状态文件注明是补标定之前的快照。

```powershell
python -m pytest tests/test_avm_slam_lut.py -q
python tools/export_avm_slam_fpga.py --calibration source/stitch.json --param source/param.yaml --videos '原始四路视频目录' --audit-root source --output '新的输出目录'
python tools/verify_avm_slam_package.py .
iverilog -g2012 -s tb_avm_4ch_640_stream -o avm_tb.vvp rtl/avm_4ch_640_stream.v rtl/tb_avm_4ch_640_stream.v
vvp avm_tb.vvp +LUT=lut/avm_slam_96bit.hex +SRC0=golden/frame_0000/slot_0_left.hex +SRC1=golden/frame_0000/slot_1_front.hex +SRC2=golden/frame_0000/slot_2_right.hex +SRC3=golden/frame_0000/slot_3_back.hex +EXPECTED=golden/frame_0000/expected_rgb565.hex
```

源 MP4 为可变帧间隔 0.08/0.12 秒，并有一次 0.56 秒间隔。预览视频在 25 fps 时间格上保持最近帧，保留原始 298.64 秒时长，**不会把 2982 帧当成 25 fps 加速播放**。25 fps 的编码格式不表示产生了 25 Hz 新相机数据。

本次第一帧参考 RTL 仿真为 409600 像素、0 mismatch、10407293 个周期。该 testbench 使用理想的一周期存储器响应；若按 100 MHz 推算，仅约 9.61 帧/秒，不能把它当作真实 DDR 下的性能承诺。实时实现需要结合正式工程核对时钟及 burst/cache 访存。其余帧检查结果以 `validation.json` 与逐帧仿真日志为准。

## 7. 使用边界

- 这是地面鸟瞰融合：离地的车辆、柱子、墙体会拉伸并可能出现重影；不是三维重建。
- 640×320 工作输入会损失原始鱼眼细节，远处映射被放大后会变软。
- 覆盖率是所选地面 ROI 的投影可采样比例，不是地面测量精度或真实无盲区认证。
- 本次提供的单张 BEV 无时间戳；对照图仅供视觉核对，未用它拟合变换，也不报告为同帧几何真值。
- 这份表绑定本数据集的相机安装和镜头。换成比赛实物相机后必须用实物标定重算。

## 8. 来源

- 用户提供的原始参数和视频，哈希见 manifest。
- 本项目 `rtl/avm_640/avm_4ch_640_stream.v` 与 `tools/avm_fpga_fixed_model.py`，打包时记录精确文件哈希。
- 数据集作者页面：https://github.com/yale-cv/avm-slam_dataset
- 鱼眼公式：https://docs.opencv.org/4.13.0/db/d58/group__calib3d__fisheye.html
