# 验证证据索引

- `real_photo_results.jpg`：RK3566 真实原生 ROI 输出，包含正确箭头与 UNKNOWN。裁剪来源、作者和许可见 `../../../../tests/fixtures/real_signals/ATTRIBUTION.md` 与 `../photo_sources.json`；该拼图仅组合/缩放展示，不改变分类输入。各照片区域保留源图的 CC BY-SA 许可。
- `arm_photo_regressions.json`：Python 3.8.10、OpenCV 4.10.0 的 9 个固定真实 ROI 回归；不是独立准确率。
- `photo_report.json`：15 张照片、33 个手工 ROI 的分组诊断统计。照片精确标签和下载字节哈希在上级目录。
- `photo_v2_frozen_report.json`：修复圆灯误判掉头之前的冻结留出结果。修复后该组成为回归集，不再是独立留出集。
- `red_video_report.json`、`green_video_report.json`：真实比赛视频的稀疏标注统计。PC 缓存检测器不是实机 B05N。
- `board_ab_summary.json`：早期算法同视频 30 秒 A/B，非新版形态识别性能。
- `qt_signal_debug.png`、`qt_video_debug.png`：新版 Qt 实机 Tesla/视频页调试截图；日志未发现 QML/属性模块异常。开发二进制 SHA-256 为 `1d3c3bba894eada87c9d563979a7a6338553f193e02c7eb808f64e2e3dcc4c97`。
- `qt_smoke_restored.json`、`qt_video_restored.json`：旧版隔离回放恢复收据，仅证明发送 SIGCONT，不能证明恢复单实例；后续补充核验见 lifecycle_recovery.json。正式启动脚本没有修改。实机墙钟不准确，验证时间使用主机记录和板端 monotonic 时间。

完整每帧 JSONL、对照视频、错误裁剪和原始 A/B 采样作为本次本地产物交付，未把视频大文件加入 Git。截图是实际运行结果，UNKNOWN 如实保留，不把调试界面包装成验收成功。

- `lifecycle_recovery.json`：清理 shell wait/SIGSTOP 造成的重复 Qt，原启动器重启后仅有一个正式 HMI。
- `final_board_state.json`：最终采集故障和正式程序状态；不能声称实时硬件链路验收完成。
