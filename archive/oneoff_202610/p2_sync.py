from pathlib import Path
h=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.h')
s=h.read_text(encoding='utf-8')
s=s.replace('''        std::atomic<uint64_t> rendered_frames{0};''','''        std::atomic<uint64_t> rendered_frames{0};
        std::atomic<uint64_t> synchronize_calls{0};''',1)
h.write_text(s,encoding='utf-8')
c=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=c.read_text(encoding='utf-8')
s=s.replace('''    void synchronize(QQuickFramebufferObject *item) override {
        auto *video = static_cast<VideoItem *>(item);
        viewport_ = video->sourceViewport();
        pgv_flip_ = video->pgvSource();
        // The producer marks the item dirty for every accepted frame. Taking
        // the newest frame during synchronize() is what makes Qt re-upload and
        // re-composite the FBO; pulling only in render() leaves a stale frame
        // on this xcb/EGL path even though QML animations continue.
        if (!pending_) pending_ = display_->take();
    }''','''    void synchronize(QQuickFramebufferObject *item) override {
        auto *video = static_cast<VideoItem *>(item);
        viewport_ = video->sourceViewport();
        pgv_flip_ = video->pgvSource();
        metrics_->synchronize_calls.fetch_add(1);
        // Always prefer the newest complete frame. If a render was delayed,
        // replace the pending frame instead of waiting for it to be uploaded.
        if (auto newer = display_->take()) pending_ = std::move(newer);
    }''',1)
c.write_text(s,encoding='utf-8')
m=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.cpp')
s=m.read_text(encoding='utf-8')
s=s.replace('''    report["video_item_render_fps_last_window"] = video->renderFps();''','''    report["video_item_render_fps_last_window"] = video->renderFps();
    report["video_synchronize_fps_last_window"] = video->synchronizeFps();''',1)
m.write_text(s,encoding='utf-8')
# add method
h=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.h')
s=h.read_text(encoding='utf-8')
s=s.replace('''    double renderFps() const { return render_metrics_->render_fps.load(); }''','''    double renderFps() const { return render_metrics_->render_fps.load(); }
    double synchronizeFps() const { return render_metrics_->synchronize_fps.load(); }''',1)
s=s.replace('''        std::atomic<double> render_fps{0.0};''','''        std::atomic<double> render_fps{0.0};
        std::atomic<double> synchronize_fps{0.0};''',1)
h.write_text(s,encoding='utf-8')
# update metrics compute
c=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=c.read_text(encoding='utf-8')
s=s.replace('''    render_metrics_->render_fps.store((rendered - last_rendered_for_stats_) / seconds);
    last_produced_for_stats_ = produced;''','''    render_metrics_->render_fps.store((rendered - last_rendered_for_stats_) / seconds);
    render_metrics_->synchronize_fps.store((render_metrics_->synchronize_calls.load() - last_sync_for_stats_) / seconds);
    last_sync_for_stats_ = render_metrics_->synchronize_calls.load();
    last_produced_for_stats_ = produced;''',1)
c.write_text(s,encoding='utf-8')
print('sync replace + metrics')
