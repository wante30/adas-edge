from pathlib import Path
h=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.h')
s=h.read_text(encoding='utf-8')
s=s.replace('''        qulonglong lastTakeDepth() const { return last_take_depth_.load(); }''','''        qulonglong lastTakeDepth() const { return last_take_depth_.load(); }
        qulonglong taken() const { return taken_.load(); }''',1)
s=s.replace('''        std::atomic<qulonglong> last_take_depth_{0};''','''        std::atomic<qulonglong> last_take_depth_{0};
        std::atomic<qulonglong> taken_{0};''',1)
s=s.replace('''    qulonglong queueLastTakeDepth() const { return display_queue_->lastTakeDepth(); }''','''    qulonglong queueLastTakeDepth() const { return display_queue_->lastTakeDepth(); }
    qulonglong queueTaken() const { return display_queue_->taken(); }''',1)
h.write_text(s,encoding='utf-8')
c=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=c.read_text(encoding='utf-8')
s=s.replace('''    last_take_depth_.store(depth);
    // Latest-only display semantics''','''    last_take_depth_.store(depth);
    taken_.fetch_add(1);
    // Latest-only display semantics''',1)
s=s.replace('''void VideoItem::updateMetrics() {
    const double seconds = std::max(0.001, stats_clock_.elapsed() / 1000.0);
    const uint64_t produced = produced_frames_.load();
    const uint64_t consumed = render_metrics_->uploaded_frames.load();
    const uint64_t rendered = render_metrics_->rendered_frames.load();
    input_fps_.store((produced - last_produced_for_stats_) / seconds);
    consumed_fps_.store((consumed - last_consumed_for_stats_) / seconds);
    render_metrics_->render_fps.store((rendered - last_rendered_for_stats_) / seconds);''','''void VideoItem::updateMetrics() {
    // Windowed counters: restart() gives the actual interval since the last
    // sample. Do not divide cumulative counts by total process uptime.
    const double seconds = std::max(0.001, stats_clock_.restart() / 1000.0);
    const uint64_t produced = produced_frames_.load();
    const uint64_t consumed = render_metrics_->uploaded_frames.load();
    const uint64_t rendered = render_metrics_->rendered_frames.load();
    input_fps_.store((produced - last_produced_for_stats_) / seconds);
    consumed_fps_.store((consumed - last_consumed_for_stats_) / seconds);
    render_metrics_->render_fps.store((rendered - last_rendered_for_stats_) / seconds);
    last_produced_for_stats_ = produced;
    last_consumed_for_stats_ = consumed;
    last_rendered_for_stats_ = rendered;''',1)
# update cpu last after calculation
s=s.replace('''    cpu_percent_.store(last_cpu_ticks_ && ticksPerSecond > 0
                           ? 100.0 * cpuDelta / (ticksPerSecond * seconds * std::max(1, QThread::idealThreadCount()))
                           : 0.0);''','''    cpu_percent_.store(last_cpu_ticks_ && ticksPerSecond > 0
                           ? 100.0 * cpuDelta / (ticksPerSecond * seconds * std::max(1, QThread::idealThreadCount()))
                           : 0.0);
    last_cpu_ticks_ = cpuTicks;''',1)
c.write_text(s,encoding='utf-8')
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('''    report["display_queue_last_take_depth"] = static_cast<qint64>(video->queueLastTakeDepth());''','''    report["display_queue_last_take_depth"] = static_cast<qint64>(video->queueLastTakeDepth());
    report["display_queue_taken"] = static_cast<qint64>(video->queueTaken());''',1)
p.write_text(s,encoding='utf-8')
print('stats window fix')
