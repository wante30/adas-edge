from pathlib import Path
h=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.h')
s=h.read_text(encoding='utf-8')
s=s.replace('''        qint64 producedNs = 0;
        uint64_t sourceIndex = 0;''','''        qint64 producedNs = 0;
        qint64 sourceTimestampNs = 0;
        uint64_t sourceIndex = 0;''',1)
s=s.replace('''        void push(const std::shared_ptr<Frame> &frame);
        std::shared_ptr<Frame> take();
        void clear();

    private:
        mutable QMutex mutex_;
        std::deque<std::shared_ptr<Frame>> frames_;''','''        void push(const std::shared_ptr<Frame> &frame);
        std::shared_ptr<Frame> take();
        void clear();
        qulonglong droppedOnTake() const { return dropped_on_take_.load(); }
        qulonglong overwritten() const { return overwritten_.load(); }
        qulonglong maxDepth() const { return max_depth_.load(); }
        qulonglong lastTakeDepth() const { return last_take_depth_.load(); }

    private:
        mutable QMutex mutex_;
        std::deque<std::shared_ptr<Frame>> frames_;
        std::atomic<qulonglong> dropped_on_take_{0};
        std::atomic<qulonglong> overwritten_{0};
        std::atomic<qulonglong> max_depth_{0};
        std::atomic<qulonglong> last_take_depth_{0};''',1)
s=s.replace('''        mutable QMutex upload_mutex;
        std::vector<double> upload_samples;''','''        mutable QMutex upload_mutex;
        std::vector<double> upload_samples;
        mutable QMutex age_mutex;
        std::vector<double> source_age_samples;''',1)
s=s.replace('''    double uploadP95Ms() const;
    double sourceToUploadMs() const { return render_metrics_->source_to_upload_ms.load(); }''','''    double uploadP95Ms() const;
    double sourceToUploadMs() const { return render_metrics_->source_to_upload_ms.load(); }
    double sourceToUploadP50Ms() const;
    double sourceToUploadP95Ms() const;
    qulonglong queueDroppedOnTake() const { return display_queue_->droppedOnTake(); }
    qulonglong queueOverwritten() const { return display_queue_->overwritten(); }
    qulonglong queueMaxDepth() const { return display_queue_->maxDepth(); }
    qulonglong queueLastTakeDepth() const { return display_queue_->lastTakeDepth(); }''',1)
h.write_text(s,encoding='utf-8')

c=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=c.read_text(encoding='utf-8')
# queue stats, keep front for baseline
s=s.replace('''void VideoItem::DisplayQueue::push(const std::shared_ptr<Frame> &frame) {
    QMutexLocker locker(&mutex_);
    if (frames_.size() >= 4) frames_.pop_front();
    frames_.push_back(frame);
}

std::shared_ptr<VideoItem::Frame> VideoItem::DisplayQueue::take() {
    QMutexLocker locker(&mutex_);
    if (frames_.empty()) return {};
    auto frame = frames_.front();
    frames_.pop_front();
    return frame;
}''','''void VideoItem::DisplayQueue::push(const std::shared_ptr<Frame> &frame) {
    QMutexLocker locker(&mutex_);
    if (frames_.size() >= 4) {
        frames_.pop_front();
        overwritten_.fetch_add(1);
    }
    frames_.push_back(frame);
    max_depth_.store(std::max<qulonglong>(max_depth_.load(), frames_.size()));
}

std::shared_ptr<VideoItem::Frame> VideoItem::DisplayQueue::take() {
    QMutexLocker locker(&mutex_);
    if (frames_.empty()) return {};
    last_take_depth_.store(frames_.size());
    auto frame = frames_.front();
    frames_.pop_front();
    return frame;
}''',1)
# set source timestamp producer PGV
s=s.replace('''        slot->producedNs = steadyNs();
        source_frame_id_.store(sourceFrame);''','''        slot->producedNs = steadyNs();
        slot->sourceTimestampNs = static_cast<qint64>(timestampNs);
        source_frame_id_.store(sourceFrame);''',1)
# raw replay source timestamp
s=s.replace('''    frame->producedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    if (replay_frames_.isEmpty()) {''','''    frame->producedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    frame->sourceTimestampNs = frame->producedNs;
    if (replay_frames_.isEmpty()) {''',1)
# age use source timestamp and samples
s=s.replace('''                const double sourceAgeMs = std::max<qint64>(0, nowNs - frame->producedNs) / 1.0e6;
                metrics_->upload_ms.store(uploadMs);''','''                const qint64 sourceNs = frame->sourceTimestampNs > 0
                    ? frame->sourceTimestampNs : frame->producedNs;
                const double sourceAgeMs = std::max<qint64>(0, nowNs - sourceNs) / 1.0e6;
                metrics_->upload_ms.store(uploadMs);''',1)
s=s.replace('''                QMutexLocker locker(&metrics_->upload_mutex);
                metrics_->upload_samples.push_back(uploadMs);
                if (metrics_->upload_samples.size() > 3600) {
                    metrics_->upload_samples.erase(metrics_->upload_samples.begin(),
                                                   metrics_->upload_samples.begin() + 600);
                }''','''                QMutexLocker locker(&metrics_->upload_mutex);
                metrics_->upload_samples.push_back(uploadMs);
                if (metrics_->upload_samples.size() > 3600) {
                    metrics_->upload_samples.erase(metrics_->upload_samples.begin(),
                                                   metrics_->upload_samples.begin() + 600);
                }
                metrics_->source_age_samples.push_back(sourceAgeMs);
                if (metrics_->source_age_samples.size() > 3600) {
                    metrics_->source_age_samples.erase(metrics_->source_age_samples.begin(),
                                                       metrics_->source_age_samples.begin() + 600);
                }''',1)
# add methods near uploadP95
s=s.replace('''double VideoItem::uploadP95Ms() const {
    QMutexLocker locker(&render_metrics_->upload_mutex);
    if (render_metrics_->upload_samples.empty()) return 0.0;
    auto samples = render_metrics_->upload_samples;
    std::sort(samples.begin(), samples.end());
    return samples[static_cast<size_t>(0.95 * (samples.size() - 1))];
}''','''double VideoItem::uploadP95Ms() const {
    QMutexLocker locker(&render_metrics_->upload_mutex);
    if (render_metrics_->upload_samples.empty()) return 0.0;
    auto samples = render_metrics_->upload_samples;
    std::sort(samples.begin(), samples.end());
    return samples[static_cast<size_t>(0.95 * (samples.size() - 1))];
}

double VideoItem::sourceToUploadP50Ms() const {
    QMutexLocker locker(&render_metrics_->age_mutex);
    if (render_metrics_->source_age_samples.empty()) return 0.0;
    auto samples = render_metrics_->source_age_samples;
    std::sort(samples.begin(), samples.end());
    return samples[static_cast<size_t>(0.50 * (samples.size() - 1))];
}

double VideoItem::sourceToUploadP95Ms() const {
    QMutexLocker locker(&render_metrics_->age_mutex);
    if (render_metrics_->source_age_samples.empty()) return 0.0;
    auto samples = render_metrics_->source_age_samples;
    std::sort(samples.begin(), samples.end());
    return samples[static_cast<size_t>(0.95 * (samples.size() - 1))];
}''',1)
c.write_text(s,encoding='utf-8')
print('instrumented baseline queue/stats')
