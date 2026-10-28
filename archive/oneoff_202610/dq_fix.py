from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=p.read_text(encoding='utf-8')
old='''std::shared_ptr<VideoItem::Frame> VideoItem::DisplayQueue::take() {
    QMutexLocker locker(&mutex_);
    if (frames_.empty()) return {};
    last_take_depth_.store(frames_.size());
    auto frame = frames_.front();
    frames_.pop_front();
    return frame;
}'''
new='''std::shared_ptr<VideoItem::Frame> VideoItem::DisplayQueue::take() {
    QMutexLocker locker(&mutex_);
    if (frames_.empty()) return {};
    const qulonglong depth = frames_.size();
    last_take_depth_.store(depth);
    // Latest-only display semantics: keep the newest complete frame and
    // discard older queued frames in the same critical section. AI has its
    // own latest_ai_frame_ slot and is not affected by this consumer.
    auto frame = std::move(frames_.back());
    dropped_on_take_.fetch_add(depth > 0 ? depth - 1 : 0);
    frames_.clear();
    return frame;
}'''
if old not in s: raise SystemExit('take marker')
s=s.replace(old,new,1)
p.write_text(s,encoding='utf-8')
print('latest-only take installed')
