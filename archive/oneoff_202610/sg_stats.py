from pathlib import Path
# add public queue + external record
h=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.h')
s=h.read_text(encoding='utf-8')
needle='''    qulonglong queueTaken() const { return display_queue_->taken(); }'''
rep='''    qulonglong queueTaken() const { return display_queue_->taken(); }
    std::shared_ptr<DisplayQueue> displayQueue() const { return display_queue_; }
    void recordExternalUpload(const Frame &frame, double uploadMs, double sourceAgeMs);'''
if needle not in s: raise SystemExit('header queue marker')
s=s.replace(needle,rep,1)
h.write_text(s,encoding='utf-8')
# implement record in cpp before uploadP95
c=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=c.read_text(encoding='utf-8')
needle='''double VideoItem::uploadP95Ms() const {'''
rep='''void VideoItem::recordExternalUpload(const Frame &frame, double uploadMs, double sourceAgeMs) {
    metrics_->upload_ms.store(uploadMs);
    metrics_->source_to_upload_ms.store(sourceAgeMs);
    metrics_->frame_id.store(frame.id);
    metrics_->uploaded_frames.fetch_add(1);
    {
        QMutexLocker contentLocker(&metrics_->replay_content_mutex);
        const quint64 sourceIndex = static_cast<quint64>(frame.sourceIndex);
        if (!metrics_->replay_content_indices.contains(sourceIndex)) {
            metrics_->replay_content_indices.insert(sourceIndex);
            metrics_->unique_replay_content_frames.fetch_add(1);
        }
    }
    {
        QMutexLocker locker(&metrics_->upload_mutex);
        metrics_->upload_samples.push_back(uploadMs);
        if (metrics_->upload_samples.size() > 3600)
            metrics_->upload_samples.erase(metrics_->upload_samples.begin(), metrics_->upload_samples.begin() + 600);
        metrics_->source_age_samples.push_back(sourceAgeMs);
        if (metrics_->source_age_samples.size() > 3600)
            metrics_->source_age_samples.erase(metrics_->source_age_samples.begin(), metrics_->source_age_samples.begin() + 600);
    }
    noteRenderedFrame(frame.id, frame.sourceTimestampNs > 0 ? frame.sourceTimestampNs : frame.producedNs);
}

double VideoItem::uploadP95Ms() const {'''
if needle not in s: raise SystemExit('record marker')
s=s.replace(needle,rep,1)
c.write_text(s,encoding='utf-8')
print('video stats API')
