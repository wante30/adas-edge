from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.h')
s=p.read_text(encoding='utf-8')
old='''    Q_PROPERTY(int maxTrackPredictionMs READ maxTrackPredictionMs
               WRITE setMaxTrackPredictionMs NOTIFY metricsChanged)'''
new='''    Q_PROPERTY(int maxTrackPredictionMs READ maxTrackPredictionMs
               WRITE setMaxTrackPredictionMs NOTIFY metricsChanged)
    Q_PROPERTY(bool spatialViewActive READ spatialViewActive
               WRITE setSpatialViewActive NOTIFY spatialViewActiveChanged)'''
if old not in s: raise SystemExit('prop')
s=s.replace(old,new)
old='''    void setMaxTrackPredictionMs(int value) {
        max_track_prediction_ms_.store(std::clamp(value, 50, 1000));
    }'''
new='''    void setMaxTrackPredictionMs(int value) {
        max_track_prediction_ms_.store(std::clamp(value, 50, 1000));
    }
    bool spatialViewActive() const { return spatial_view_active_.load(); }
    void setSpatialViewActive(bool value);'''
if old not in s: raise SystemExit('setter')
s=s.replace(old,new)
old='''    void frameAdvanced();
    void sourceViewportChanged();'''
new='''    void frameAdvanced();
    void spatialViewActiveChanged();
    void sourceViewportChanged();'''
if old not in s: raise SystemExit('signal')
s=s.replace(old,new)
old='''    void runAiBridge(double fps, double aiHz, QString frameFile);'''
new='''    void runAiBridge(double fps, double aiHz, QString frameFile);
    void writeTeslaSceneTracks(const QVariantList &boxes, uint64_t frameId, qint64 timestampNs);'''
if old not in s: raise SystemExit('decl')
s=s.replace(old,new)
old='''    std::atomic<int> max_track_prediction_ms_{250};'''
new='''    std::atomic<int> max_track_prediction_ms_{250};
    std::atomic<bool> spatial_view_active_{false};'''
if old not in s: raise SystemExit('member')
s=s.replace(old,new)
p.write_text(s,encoding='utf-8')
print('header patched')
