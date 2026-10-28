from pathlib import Path
# publisher generation per content loop
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tools\pgv_local_publisher.py')
s=p.read_text(encoding='utf-8')
s=s.replace('''    transport_index = 0
    while True:''','''    transport_index = 0
    last_source_index = -1
    cycle = 0
    while True:''',1)
s=s.replace('''        source_index = int(transport_index * args.source_fps / args.fps) % frame_count
        offset = source_index * FRAME_BYTES''','''        source_index = int(transport_index * args.source_fps / args.fps) % frame_count
        if last_source_index >= 0 and source_index < last_source_index:
            cycle += 1
        last_source_index = source_index
        offset = source_index * FRAME_BYTES''',1)
s=s.replace('''                args.generation, crc[source_index], FRAME_BYTES, stamp))''','''                args.generation + cycle, crc[source_index], FRAME_BYTES, stamp))''',1)
p.write_text(s,encoding='utf-8')
# C++ header atomics
h=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.h')
s=h.read_text(encoding='utf-8')
s=s.replace('''    std::atomic<bool> ai_running_{false};''','''    std::atomic<bool> ai_running_{false};
    std::atomic<uint32_t> ai_epoch_{0};
    uint32_t last_ai_epoch_ = 0;''',1)
h.write_text(s,encoding='utf-8')
# C++ producer generation + AI result epoch
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('''        if (haveFrame && generation == lastGeneration) {
            const uint32_t delta = sourceFrame - static_cast<uint32_t>(lastSourceFrame);
            if (delta == 0 || delta >= 0x80000000U) continue;
            source_dropped_frames_.fetch_add(delta - 1);
        }''','''        if (haveFrame && generation != lastGeneration) {
            // Local PGVF cycles increment generation. Start a new tracker epoch
            // so the tail of the previous loop cannot be associated with the
            // first vehicles of the next loop.
            ai_epoch_.fetch_add(1);
        } else if (haveFrame && generation == lastGeneration) {
            const uint32_t delta = sourceFrame - static_cast<uint32_t>(lastSourceFrame);
            if (delta == 0 || delta >= 0x80000000U) continue;
            source_dropped_frames_.fetch_add(delta - 1);
        }''',1)
s=s.replace('''            {
                QMutexLocker locker(&ai_metrics_mutex_);
                if (frameId < latest_ai_frame_id_.load() || timestampNs <= 0) {
                    tracks_.clear();
                    next_track_id_ = 1;
                }''','''            {
                QMutexLocker locker(&ai_metrics_mutex_);
                const uint32_t resultEpoch = ai_epoch_.load();
                if (resultEpoch != last_ai_epoch_) {
                    tracks_.clear();
                    next_track_id_ = 1;
                    last_ai_epoch_ = resultEpoch;
                }
                if (frameId < latest_ai_frame_id_.load() || timestampNs <= 0) {
                    tracks_.clear();
                    next_track_id_ = 1;
                }''',1)
p.write_text(s,encoding='utf-8')
print('loop epoch + publisher generation')
