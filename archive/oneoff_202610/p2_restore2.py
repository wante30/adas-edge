from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('''        // Always prefer the newest complete frame. If a render was delayed,
        // replace the pending frame instead of waiting for it to be uploaded.
        if (auto newer = display_->take()) pending_ = std::move(newer);''','''        // Producer marks the item dirty for every accepted frame. Keep the
        // existing pending frame until render() uploads it; replacing it in
        // synchronize() was measured to add work without improving throughput.
        if (!pending_) pending_ = display_->take();''',1)
p.write_text(s,encoding='utf-8')
print('restored sync')
