from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=p.read_text(encoding='utf-8')
start=s.index('void VideoItem::recordExternalUpload')
end=s.index('double VideoItem::uploadP95Ms', start)
block=s[start:end].replace('metrics_->','render_metrics_->')
s=s[:start]+block+s[end:]
p.write_text(s,encoding='utf-8')
print('metrics fix')
