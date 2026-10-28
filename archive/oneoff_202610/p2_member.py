from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.h')
s=p.read_text(encoding='utf-8')
s=s.replace('''    uint64_t last_rendered_for_stats_ = 0;''','''    uint64_t last_rendered_for_stats_ = 0;
    uint64_t last_sync_for_stats_ = 0;''',1)
p.write_text(s,encoding='utf-8')
print('sync member')
