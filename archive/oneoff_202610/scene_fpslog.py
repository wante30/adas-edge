from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('#include <QDateTime>\n','#include <QDateTime>\n#include <cstdio>\n')
s=s.replace('''            qWarning("TESLA_SCENE_FPS %.1f", frames_ / (elapsed_ - last_report_));''','''            std::fprintf(stderr, "TESLA_SCENE_FPS %.1f\\n", frames_ / (elapsed_ - last_report_));''')
p.write_text(s,encoding='utf-8')
print('scene fps log')
