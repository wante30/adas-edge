from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('qInfo("TESLA_SCENE_FPS %.1f", frames_ / (elapsed_ - last_report_));','qWarning("TESLA_SCENE_FPS %.1f", frames_ / (elapsed_ - last_report_));')
s=s.replace('''        const char *classes[] = {"car", "car", "car", "car", "car"};
        const float lanes[] = {-2.2f, 2.2f, 0.0f, 3.0f, -3.0f};''','''        const char *classes[] = {"car", "car", "car", "car", "car", "car"};
        const float lanes[] = {-2.4f, 2.4f, 0.0f, 3.1f, -3.1f, 1.2f};''')
s=s.replace('''        for (int i = 0; i < 5; ++i) {
            const float z = 5.0f + std::fmod(4.0f + i * 7.0f + phase * 34.0f, 34.0f);
            addBillboard(lanes[i], z, QString::fromLatin1(classes[i]));
        }''','''        for (int i = 0; i < 6; ++i) {
            const float z = 4.0f + std::fmod(2.0f + i * 5.0f + elapsed_ * 6.0f, 30.0f);
            addBillboard(lanes[i], z, QString::fromLatin1(classes[i]));
        }''')
p.write_text(s,encoding='utf-8')
print('demo motion tuned')
