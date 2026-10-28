from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('''    double x = 0, y = 0, w = 0, h = 0;
};''','''    double x = 0, y = 0, w = 0, h = 0;
    double confidence = 0;
    int missed = 0;
    bool predicted = false;
};''',1)
s=s.replace('''            b.h = map.value(QStringLiteral("height")).toDouble();
            if (b.id > 0 && b.w > 2 && b.h > 2) next.push_back(b);''','''            b.h = map.value(QStringLiteral("height")).toDouble();
            b.confidence = map.value(QStringLiteral("confidence")).toDouble();
            b.missed = map.value(QStringLiteral("missed")).toInt();
            b.predicted = map.value(QStringLiteral("predicted")).toBool();
            // Stale/predicted tracks are retained by the tracker for continuity,
            // but must not be drawn as if they were current detections.
            if (b.id > 0 && b.w > 2 && b.h > 2 && !b.predicted && b.missed == 0 &&
                b.confidence >= 0.20)
                next.push_back(b);''')
p.write_text(s,encoding='utf-8')
print('live stale filter')
