from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=p.read_text(encoding='utf-8')
old='                for (const QVariantMap &item : ordered) boxes.append(item);\n            }\n'
new='                for (const QVariantMap &item : ordered) boxes.append(item);\n            }\n            writeTeslaSceneTracks(boxes, frameId, static_cast<qint64>(timestampNs));\n'
if old not in s:
    # show candidate
    i=s.index('boxes.append(item)')
    print(repr(s[i:i+80]))
    raise SystemExit('marker')
p.write_text(s.replace(old,new,1),encoding='utf-8')
print('inserted')
