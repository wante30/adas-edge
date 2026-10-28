from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('''    const float r = ego ? .20f : (truck ? .38f : (bus ? .30f : .56f));
    const float g = ego ? .28f : (truck ? .42f : (bus ? .42f : .60f));
    const float b = ego ? .38f : (truck ? .46f : (bus ? .50f : .64f));''','''    const float r = ego ? .20f : (truck ? .24f : (bus ? .20f : .36f));
    const float g = ego ? .28f : (truck ? .28f : (bus ? .30f : .41f));
    const float b = ego ? .38f : (truck ? .31f : (bus ? .35f : .46f));''')
s=s.replace('''        const char *classes[] = {"car", "car", "truck", "car", "bus"};
        const float lanes[] = {-2.2f, 2.2f, 0.0f, 3.0f, -3.0f};''','''        const char *classes[] = {"car", "car", "car", "car", "car"};
        const float lanes[] = {-2.2f, 2.2f, 0.0f, 3.0f, -3.0f};''')
p.write_text(s,encoding='utf-8')
# QML labels
q=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.qml')
s=q.read_text(encoding='utf-8')
s=s.replace('text: "LIVE TRACKS · 5 Hz · 非实时几何"','text: root.sceneDemo ? "DEMO · GPU 连续动画" : "LIVE TARGETS · GPU · 非实时几何"')
s=s.replace('text: "目标来自真实 Track ID · 坐标未标定"','text: root.sceneDemo ? "演示目标 · 非真实检测" : "真实 Track ID · 坐标未标定"')
q.write_text(s,encoding='utf-8')
print('demo colors/labels')
