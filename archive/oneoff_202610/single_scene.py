from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tools\tesla_scene_renderer.py')
s=p.read_text(encoding='utf-8')
s=s.replace('''            target = OUT_A if tick % 2 == 0 else OUT_B''','''            target = OUT_A''')
p.write_text(s,encoding='utf-8')
q=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.qml')
s=q.read_text(encoding='utf-8')
s=s.replace('''                root.sceneTick = (root.sceneTick + 1) % 2
                teslaScene.source = root.sceneTick
                        ? "file:///dev/shm/tesla_scene_b.png"
                        : "file:///dev/shm/tesla_scene_a.png"''','''                root.sceneTick = (root.sceneTick + 1) % 100000
                teslaScene.source = "file:///dev/shm/tesla_scene_a.png?t=" + root.sceneTick''')
q.write_text(s,encoding='utf-8')
print('single/query patched')
