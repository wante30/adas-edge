from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=p.read_text(encoding='utf-8')
old='''        // Do not self-trigger another render here. The producer calls
        // requestFrameUpdate() for every accepted frame; a trailing update()
        // keeps redrawing the old texture forever and competes with producer/AI.
    }'''
new='''        update();
    }'''
if old not in s: raise SystemExit('restore marker')
s=s.replace(old,new,1)
p.write_text(s,encoding='utf-8')
m=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\main.cpp')
s=m.read_text(encoding='utf-8')
s=s.replace('''    report["qt_scene_graph_render_callback_fps_last_window"] = video->renderFps();''','''    report["qt_scene_graph_render_callback_fps_last_window"] = video->renderFps();
    report["video_item_render_fps_last_window"] = video->renderFps();''',1)
m.write_text(s,encoding='utf-8')
print('restored update + video render metric')
