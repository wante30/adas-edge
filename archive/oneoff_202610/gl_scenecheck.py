from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('''        frames_++;
        if (elapsed_ - last_report_ > 1.0) {''','''        {
            const GLenum sceneError = f->glGetError();
            if (sceneError != GL_NO_ERROR)
                qWarning() << "GL error at TeslaSceneItem::render end" << sceneError;
        }
        frames_++;
        if (elapsed_ - last_report_ > 1.0) {''',1)
p.write_text(s,encoding='utf-8')
print('scene check')
