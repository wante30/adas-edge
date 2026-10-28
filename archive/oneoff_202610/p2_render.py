from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=p.read_text(encoding='utf-8')
old='''        {
            const GLenum endError = f->glGetError();
            if (endError != GL_NO_ERROR) {
                qWarning() << "GL error at VideoItem::render end" << endError;
            }
        }
        update();
    }'''
new='''        {
            const GLenum endError = f->glGetError();
            if (endError != GL_NO_ERROR) {
                qWarning() << "GL error at VideoItem::render end" << endError;
            }
        }
        // Do not self-trigger another render here. The producer calls
        // requestFrameUpdate() for every accepted frame; a trailing update()
        // keeps redrawing the old texture forever and competes with producer/AI.
    }'''
if old not in s: raise SystemExit('render update marker')
s=s.replace(old,new,1)
p.write_text(s,encoding='utf-8')
print('removed trailing render update')
