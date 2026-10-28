from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('''        program_->disableAttributeArray(0);
        program_->disableAttributeArray(1);
        program_->release();
        update();''','''        program_->disableAttributeArray(0);
        program_->disableAttributeArray(1);
        program_->release();
        {
            const GLenum endError = f->glGetError();
            if (endError != GL_NO_ERROR)
                qWarning() << "GL error at VideoItem::render end" << endError;
        }
        update();''',1)
p.write_text(s,encoding='utf-8')
print('end check')
