from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=p.read_text(encoding='utf-8')
start=s.index('            if (!texture_) {', s.index('if (frame && frame->id != uploaded_id_)'))
end=s.index('            const GLenum uploadError = f->glGetError();', start)
new='''            f->glActiveTexture(GL_TEXTURE0);
            GLenum stale = f->glGetError();
            while (stale != GL_NO_ERROR) {
                qWarning() << "pre-existing GL error before RGB565 upload" << stale;
                stale = f->glGetError();
            }
            GLenum uploadError = GL_NO_ERROR;
            if (!texture_) {
                f->glGenTextures(1, &texture_);
                if (!texture_) {
                    pending_.reset();
                    update();
                    return;
                }
                f->glBindTexture(GL_TEXTURE_2D, texture_);
                f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                f->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB565, kSourceWidth, kSourceHeight, 0,
                                GL_RGB, GL_UNSIGNED_SHORT_5_6_5, frame->pixels.data());
                uploadError = f->glGetError();
                if (uploadError != GL_NO_ERROR)
                    qWarning() << "RGB565 glTexImage2D failed" << uploadError;
            } else {
                f->glBindTexture(GL_TEXTURE_2D, texture_);
                f->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, kSourceWidth, kSourceHeight,
                                   GL_RGB, GL_UNSIGNED_SHORT_5_6_5, frame->pixels.data());
                uploadError = f->glGetError();
                if (uploadError != GL_NO_ERROR)
                    qWarning() << "RGB565 glTexSubImage2D failed" << uploadError;
            }
'''
s=s[:start]+new+s[end:]
# replace following const uploadError line to avoid redefine
s=s.replace('            const GLenum uploadError = f->glGetError();\n            if (uploadError == GL_NO_ERROR) {','            if (uploadError == GL_NO_ERROR) {',1)
p.write_text(s,encoding='utf-8')
print('phase checks')
