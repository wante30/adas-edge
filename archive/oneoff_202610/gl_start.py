from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=p.read_text(encoding='utf-8')
needle='''        metrics_->rendered_frames.fetch_add(1);
        if (!program_) initGL();'''
rep='''        metrics_->rendered_frames.fetch_add(1);
        auto *startContext = QOpenGLContext::currentContext();
        if (startContext) {
            GLenum startError = startContext->functions()->glGetError();
            if (startError != GL_NO_ERROR)
                qWarning() << "GL error present at VideoItem::render start" << startError;
        }
        if (!program_) initGL();'''
if needle not in s: raise SystemExit('render start marker')
s=s.replace(needle,rep,1)
p.write_text(s,encoding='utf-8')
print('start check')
