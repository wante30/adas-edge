from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
needle='''        auto *f = ctx->functions();
        if (!program_) initGL();
        if (!program_) return;

        f->glViewport(0, 0, static_cast<GLint>(viewport_.width()), static_cast<GLint>(viewport_.height()));'''
rep='''        auto *f = ctx->functions();
        auto checkGl = [f](const char *tag) {
            const GLenum e = f->glGetError();
            if (e != GL_NO_ERROR) qWarning() << "Tesla GL error" << tag << e;
        };
        checkGl("render_start");
        if (!program_) initGL();
        if (!program_) return;
        checkGl("initGL");

        f->glViewport(0, 0, static_cast<GLint>(viewport_.width()), static_cast<GLint>(viewport_.height()));
        checkGl("viewport");'''
if needle not in s: raise SystemExit('render start marker')
s=s.replace(needle,rep,1)
s=s.replace('''        f->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        std::vector<float> vertices;''','''        f->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        checkGl("clear");

        std::vector<float> vertices;''',1)
s=s.replace('''        buildScene(vertices);
        if (vertices.empty()) return;''','''        buildScene(vertices);
        checkGl("buildScene");
        if (vertices.empty()) return;''',1)
s=s.replace('''        vbo_.allocate(vertices.data(), static_cast<int>(vertices.size() * sizeof(float)));
        program_->bind();''','''        vbo_.allocate(vertices.data(), static_cast<int>(vertices.size() * sizeof(float)));
        checkGl("vbo_allocate");
        program_->bind();''',1)
s=s.replace('''        program_->setUniformValue("uMVP", mvp);
        f->glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size() / 6));''','''        program_->setUniformValue("uMVP", mvp);
        checkGl("uniforms");
        f->glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size() / 6));
        checkGl("draw");''',1)
s=s.replace('''        vbo_.release();
        vao_.release();
        drawBillboards(mvp);

        frames_++;''','''        vbo_.release();
        vao_.release();
        checkGl("release");
        drawBillboards(mvp);
        checkGl("billboards");

        frames_++;''',1)
# remove old end check to not double? leave but it catches; no
p.write_text(s,encoding='utf-8')
print('scene checkpoints')
