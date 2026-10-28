from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('#include <QOpenGLShaderProgram>\n','#include <QOpenGLShaderProgram>\n#include <QOpenGLTexture>\n#include <QImage>\n')
# Billboard struct after LiveCar
s=s.replace('''struct DemoCar {
    float x = 0, z = 0;
    QString cls = QStringLiteral("car");
    float phase = 0;
};''','''struct DemoCar {
    float x = 0, z = 0;
    QString cls = QStringLiteral("car");
    float phase = 0;
};

struct Billboard {
    float x = 0, z = 0, w = 1.9f, h = 2.0f;
    QVector3D tint{0.55f, 0.60f, 0.65f};
};''')
# add billboard helper before addDemo inside SceneRenderer? Insert member function before addDemo
s=s.replace('''    void addDemo(std::vector<float> &v) {''','''    void addBillboard(float x, float z, const QString &cls) {
        Billboard b;
        b.x = x; b.z = z;
        if (cls == "truck") { b.w = 2.5f; b.tint = QVector3D(.42f, .46f, .50f); }
        else if (cls == "bus") { b.w = 2.7f; b.tint = QVector3D(.34f, .44f, .52f); }
        else if (cls == "person") { b.w = .75f; b.tint = QVector3D(.65f, .66f, .58f); }
        else { b.w = 1.95f; b.tint = QVector3D(.58f, .62f, .67f); }
        b.h = b.w * 740.0f / 680.0f;
        billboards_.push_back(b);
    }

    void addDemo(std::vector<float> &v) {''')
# demo add billboard instead of carMesh
s=s.replace('''        for (int i = 0; i < 5; ++i) {
            const float z = 5.0f + std::fmod(4.0f + i * 7.0f + phase * 34.0f, 34.0f);
            carMesh(v, lanes[i], z, QString::fromLatin1(classes[i]), false);
        }''','''        for (int i = 0; i < 5; ++i) {
            const float z = 5.0f + std::fmod(4.0f + i * 7.0f + phase * 34.0f, 34.0f);
            addBillboard(lanes[i], z, QString::fromLatin1(classes[i]));
        }''')
# live add billboard instead carMesh
s=s.replace('''        for (const LiveCar &car : live_) carMesh(v, car.x, car.z, car.cls, false);''','''        for (const LiveCar &car : live_) addBillboard(car.x, car.z, car.cls);''')
# init billboards clear at buildScene start
s=s.replace('''    void buildScene(std::vector<float> &v) {
        // Ground, grass shoulders''','''    void buildScene(std::vector<float> &v) {
        billboards_.clear();
        // Ground, grass shoulders''')
# Need add billboards draw after opaque in render
s=s.replace('''        f->glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size() / 6));
        program_->disableAttributeArray(pos);
        program_->disableAttributeArray(col);
        vbo_.release();
        vao_.release();''','''        f->glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size() / 6));
        program_->disableAttributeArray(pos);
        program_->disableAttributeArray(col);
        vbo_.release();
        vao_.release();
        drawBillboards(mvp);''')
# Insert drawBillboards before buildScene
s=s.replace('''    void buildScene(std::vector<float> &v) {''','''    void drawBillboards(const QMatrix4x4 &mvp) {
        if (!texture_ || billboards_.empty()) return;
        auto *ctx = QOpenGLContext::currentContext();
        if (!ctx) return;
        auto *f = ctx->functions();
        std::sort(billboards_.begin(), billboards_.end(), [](const Billboard &a, const Billboard &b) { return a.z > b.z; });
        std::vector<float> verts;
        verts.reserve(billboards_.size() * 6 * 5);
        for (const Billboard &b : billboards_) {
            const float x0 = b.x - b.w * .5f, x1 = b.x + b.w * .5f;
            const float y0 = .02f, y1 = b.h;
            const float z = b.z;
            auto vtx = [&](float x, float y, float u, float v) {
                verts.insert(verts.end(), {x, y, z, u, v});
            };
            vtx(x0, y0, 0, 1); vtx(x1, y0, 1, 1); vtx(x1, y1, 1, 0);
            vtx(x0, y0, 0, 1); vtx(x1, y1, 1, 0); vtx(x0, y1, 0, 0);
        }
        if (!billboard_program_) return;
        billboard_vao_.bind(); billboard_vbo_.bind();
        billboard_vbo_.allocate(verts.data(), int(verts.size() * sizeof(float)));
        billboard_program_->bind();
        billboard_program_->setUniformValue("uMVP", mvp);
        billboard_program_->setUniformValue("uTex", 0);
        const int p = billboard_program_->attributeLocation("aPos");
        const int uv = billboard_program_->attributeLocation("aUV");
        billboard_program_->enableAttributeArray(p);
        billboard_program_->enableAttributeArray(uv);
        billboard_program_->setAttributeBuffer(p, GL_FLOAT, 0, 3, 5 * sizeof(float));
        billboard_program_->setAttributeBuffer(uv, GL_FLOAT, 3 * sizeof(float), 2, 5 * sizeof(float));
        f->glEnable(GL_BLEND);
        f->glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        f->glDepthMask(GL_FALSE);
        texture_->bind(0);
        f->glDrawArrays(GL_TRIANGLES, 0, GLsizei(verts.size() / 5));
        texture_->release();
        f->glDepthMask(GL_TRUE);
        f->glDisable(GL_BLEND);
        billboard_program_->disableAttributeArray(p);
        billboard_program_->disableAttributeArray(uv);
        billboard_vbo_.release();
        billboard_vao_.release();
        billboards_.clear();
    }

    void buildScene(std::vector<float> &v) {''')
# init billboard resources in initGL after vao create; add texture
s=s.replace('''        vbo_.create();
        vao_.create();
        vbo_.setUsagePattern(QOpenGLBuffer::DynamicDraw);''','''        vbo_.create();
        vao_.create();
        vbo_.setUsagePattern(QOpenGLBuffer::DynamicDraw);

        billboard_program_ = std::make_unique<QOpenGLShaderProgram>();
        const char *bvs = R"(#version 300 es
        in vec3 aPos; in vec2 aUV; uniform mat4 uMVP; out vec2 vUV;
        void main(){ vUV=aUV; gl_Position=uMVP*vec4(aPos,1.0); })";
        const char *bfs = R"(#version 300 es
        precision mediump float; in vec2 vUV; uniform sampler2D uTex; out vec4 fragColor;
        void main(){ vec4 t=texture(uTex,vUV); if(t.a<0.04) discard; fragColor=vec4(t.rgb, t.a); })";
        if (billboard_program_->addShaderFromSourceCode(QOpenGLShader::Vertex, bvs) &&
            billboard_program_->addShaderFromSourceCode(QOpenGLShader::Fragment, bfs) &&
            billboard_program_->link()) {
            billboard_vbo_.create();
            billboard_vbo_.setUsagePattern(QOpenGLBuffer::DynamicDraw);
            billboard_vao_.create();
            QImage image(QStringLiteral(":/assets/ego_car_rear.png"));
            if (!image.isNull()) {
                texture_ = new QOpenGLTexture(image.mirrored());
                texture_->setMinificationFilter(QOpenGLTexture::Linear);
                texture_->setMagnificationFilter(QOpenGLTexture::Linear);
            }
        } else {
            qWarning("billboard shader failed: %s", qPrintable(billboard_program_->log()));
            billboard_program_.reset();
        }''')
# members add before existing private members
s=s.replace('''    std::unique_ptr<QOpenGLShaderProgram> program_;
    QOpenGLBuffer vbo_{QOpenGLBuffer::VertexBuffer};''','''    std::unique_ptr<QOpenGLShaderProgram> program_;
    std::unique_ptr<QOpenGLShaderProgram> billboard_program_;
    QOpenGLTexture *texture_ = nullptr;
    std::vector<Billboard> billboards_;
    QOpenGLBuffer billboard_vbo_{QOpenGLBuffer::VertexBuffer};
    QOpenGLVertexArrayObject billboard_vao_;
    QOpenGLBuffer vbo_{QOpenGLBuffer::VertexBuffer};''')
p.write_text(s,encoding='utf-8')
print('billboards integrated')
