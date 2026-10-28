from pathlib import Path
h=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.h')
s=h.read_text(encoding='utf-8').replace('    Renderer *createRenderer() const override;','    QQuickFramebufferObject::Renderer *createRenderer() const override;')
h.write_text(s,encoding='utf-8')
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\tesla_scene_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('#include <QDateTime>\n','#include <QDateTime>\n#include <QDebug>\n')
s=s.replace('#include <algorithm>\n','#include <algorithm>\n#include <memory>\n')
old='''QQuickFramebufferObject::Renderer *TeslaSceneItem::createRenderer() const {
    return new class SceneRenderer();
}

'''
if old not in s: raise SystemExit('create marker')
s=s.replace(old,'',1)
s=s.replace('''        const QString key = scene->sourceKey();
        if (key != source_key_) {''','''        viewport_ = scene->size().toSize();
        const QString key = scene->sourceKey();
        if (key != source_key_) {''')
# append creator before EOF after class end; append at file end
s += '''

QQuickFramebufferObject::Renderer *TeslaSceneItem::createRenderer() const {
    return new SceneRenderer();
}
'''
p.write_text(s,encoding='utf-8')
print('cpp patched')
