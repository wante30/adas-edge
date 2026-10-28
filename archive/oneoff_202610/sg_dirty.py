from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_texture_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('''        node->setTexture(texture_);
        node->setOwnsTexture(false);''','''        node->setTexture(texture_);
        node->setOwnsTexture(false);
        node->markDirty(QSGNode::DirtyMaterial | QSGNode::DirtyGeometry);''',1)
p.write_text(s,encoding='utf-8')
print('mark node dirty')
