from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_texture_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('''        const QRectF view = video_->sourceViewport();
        node->setSourceRect(QRectF(view.left(), view.top(), view.width(), view.height()));''','''        // Temporary full-frame source rect for texture-path validation.
        node->setSourceRect(QRectF(0, 0, 1, 1));''',1)
p.write_text(s,encoding='utf-8')
print('full source rect')
