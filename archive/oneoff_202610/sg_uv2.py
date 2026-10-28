from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_texture_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('node->setSourceRect(QRectF(0, 0, 1, 1));','node->setSourceRect(QRectF(0, 0, textureSize_.width(), textureSize_.height()));',1)
p.write_text(s,encoding='utf-8')
print('pixel source rect')
