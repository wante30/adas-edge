from pathlib import Path
h=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_texture_item.h')
s=h.read_text(encoding='utf-8')
s=s.replace('''class VideoItem;
class QSGSimpleTextureNode;''','''class VideoItem;
class QSGSimpleTextureNode;
class QSGTexture;''',1)
s=s.replace('''    class NativeTexture *texture_ = nullptr;''','''    QSGTexture *texture_ = nullptr;''',1)
h.write_text(s,encoding='utf-8')
c=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_texture_item.cpp')
s=c.read_text(encoding='utf-8')
s=s.replace('''class VideoTextureItem::NativeTexture : public QSGTexture {''','''class NativeTexture : public QSGTexture {''',1)
c.write_text(s,encoding='utf-8')
print('fixed native texture')
