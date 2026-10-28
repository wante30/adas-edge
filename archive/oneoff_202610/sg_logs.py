from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_texture_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('''VideoTextureItem::VideoTextureItem(QQuickItem *parent) : QQuickItem(parent) {
    setFlag(ItemHasContents, true);
}''','''VideoTextureItem::VideoTextureItem(QQuickItem *parent) : QQuickItem(parent) {
    setFlag(ItemHasContents, true);
    qWarning() << "VideoTextureItem constructed";
}''',1)
s=s.replace('''QSGNode *VideoTextureItem::updatePaintNode(QSGNode *oldNode, UpdatePaintNodeData *) {
    if (!video_) return oldNode;''','''QSGNode *VideoTextureItem::updatePaintNode(QSGNode *oldNode, UpdatePaintNodeData *) {
    qWarning() << "VideoTextureItem updatePaintNode size" << width() << height() << "old" << (oldNode != nullptr);
    if (!video_) return oldNode;''',1)
s=s.replace('''            if (QSGTexture *test = window()->createTextureFromImage(image)) {''','''            qWarning() << "SG_TEST_IMAGE creating" << width() << height();
            if (QSGTexture *test = window()->createTextureFromImage(image)) {''',1)
p.write_text(s,encoding='utf-8')
print('logs')
