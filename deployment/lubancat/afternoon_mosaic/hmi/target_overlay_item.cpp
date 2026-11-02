#include "video_item.h"
#include "target_overlay_item.h"
#include <QPainter>
#include <QFontMetrics>
#include <QQuickWindow>
#include <QSGGeometryNode>
#include <QSGVertexColorMaterial>
#include <QSGSimpleTextureNode>
#include <QTimer>
#include <algorithm>
#include <cmath>

namespace {
struct OverlayNode : QSGGeometryNode {
    struct Label : QSGSimpleTextureNode { QString text; QColor color; };
    QVector<Label *> labels;
    QHash<QString,QSGTexture *> textures;
    ~OverlayNode() override { qDeleteAll(textures); }
    OverlayNode() {
        setGeometry(new QSGGeometry(QSGGeometry::defaultAttributes_ColoredPoint2D(),0));
        geometry()->setDrawingMode(GL_TRIANGLES);
        setFlag(OwnsGeometry);
        setMaterial(new QSGVertexColorMaterial);
        setFlag(OwnsMaterial);
    }
};
QString labelFor(const QVariantMap &b) {
    const QString kind=b.value("hazardKind").toString().toUpper();
    if (b.value("hazard").toBool()) {
        static const QHash<QString,QString> labels={
            {"PEDESTRIAN_NEAR_CROSSWALK",QString::fromUtf8("斑马线附近行人")},
            {"CROSSWALK_BRAKE_PULSE",QString::fromUtf8("斑马线点刹 · DRY RUN")},
            {"TWO_WHEELER_PATH_CONFLICT",QString::fromUtf8("两轮车路径冲突")},
            {"ONCOMING_PATH_CONFLICT",QString::fromUtf8("对向车辆路径冲突")},
            {"LEAD_RAPID_CLOSING",QString::fromUtf8("前车快速接近")},
            {"FRONT_COLLISION_RISK",QString::fromUtf8("前方碰撞风险")},
            {"PEDESTRIAN_CROSSING_PATH",QString::fromUtf8("行人横穿行驶区域")},
            {"PEDESTRIAN_IN_PATH",QString::fromUtf8("前方行人风险")},
            {"RED_SIGNAL_CONFIRMED",QString::fromUtf8("确认红灯 · 注意信号")},
            {"LEAD_BRAKE_LAMP_RISE",QString::fromUtf8("前车制动灯变化")},
            {"SUSPECTED_BRAKING_WITH_CLOSING",QString::fromUtf8("前车制动且快速接近")}
        };
        if(labels.contains(kind))return labels.value(kind);
        if(kind.contains("NEAR")||kind.contains("CLOSE"))return b.value("hazardTargetClass").toString()=="person"
            ?QString::fromUtf8("行人距离过近"):QString::fromUtf8("前车距离过近");
        if(kind.contains("PEDESTRIAN")||b.value("hazardTargetClass").toString()=="person")return QString::fromUtf8("前方行人风险");
        if(kind.contains("COLLISION"))return QString::fromUtf8("前方碰撞风险");
        if(kind.contains("CUT_IN"))return QString::fromUtf8("车辆切入风险");
        return QString::fromUtf8("危险事件 · ")+QString(kind).replace('_',' ');
    }
    QString cls=b.value("className").toString().toUpper();
    if(cls=="TRAFFIC LIGHT") cls="SIGNAL";
    return cls+"  "+QString::number(qRound(b.value("confidence").toDouble()*100))+"%";
}
void rect(QVector<QSGGeometry::ColoredPoint2D> &v,float x,float y,float w,float h,QColor c) {
    const float pts[12]={x,y,x+w,y,x,y+h,x,y+h,x+w,y,x+w,y+h};
    for(int i=0;i<6;++i) { QSGGeometry::ColoredPoint2D p;
        p.set(pts[i*2],pts[i*2+1],qRound(c.red()*c.alphaF()),qRound(c.green()*c.alphaF()),qRound(c.blue()*c.alphaF()),c.alpha());v.append(p); }
}
}
TargetOverlayItem::TargetOverlayItem(QQuickItem *parent):QQuickItem(parent) {
    setFlag(ItemHasContents);
    setClip(true);
    auto timer=new QTimer(this);timer->setInterval(33);
    connect(timer,&QTimer::timeout,this,[this]{if(isVisible()&&!boxes_.isEmpty())update();});
    timer->start();
}
void TargetOverlayItem::setBoxes(const QVariantList &value) { boxes_=value;update(); }
QSGNode *TargetOverlayItem::updatePaintNode(QSGNode *old,UpdatePaintNodeData *) {
    auto node=static_cast<OverlayNode *>(old);
    if(!node)node=new OverlayNode;
    if(node->textures.size()>1024) {
        for(auto label:node->labels) {node->removeChildNode(label);delete label;}
        node->labels.clear();qDeleteAll(node->textures);node->textures.clear();
    }
    QVector<QSGGeometry::ColoredPoint2D> vertices;vertices.reserve(boxes_.size()*30);
    const double sx=width()/1280.,sy=height()/720.;
    int used=0;
    for(const QVariant &value:boxes_) {
        const QVariantMap b=value.toMap();
        double prediction=0;
        const qint64 detected=b.value("detectTimestampNs").toLongLong();
        if(tracking_mode_&&detected>0&&display_ns_>detected)
            prediction=std::min(prediction_ms_/1000.,(display_ns_-detected)/1e9);
        if(tracking_mode_==1) {
            const double lag=std::max(0.,double(display_frame_)-b.value("frameId").toDouble());
            prediction=std::min(prediction_ms_/1000.,std::min(4.,lag)/std::max(1.,source_fps_));
        }
        const double x=(b.value("x").toDouble()+b.value("vx").toDouble()*prediction)*sx;
        const double y=(b.value("y").toDouble()+b.value("vy").toDouble()*prediction)*sy;
        const double w=std::max(1.,b.value("width").toDouble()+b.value("vw").toDouble()*prediction)*sx;
        const double h=std::max(1.,b.value("height").toDouble()+b.value("vh").toDouble()*prediction)*sy;
        if(!std::isfinite(x+y+w+h)||w<=8||h<=8)continue;
        const QString severity=b.value("hazardSeverity").toString();
        const bool hazard=b.value("hazard").toBool()&&(severity=="warning"||severity=="critical");
        QColor color=hazard?QColor(severity=="critical"?"#FF5064":"#FFD34D"):
            QColor(b.value("className").toString()=="traffic light"?"#FF5064":"#00E676");
        const QString state=b.value("state").toString();
        const double opacity=state=="predicted"?.42:(state=="tentative"?.62:1.);
        const bool primary=used==0;
        QColor fill=color;fill.setAlphaF((primary?.045:.012)*opacity);
        rect(vertices,x,y,w,h,fill);
        QColor border=color;border.setAlphaF((hazard?1.:(primary?.62:.34))*opacity);
        const float t=hazard?2.2f:(primary?1.2f:.8f);
        rect(vertices,x,y,w,t,border);rect(vertices,x,y+h-t,w,t,border);
        rect(vertices,x,y,t,h,border);rect(vertices,x+w-t,y,t,h,border);
        while(node->labels.size()<=used) { auto l=new OverlayNode::Label;l->setOwnsTexture(false);
            node->labels.append(l);node->appendChildNode(l); }
        auto l=node->labels[used++];
        const QString text=labelFor(b);
        if(l->text!=text||l->color!=color||!l->texture()) {
            const QString key=font_family_+"|"+text+"|"+color.name(QColor::HexArgb);
            QSGTexture *texture=node->textures.value(key,nullptr);
            if(!texture) {
            QFont font(font_family_,9);font.setPixelSize(9);font.setWeight(QFont::DemiBold);
            QFontMetrics fm(font);QImage image(std::min(190,14+fm.horizontalAdvance(text)),18,QImage::Format_ARGB32_Premultiplied);
            image.fill(QColor("#E8020C12"));QPainter painter(&image);
            painter.setPen(color);painter.drawRect(0,0,image.width()-1,17);painter.setFont(font);
            painter.drawText(QRect(7,0,image.width()-14,18),Qt::AlignVCenter,text);painter.end();
            texture=window()->createTextureFromImage(image);
            node->textures.insert(key,texture);
            }
            l->setTexture(texture);
            l->text=text;l->color=color;
        }
        l->setRect(x,y-19,l->texture()->textureSize().width(),18);
    }
    while(node->labels.size()>used) { auto l=node->labels.takeLast();node->removeChildNode(l);delete l; }
    node->geometry()->allocate(vertices.size());
    std::copy(vertices.begin(),vertices.end(),node->geometry()->vertexDataAsColoredPoint2D());
    node->markDirty(QSGNode::DirtyGeometry);
    return node;
}

void TargetOverlayItem::setVideoSource(QObject *value) {
    if(video_source_==value)return;
    if(video_source_)disconnect(video_source_,nullptr,this,nullptr);
    video_source_=value;
    if(auto video=qobject_cast<VideoItem *>(value)) {
        connect(video,&VideoItem::aiBoxesChanged,this,[this,video]{setBoxes(video->aiBoxes());});
        setBoxes(video->aiBoxes());
    }
}
