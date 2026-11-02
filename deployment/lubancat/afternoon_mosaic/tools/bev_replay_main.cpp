// Verification-only player: feeds recorded RKNN measurements into the real Qt renderer.
// No AI service, control transport, risk evaluation or fabricated detections.
#include "../hmi/video_item.h"
#include "../hmi/tesla_scene_item.h"
#include <QGuiApplication>
#include <QQuickView>
#include <QQmlComponent>
#include <QQmlContext>
#include <QQmlEngine>
#include <QQuickItem>
#include <QDebug>
#include <algorithm>
#include <QQuickImageProvider>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFile>
#include <QElapsedTimer>
#include <QTimer>
#include <QSurfaceFormat>
#include <chrono>

static qint64 steadyNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
class RawFrames : public QQuickImageProvider {
    QFile file_;
public:
    explicit RawFrames(const QString &path):QQuickImageProvider(QQuickImageProvider::Image),file_(path){file_.open(QIODevice::ReadOnly);}
    QImage requestImage(const QString &id,QSize *size,const QSize &) override {
        const qint64 bytes=1280*720*2;
        const qint64 count=file_.size()/bytes;
        if(count<=0)return {};
        const qint64 index=std::max<qint64>(0,id.toLongLong()-1)%count;
        file_.seek(index*bytes);const auto data=file_.read(bytes);
        if(data.size()!=bytes)return {};
        QImage full(reinterpret_cast<const uchar *>(data.constData()),1280,720,QImage::Format_RGB16);
        QImage image=full.copy(640,360,640,360);*size=image.size();return image;
    }
};
int main(int argc,char **argv) {
    QSurfaceFormat format;format.setRenderableType(QSurfaceFormat::OpenGLES);format.setDepthBufferSize(0);format.setStencilBufferSize(0);QSurfaceFormat::setDefaultFormat(format);
    QGuiApplication app(argc,argv);const auto args=app.arguments();
    if(args.size()<4)return 2;
    QFile input(args[1]);if(!input.open(QIODevice::ReadOnly))return 3;
    QVector<QJsonObject> rows;
    while(!input.atEnd()){auto o=QJsonDocument::fromJson(input.readLine()).object();if(!o.isEmpty())rows.append(o);}
    if(rows.isEmpty())return 4;
    const bool probe=args.contains("--probe");
    if(probe) {
        // Explicitly labelled test fixture derived from one recorded real bbox.
        // These modified metadata rows are excluded from real A/B measurements.
        QJsonObject sample;
        for(const auto &row:rows)for(const auto &box:row["boxes"].toArray())
            if(box.toObject()["serverTrackId"].toInt()>0 && box.toObject()["sourceId"]=="Q4")sample=box.toObject();
        if(sample.isEmpty())return 6;
        rows.clear();
        const QStringList cases={"valid","untracked","wrong_source","wrong_epoch","predicted","new_context","offline","empty_switch"};
        for(int phase=0;phase<cases.size();++phase)for(int tick=0;tick<14;++tick) {
            const qint64 t=1000000000LL+phase*1400000000LL+tick*100000000LL;
            auto box=sample;box["detectTimestampNs"]=t;
            if(phase==1){box["serverTrackId"]=0;box["trackId"]=1000000001;}
            if(phase==2)box["sourceId"]="Q1";
            if(phase==4)box["predicted"]=true;
            QJsonObject row;row["arrival_ns"]=t;row["probe"]=cases[phase];
            row["source_key"]="CONTROLLED_FIXTURE:"+(phase==6?QString("new_context"):cases[phase]);
            row["boxes"]=phase==7?QJsonArray{}:QJsonArray{box};rows.append(row);
        }
    }
    qmlRegisterType<TeslaSceneItem>("QtGpuProbe",1,0,"TeslaSceneItem");
    QQuickView view;view.engine()->addImageProvider("recording",new RawFrames(args[2]));
    view.rootContext()->setContextProperty("variantLabel",probe?QString("CONTROLLED METADATA FIXTURE"):args[3]);
    QQmlComponent component(view.engine());component.setData(R"QML(
import QtQuick 2.12
import QtGpuProbe 1.0
Rectangle {
    width:1280;height:720;color:"#061019"
    Text{x:20;y:12;text:variantLabel + " | RECORDED RKNN / LOCAL REPLAY";color:"white";font.pixelSize:23}
    Image{id:frame;objectName:"sourceImage";x:0;y:60;width:640;height:360;cache:false}
    Text{x:20;y:440;text:"Q4 source frame + recorded measurements\nSame measurements and timestamps in both renderers\nBEV is relative display only; no actuation";color:"#a0c7cf";font.pixelSize:19}
    TeslaSceneItem{objectName:"bevScene";x:640;y:60;width:640;height:640;roadQuadrant:3;displayView:"front";demoMode:false;online:true}
}
)QML",QUrl());
    QObject *root=component.create();if(!root){qCritical()<<component.errors();return 5;}
    view.setContent(QUrl(),&component,root);
    auto *scene=root->findChild<TeslaSceneItem *>("bevScene");
    auto *image=root->findChild<QObject *>("sourceImage");
    VideoItem identitySource;scene->setVideoSource(&identitySource);
    scene->setOnline(true);scene->setDisplayView("front");scene->setRoadQuadrant(3);
    const qint64 origin=rows.first()["arrival_ns"].toVariant().toLongLong();
    const qint64 start=steadyNs()+500000000;
    int index=0;QTimer timer;timer.setInterval(4);
    QObject::connect(&timer,&QTimer::timeout,&app,[&]{
        const qint64 now=steadyNs();
        while(index<rows.size() && rows[index]["arrival_ns"].toVariant().toLongLong()-origin<=now-start){
            const auto row=rows[index++];QVariantList boxes;
            for(const auto &entry:row["boxes"].toArray()){
                auto b=entry.toObject().toVariantMap();
                const qint64 timestamp=b["detectTimestampNs"].toLongLong();
                b["detectTimestampNs"]=start+(timestamp-origin);
                b["hazardExpiresNs"]=start+(b["hazardExpiresNs"].toLongLong()-origin);
                b["isTracked"]=b["serverTrackId"].toInt()>0;
                // Bind the recorded identity to this isolated player epoch.
#ifdef BEV_FIXED_PLAYER
                b["trackerEpoch"]=row["probe"].toString()=="wrong_epoch"?QString("stale-epoch"):identitySource.aiTrackingEpoch();
#endif
                boxes.append(b);
            }
            scene->setSourceKey(row["source_key"].toString());scene->setOnline(row["probe"].toString()!="offline");scene->setBoxes(boxes);
            if(!boxes.isEmpty())image->setProperty("source",QUrl("image://recording/"+boxes.first().toMap()["frameId"].toString()));
        }
        if(index==rows.size() && now-start>rows.last()["arrival_ns"].toVariant().toLongLong()-origin+1200000000){scene->setOnline(false);app.quit();}
    });timer.start();view.showFullScreen();return app.exec();
}
