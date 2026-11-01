#pragma once
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>
#include <QVariantMap>
#include <QFile>
#include <QSet>
#include <cmath>
#include <algorithm>

// The receiver validates identity and lifetime, but never reclassifies a risk.
namespace RiskLink {
inline int rank(const QString &s) { return s == "critical" ? 2 : (s == "warning" ? 1 : 0); }
inline QString source(const QVariantMap &b, const QString &mode) {
    if (b.contains("sourceId")) return b.value("sourceId").toString();
    if (mode == "ROAD_Q1") return "Q1";
    const double cx=b.value("x").toDouble()+b.value("width").toDouble()/2;
    const double cy=b.value("y").toDouble()+b.value("height").toDouble()/2;
    return QString("Q%1").arg(1+(cx>=640?1:0)+(cy>=360?2:0));
}
inline QVariantList associate(const QJsonArray &events, QVariantList &boxes, qint64 now,
                              const QString &mode, const QString &epoch) {
    QVariantList active;
    for (const QJsonValue &value : events) {
        const QJsonObject e=value.toObject();
        const QString severity=e.value("severity").toString();
        const QString space=mode=="ROAD_Q1"?"ROAD_VIEW_1280x720":"MOSAIC_1280x720";
        if (e.value("event_id").toString().isEmpty() || !e.value("valid").toBool()
            || e.value("lifecycle_state").toString()!="ACTIVE" || !rank(severity)
            || e.value("inference_mode").toString()!=mode
            || e.value("mode_epoch").toString()!=epoch
            || e.value("coordinate_space").toString()!=space
            || e.value("expires_at_ns").toDouble()<=now) continue;
        QVariantMap detail=e.toVariantMap();
        const auto raw=e.value("bbox").toArray();
        int best=-1; double bestScore=.35;
        if (raw.size()==4) {
            const double x1=raw[0].toDouble(), y1=raw[1].toDouble();
            const double x2=raw[2].toDouble(), y2=raw[3].toDouble();
            if (!std::isfinite(x1+y1+x2+y2) || x2<=x1 || y2<=y1) continue;
            for (int i=0;i<boxes.size();++i) {
                const auto b=boxes[i].toMap();
                if (source(b,mode)!=e.value("source_id").toString()
                    || b.value("predicted").toBool() || b.value("missed").toInt()>0) continue;
                if (b.contains("serverTrackId")) {
                    if (b.value("serverTrackId").toInt()==e.value("track_id").toInt()) { best=i; break; }
                    continue; // An explicit identity must never fall back to another track.
                }
                const QString cls=e.value("target_class").toString();
                if (!cls.isEmpty() && cls.compare(b.value("className").toString(),Qt::CaseInsensitive)) continue;
                const double ax=b.value("x").toDouble(), ay=b.value("y").toDouble();
                const double bx=ax+b.value("width").toDouble(), by=ay+b.value("height").toDouble();
                const double inter=std::max(0.,std::min(bx,x2)-std::max(ax,x1))*std::max(0.,std::min(by,y2)-std::max(ay,y1));
                const double iou=inter/std::max(1e-9,(bx-ax)*(by-ay)+(x2-x1)*(y2-y1)-inter);
                if(iou>bestScore) {bestScore=iou;best=i;}
            }
        }
        detail.insert("matched",best>=0);
        if (best>=0) {
            auto b=boxes[best].toMap();
            detail.insert("qt_track_id",b.value("trackId"));
            if(rank(severity)>=rank(b.value("hazardSeverity").toString())) {
                b.insert("hazard",true); b.insert("hazardKind",e.value("event_type").toString());
                b.insert("hazardSeverity",severity); b.insert("hazardEventId",e.value("event_id").toString());
                b.insert("hazardTrackId",e.value("track_id").toInt());
                b.insert("hazardTargetClass",e.value("target_class").toString());
                b.insert("hazardExpiresNs",e.value("expires_at_ns").toDouble()); boxes[best]=b;
            }
        }
        active.append(detail);
    }
    return active;
}
inline bool expire(QVariantList &events,QVariantList &boxes,qint64 now) {
    bool changed=false; QSet<QString> ids;
    for(int i=events.size()-1;i>=0;--i) {
        const auto e=events[i].toMap();
        if(e.value("expires_at_ns").toLongLong()<=now){events.removeAt(i);changed=true;}
        else ids.insert(e.value("event_id").toString());
    }
    for(int i=0;i<boxes.size();++i) {
        auto b=boxes[i].toMap();
        if(b.value("hazard").toBool() && !ids.contains(b.value("hazardEventId").toString())) {
            b.insert("hazard",false); b.insert("hazardSeverity","");b.insert("hazardKind","");
            b.insert("hazardEventId",""); boxes[i]=b; changed=true;
        }
    }
    return changed;
}
inline void logApplied(const QVariantList &events,const QVariantList &boxes,qint64 now) {
    static QFile file(qEnvironmentVariable("QT_RISK_TIMELINE_LOG"));
    if(file.fileName().isEmpty()) return;
    if(!file.isOpen() && !file.open(QIODevice::Append)) return;
    static QByteArray previous;
    QJsonObject state;state.insert("events",QJsonArray::fromVariantList(events));
    QJsonArray marked;
    for(const auto &box:boxes) {const auto b=box.toMap();if(b.value("hazard").toBool()) {
        QJsonObject m;m.insert("track_id",b.value("trackId").toInt());m.insert("event_id",b.value("hazardEventId").toString());
        m.insert("severity",b.value("hazardSeverity").toString());marked.append(m);}}
    state.insert("marked_boxes",marked);
    QJsonArray unmatched;
    for(const auto &event:events) {const auto e=event.toMap();if(!e.value("matched").toBool())unmatched.append(e.value("event_id").toString());}
    state.insert("unmatched_hazard",unmatched);
    const QByteArray signature=QJsonDocument(state).toJson(QJsonDocument::Compact);
    if(signature==previous) return;
    previous=signature; state.insert("ui_applied_ns",double(now));state.insert("kind","ui_applied");
    file.write(QJsonDocument(state).toJson(QJsonDocument::Compact)+'\n');
    file.flush();
}
}
