"""Add identical opt-in display diagnostics to baseline and candidate source."""
from pathlib import Path
import argparse

def instrument(root):
    path=Path(root)/"hmi/tesla_scene_item.cpp"
    s=path.read_text(encoding="utf-8")
    if "QT_BEV_TRACE" in s:return
    s=s.replace("    int id = 0;", "    int id = 0;\n    int serverId = 0;\n    QString epoch;")
    s=s.replace("    double born=0;", "    double born=0;\n    SourceBox lastBox;\n    float rawX=0,rawZ=0;")
    s=s.replace("            b.cls = map.value", "            b.serverId = map.value(\"serverTrackId\").toInt();\n            b.epoch = map.value(\"trackerEpoch\").toString();\n            b.cls = map.value",1)
    marker="    // Auditable bridge from real tracker output"
    i=s.index(marker)
    s=s[:i]+'''    static QFile measurements(qEnvironmentVariable("QT_BEV_MEASUREMENTS"));
    if (!measurements.fileName().isEmpty() && (measurements.isOpen() || measurements.open(QIODevice::WriteOnly))) {
        const qint64 nowNs=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        QJsonObject row;row["arrival_ns"]=nowNs;row["source_key"]=sourceKey();
        row["boxes"]=QJsonArray::fromVariantList(boxes);
        measurements.write(QJsonDocument(row).toJson(QJsonDocument::Compact)+'\\n');measurements.flush();
    }
'''+s[i:]
    s=s.replace("car.hazard=b.hazard;live_.insert(b.id,car);", "car.hazard=b.hazard;live_.insert(b.id,car);\n                ++births_[source_key_+QString::number(b.id)];")
    marker="        for (auto it=live_.begin();it!=live_.end();)"
    i=s.index(marker,s.index("    void addLive("))
    # Insert after the per-measurement update, before closing its loop.
    j=s.rfind("        }",0,i)
    s=s[:j]+'''            auto stored=live_.find(b.id);
            if(stored!=live_.end()) {stored->lastBox=b;stored->rawX=tx;stored->rawZ=tz;}
'''+s[j:]
    marker="            const size_t start=v.size();"
    s=s.replace(marker,'''            if (!trace_.fileName().isEmpty() && (trace_.isOpen() || trace_.open(QIODevice::WriteOnly))) {
                const auto &b=it->lastBox;
                QJsonObject row;row["render_ns"]=absoluteNs;row["visual_dt"]=dt_;row["capture_timestamp_ns"]=b.measuredNs;
                row["track_id"]=it.key();row["server_track_id"]=b.serverId;row["source_id"]=b.source;row["epoch"]=b.epoch;
                row["source_key"]=source_key_;row["bbox"]=QJsonArray{b.x,b.y,b.w,b.h};row["bbox_width"]=b.w;
                row["raw_x"]=it->rawX;row["raw_z"]=it->rawZ;row["filtered_x"]=it->x;row["filtered_z"]=it->z;
                row["measurement_age_s"]=double(absoluteNs-b.measuredNs)/1e9;row["fade"]=fade;
                row["birth_count"]=births_.value(source_key_+QString::number(it.key()));
                row["display_only_hold"]=absoluteNs-b.measuredNs>200000000LL;
                trace_.write(QJsonDocument(row).toJson(QJsonDocument::Compact)+'\\n');
            }
'''+marker,1)
    s=s.replace("    QHash<int, LiveCar> live_;", "    QHash<int, LiveCar> live_;\n    QFile trace_{qEnvironmentVariable(\"QT_BEV_TRACE\")};\n    QHash<QString,int> births_;")
    # Python literal escaping: produce C++ character newline, not a multi-character constant.
    s=s.replace("'\\\\n'", "'\\n'")
    path.write_text(s,encoding="utf-8")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root");instrument(p.parse_args().root)
