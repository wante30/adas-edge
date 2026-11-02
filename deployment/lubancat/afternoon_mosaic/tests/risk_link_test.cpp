#include "../hmi/risk_link.h"
#include <cassert>
int main() {
    const qint64 now=1000000000;
    QVariantMap b{{"trackId",42},{"x",100.},{"y",100.},{"width",100.},{"height",100.},{"className","car"}};
    QVariantList boxes{b};
    QJsonObject e{{"event_id","test:1"},{"valid",true},{"lifecycle_state","ACTIVE"},
        {"severity","warning"},{"inference_mode","MOSAIC_FULL"},{"mode_epoch","1"},
        {"coordinate_space","MOSAIC_1280x720"},{"expires_at_ns",double(now+400000000)},
        {"source_id","Q1"},{"target_class","car"},{"track_id",7},{"event_type","FRONT_COLLISION_RISK"},
        {"bbox",QJsonArray{100,100,200,200}}};
    auto events=RiskLink::associate(QJsonArray{e},boxes,now,"MOSAIC_FULL","1");
    assert(events.size()==1 && boxes[0].toMap().value("hazardSeverity")=="warning");
    assert(RiskLink::expire(events,boxes,now+500000000));
    assert(events.empty() && !boxes[0].toMap().value("hazard").toBool());
    boxes={b};e.insert("source_id","Q2");
    events=RiskLink::associate(QJsonArray{e},boxes,now,"MOSAIC_FULL","1");
    assert(events.size()==1 && !events[0].toMap().value("matched").toBool());
    assert(!boxes[0].toMap().value("hazard").toBool());
    e.insert("source_id","Q1");b.insert("serverTrackId",8);boxes={b};
    events=RiskLink::associate(QJsonArray{e},boxes,now,"MOSAIC_FULL","1");
    assert(!events[0].toMap().value("matched").toBool());
    assert(RiskLink::associate(QJsonArray{e},boxes,now,"MOSAIC_FULL","2").empty());
    // Q4 is the physical forward source even in the legacy ROAD_Q1 mode.
    e.insert("source_id","Q4"); e.insert("severity","critical");
    e.insert("inference_mode","ROAD_Q1");e.insert("coordinate_space","ROAD_VIEW_1280x720");
    b.insert("sourceId","Q4");b.insert("serverTrackId",7);boxes={b};
    events=RiskLink::associate(QJsonArray{e},boxes,now,"ROAD_Q1","1");
    assert(events.size()==1 && events[0].toMap().value("matched").toBool());
    assert(boxes[0].toMap().value("hazardSeverity")=="critical");
    b.insert("sourceId","Q1");boxes={b};
    events=RiskLink::associate(QJsonArray{e},boxes,now,"ROAD_Q1","1");
    assert(!events[0].toMap().value("matched").toBool());
    return 0;
}
