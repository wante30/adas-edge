#include "../hmi/vehicle_telemetry_status.h"
#include <QCoreApplication>
#include <QTemporaryDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QEventLoop>
#include <QTimer>
#include <cassert>
static void waitPoll() { QEventLoop loop; QTimer::singleShot(220,&loop,&QEventLoop::quit); loop.exec(); }
int main(int argc,char **argv) {
    QCoreApplication app(argc,argv);QTemporaryDir folder;
    const QString path=folder.path()+"/status.json";qputenv("LVT1_STATUS_FILE",path.toUtf8());
    QJsonObject value{{"schema","pango.rk3566.vehicle_status.v1"},{"online",true},
        {"sensor_stale",false},{"simulated",false},{"generation",1},{"lane_source","hsst"},
        {"current_action","none"},{"active",false},{"decision","idle"}};
    auto write=[&](){QFile f(path);assert(f.open(QIODevice::WriteOnly));f.write(QJsonDocument(value).toJson());f.close();};
    write();VehicleTelemetryStatus status;assert(status.online());int visible=0,raw=0;
    QObject::connect(&status,&VehicleTelemetryStatus::statusChanged,[&](){++visible;});
    QObject::connect(&status,&VehicleTelemetryStatus::snapshotChanged,[&](){++raw;});
    value.insert("generation",2);write();waitPoll();assert(raw==1 && visible==0);
    value.insert("current_action","brake");value.insert("active",true);value.insert("brake_allow",true);
    value.insert("decision","allow");write();waitPoll();assert(visible==1 && status.egoState()["action"]=="brake");
    QFile::remove(path);waitPoll();assert(!status.online() && visible==2);
}
