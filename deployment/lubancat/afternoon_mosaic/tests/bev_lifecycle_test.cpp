#include "../hmi/bev_lifecycle.h"
#include <cassert>
using namespace BevLife;
Observation observation(int id,Time t,double bx=.4,int cls=2) {
    Observation o;o.serverId=id;o.measuredNs=t;o.confidence=.8;o.bx=bx;o.by=.5;o.bw=.12;o.bh=.2;o.x=(bx+.06-.5)*14-1.75;o.z=20;o.cls=cls;return o;
}
int main() {
    Tracker tracker;Time t=1000000000;
    auto ids=tracker.update({observation(1,t)},t,t);
    assert(ids[0]>0 && !tracker.tracks[0].confirmed && tracker.tracks[0].opacity(t)==0);
    t+=100000000;tracker.update({observation(1,t,.405)},t,t);
    assert(tracker.tracks[0].confirmed);
    const int sceneId=tracker.tracks[0].id;
    tracker.advance(t+300000000,.1);assert(tracker.tracks[0].opacity(t+300000000)>.99);
    // The same measured target can regain a new upstream ID after a short miss.
    t+=400000000;ids=tracker.update({observation(7,t,.42)},t,t);
    assert(ids[0]==sceneId && tracker.tracks.size()==1 && tracker.reassociations==1);
    // Duplicate render/sync calls do not add hits or confirm a single observation.
    int hits=tracker.tracks[0].hits;tracker.update({observation(7,t,.42)},t,t+1000);assert(tracker.tracks[0].hits==hits);
    // A nearby person cannot inherit a vehicle track.
    t+=100000000;ids=tracker.update({observation(8,t,.42,0)},t,t);assert(ids[0]!=sceneId);
    tracker.advance(t+700000000,.1);assert(tracker.tracks.size()==1);assert(tracker.tracks[0].opacity(t+700000000)>0 && tracker.tracks[0].opacity(t+700000000)<1);
    tracker.advance(t+1500000000,.1);assert(tracker.tracks.empty());
    Tracker crossing;t=1000000000;
    crossing.update({observation(1,t,.3),observation(2,t,.5)},t,t);
    t+=100000000;auto same=crossing.update({observation(2,t,.48),observation(1,t,.32)},t,t);assert(same[0]==2 && same[1]==1);
    // Geometric reassociation must remain one-to-one.
    t+=100000000;auto reacq=crossing.update({observation(9,t,.34),observation(10,t,.46)},t,t);assert(reacq[0]!=reacq[1] && crossing.tracks.size()==2);
    crossing.clear();assert(crossing.tracks.empty());
    // One-frame detections never become a visible 3D object.
    crossing.update({observation(40,t)},t,t);crossing.advance(t+700000000,.05);assert(crossing.tracks.empty() && crossing.discardedTentative==1);
    // Global matching is not greedy: taking the first edge would lose a match.
    auto matching=assign({{.1,.2,1.,1.},{.11,.9,1.,1.}});assert(matching[0]==1 && matching[1]==0);
    // Reacquisition preserves current alpha, then recovers over time. It must
    // not jump from a nearly faded person/vehicle to a fully opaque model.
    Tracker fading;t=1000000000;
    fading.update({observation(1,t)},t,t);
    t+=100000000;fading.update({observation(1,t)},t,t);
    const Time returned=t+950000000;
    const float oldAlpha=fading.tracks[0].opacity(returned);
    assert(oldAlpha>0 && oldAlpha<.6);
    fading.update({observation(2,returned)},returned,returned);
    assert(fading.tracks.size()==1 && fading.reassociations==1);
    assert(std::abs(fading.tracks[0].opacity(returned)-oldAlpha)<.0001);
    assert(fading.tracks[0].opacity(returned+100000000)>oldAlpha);
    assert(fading.tracks[0].opacity(returned+200000000)>.99);
    // Fully expired tracks cannot be resurrected by an upstream ID reuse.
    t=returned+2000000000;
    auto reborn=fading.update({observation(2,t)},t,t);
    assert(reborn[0]!=1 && !fading.tracks[0].confirmed);
    Tracker invalid;t=1000000000;
    auto malformed=observation(1,t);malformed.x=NAN;
    invalid.update({malformed,observation(0,t)},t,t);
    assert(invalid.tracks.empty());
    t+=100000000;invalid.update({observation(1,t)},t,t);
    malformed=observation(1,t+100000000);malformed.bw=INFINITY;
    invalid.update({malformed},t+100000000,t+100000000);
    assert(invalid.tracks[0].hits==1 && std::isfinite(invalid.tracks[0].drawX));
    Axis kf;kf.p=0;
    for(int i=1;i<20;++i){kf.predict(.1,2);kf.correct(i*.1,.12);}
    assert(kf.v>.7 && kf.v<1.3);const double before=kf.p;kf.predict(.2,2);assert(kf.p>before);
}
