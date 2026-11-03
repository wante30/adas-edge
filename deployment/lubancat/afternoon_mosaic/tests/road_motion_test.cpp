#include "../hmi/road_motion.h"
#include <cassert>
#include <cmath>
int main() {
    SchematicRoadMotion a,b;
    a.advance(1000000000,1000000000); b.advance(1000000000,1000000000);
    for(int i=1;i<=30;++i) a.advance(1000000000LL+i*1000000000LL/30,1000000000LL+i*1000000000LL/30);
    for(int i=1;i<=60;++i) b.advance(1000000000LL+i*1000000000LL/60,1000000000LL+i*1000000000LL/60);
    assert(std::abs(a.distance-6.)<1e-6 && std::abs(a.distance-b.distance)<1e-6);
    const double before=a.distance;
    a.advance(3000000000,2000000000); assert(!a.moving && a.distance==before);
    a.advance(4000000000,0); assert(!a.moving && a.distance==before);
    a.advance(4100000000,4200000000); assert(!a.moving && a.distance==before);
    a.advance(4200000000,4200000000); assert(a.moving && std::abs(a.distance-before-.6)<1e-6);
    const double resumed=a.distance;
    a.advance(4200000000,4200000000); assert(a.distance==resumed);
    a.distance=35.9;a.advance(4300000000,4300000000);assert(std::abs(a.distance-.5)<1e-6);
    // All patterns remain periodic at the shared wrap; motion is toward the ego.
    assert(a.phase(4.)>=0 && a.phase(4.)<4.);
    a.advance(9000000000,9000000000);assert(std::abs(a.distance-1.22)<1e-6);
}
