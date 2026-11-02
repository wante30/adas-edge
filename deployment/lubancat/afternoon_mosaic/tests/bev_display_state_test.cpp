#include "../hmi/bev_display_state.h"
#include <cassert>
#include <cmath>
int main() {
    BevDisplay::State s;
    assert(s.observe(0,20,1000000000));
    assert(!s.observe(9,50,1000000000)); // repeated synchronization is not a new measurement
    assert(!s.observe(9,50,999999999)); // reject old results
    assert(s.observe(2,30,1100000000));
    assert(s.observe(2,30,1200000000));
    for(int i=0;i<60;++i) {s.advance(1.0/60); assert(s.x>=0 && s.x<=2 && s.z>=20 && s.z<=30);}
    assert(!s.expired(1900000000)); // a short missing measurement retains the same object
    assert(s.fade(1900000000)>0 && s.fade(1900000000)<1);
    assert(s.expired(2300000000));
    BevDisplay::State a,b;
    a.observe(0,20,1000000000); b.observe(0,20,1000000000);
    a.observe(2,30,1100000000); b.observe(2,30,1100000000);
    a.observe(2,30,1200000000); b.observe(2,30,1200000000);
    for(int i=0;i<60;++i)a.advance(1.0/60);
    for(int i=0;i<30;++i)b.advance(1.0/30);
    assert(std::abs(a.z-b.z)<1e-4); // independent of render frame rate
    assert(!s.observe(NAN,0,3000000000));
    BevDisplay::State outlier;outlier.observe(0,20,1000000000);
    outlier.observe(6,60,1100000000);outlier.advance(.1);
    assert(outlier.tx==0 && outlier.tz==20); // one width/depth outlier rejected
    outlier.observe(0,20,1200000000);outlier.advance(.1);
    assert(outlier.x==0 && outlier.z==20);
    outlier=BevDisplay::State{};outlier.observe(0,20,1200000000);
    outlier.observe(6,60,1300000000);outlier.observe(6,60,1400000000);
    assert(outlier.tx<=.40001f && outlier.tz<=21.20001f); // actual measurement dt bounds updates
    const float before=outlier.tz;outlier.observe(6,60,2200000000);
    assert(outlier.tz-before<=3.0001f); // reacquisition after a gap cannot jump

}
