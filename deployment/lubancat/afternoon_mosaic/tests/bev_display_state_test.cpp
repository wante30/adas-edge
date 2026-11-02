#include "../hmi/bev_display_state.h"
#include <cassert>
#include <cmath>
int main() {
    BevDisplay::State s;
    assert(s.observe(0,20,1000000000));
    assert(!s.observe(9,50,1000000000)); // repeated synchronization is not a new measurement
    assert(!s.observe(9,50,999999999)); // reject old results
    assert(s.observe(2,30,1100000000));
    for(int i=0;i<60;++i) {s.advance(1.0/60); assert(s.x>=0 && s.x<=2 && s.z>=20 && s.z<=30);}
    assert(!s.expired(1900000000)); // a short missing measurement retains the same object
    assert(s.fade(1900000000)>0 && s.fade(1900000000)<1);
    assert(s.expired(2200000000));
    BevDisplay::State a,b;
    a.observe(0,20,1000000000); b.observe(0,20,1000000000);
    a.observe(2,30,1100000000); b.observe(2,30,1100000000);
    for(int i=0;i<60;++i)a.advance(1.0/60);
    for(int i=0;i<30;++i)b.advance(1.0/30);
    assert(std::abs(a.z-b.z)<1e-4); // independent of render frame rate
    assert(!s.observe(NAN,0,3000000000));
}
