#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

// Display-only tracks. IDs and predictions never leave the renderer for control.
namespace BevLife {
using Time = std::int64_t;
inline double seconds(Time ns) { return double(ns) / 1e9; }
inline int family(int cls) { return cls == 2 || cls == 5 || cls == 7 ? 2 : cls; }
struct Observation {
    int serverId=0, cls=2;
    double confidence=0, bx=0, by=0, bw=0, bh=0, x=0, z=0;
    Time measuredNs=0;
};
struct Axis {
    double p=0,v=0,p00=1,p01=0,p10=0,p11=16;
    void predict(double dt, double acceleration) {
        p += v*dt;
        const double a=p00+dt*(p01+p10)+dt*dt*p11;
        const double b=p01+dt*p11,c=p10+dt*p11;
        const double q=acceleration*acceleration;
        p00=a+q*dt*dt*dt*dt/4;p01=b+q*dt*dt*dt/2;
        p10=c+q*dt*dt*dt/2;p11+=q*dt*dt;
    }
    void correct(double value,double variance) {
        const double s=p00+variance,k0=p00/s,k1=p10/s,innovation=value-p;
        const double a=p00,b=p01;
        p+=k0*innovation;v+=k1*innovation;
        p00-=k0*a;p01-=k0*b;p10-=k1*a;p11-=k1*b;
    }
};
struct Track {
    int id=0,serverId=0,cls=2,hits=0,reassociated=0;
    Observation last;
    Time firstNs=0,lastNs=0,revealNs=0;
    bool confirmed=false;
    double drawX=0,drawZ=0,boxVx=0,boxVy=0;
    double revealFrom=0;
    Axis ax,az;
    std::array<double,3> xs{},zs{};
    unsigned cursor=0;
    static double median(std::array<double,3> values) { std::sort(values.begin(),values.end());return values[1]; }
    double age(Time now) const { return std::max(0.,seconds(now-lastNs)); }
    void observe(const Observation &o,Time arrival) {
        if(o.measuredNs<=lastNs)return;
        // Reacquiring a fading track must not snap its opacity back to one.
        // Sample the old envelope before replacing its measurement timestamp.
        if(confirmed && age(arrival)>.55) {
            revealFrom=opacity(arrival);
            revealNs=arrival;
        }
        if(!hits) {
            ax.p=drawX=o.x;az.p=drawZ=o.z;xs.fill(o.x);zs.fill(o.z);firstNs=o.measuredNs;
        } else {
            const double dt=std::clamp(seconds(o.measuredNs-lastNs),.001,.6);
            const double k=1-std::exp(-dt/.25);
            boxVx+=k*(std::clamp(((o.bx+o.bw/2)-(last.bx+last.bw/2))/dt,-.6,.6)-boxVx);
            boxVy+=k*(std::clamp(((o.by+o.bh/2)-(last.by+last.bh/2))/dt,-.6,.6)-boxVy);
            xs[cursor]=o.x;zs[cursor]=o.z;cursor=(cursor+1)%3;
            ax.predict(dt,2.0);az.predict(dt,5.0);
            // Median and innovation limits reject isolated detector scale spikes.
            ax.correct(std::clamp(median(xs),ax.p-1.2,ax.p+1.2),.12);
            az.correct(std::clamp(median(zs),az.p-5.,az.p+5.),1.5);
            ax.v=std::clamp(ax.v,-4.,4.);az.v=std::clamp(az.v,-12.,12.);
        }
        if(serverId && serverId!=o.serverId)++reassociated;
        serverId=o.serverId;cls=o.cls;last=o;lastNs=o.measuredNs;++hits;
        if(!confirmed && hits>=2) { confirmed=true;revealNs=arrival; }
    }
    float opacity(Time now) const {
        if(!confirmed)return 0;
        const double a=age(now);
        const double progress=std::clamp(seconds(now-revealNs)/.20,0.,1.);
        const double enter=revealFrom+(1.-revealFrom)*progress;
        // Brief occlusions coast, then fade; never retain a moving ghost forever.
        const double leave=a<=.55?1.:std::clamp((1.4-a)/.85,0.,1.);
        return float(std::min(enter,leave));
    }
    bool expired(Time now) const { return age(now)>(confirmed?1.4:.65); }
    void advance(Time now,double dt) {
        const double coast=std::min(age(now),.45);
        const double k=1-std::exp(-8*std::clamp(dt,0.,.12));
        drawX+=std::clamp((ax.p+ax.v*coast-drawX)*k,-4.*dt,4.*dt);
        drawZ+=std::clamp((az.p+az.v*coast-drawZ)*k,-12.*dt,12.*dt);
    }
};
// Rectangular global assignment, including an unmatched dummy for every row.
inline std::vector<int> assign(const std::vector<std::vector<double>> &cost) {
    if(cost.empty())return {};
    const int n=int(cost.size()),m=int(cost.front().size());
    std::vector<double> u(n+1),v(m+1);std::vector<int> p(m+1),way(m+1);
    for(int i=1;i<=n;++i) {
        p[0]=i;int j0=0;std::vector<double> best(m+1,1e12);std::vector<bool> used(m+1,false);
        do {
            used[j0]=true;const int i0=p[j0];double delta=1e12;int j1=0;
            for(int j=1;j<=m;++j)if(!used[j]) {
                const double cur=cost[i0-1][j-1]-u[i0]-v[j];
                if(cur<best[j]){best[j]=cur;way[j]=j0;}
                if(best[j]<delta){delta=best[j];j1=j;}
            }
            for(int j=0;j<=m;++j)if(used[j]){u[p[j]]+=delta;v[j]-=delta;}else best[j]-=delta;
            j0=j1;
        }while(p[j0]);
        do{const int j1=way[j0];p[j0]=p[j1];j0=j1;}while(j0);
    }
    std::vector<int> result(n,-1);for(int j=1;j<=m;++j)if(p[j])result[p[j]-1]=j-1;return result;
}
class Tracker {
    int nextId_=1;
    Time lastFrameNs_=0;
public:
    std::vector<Track> tracks;
    int births=0,reassociations=0,discardedTentative=0;
    void clear(){tracks.clear();lastFrameNs_=0;births=reassociations=discardedTentative=0;}
    static double cost(const Track &t,const Observation &o) {
        if(family(t.cls)!=family(o.cls))return 1e4;
        const double dt=seconds(o.measuredNs-t.lastNs);
        if(dt<=0 || dt>1.2)return 1e4;
        const double ratio=o.bw/std::max(.001,t.last.bw);
        if(ratio<.5 || ratio>2.)return 1e4;
        const double dx=(o.bx+o.bw/2)-(t.last.bx+t.last.bw/2+t.boxVx*std::min(dt,.45));
        const double dy=(o.by+o.bh/2)-(t.last.by+t.last.bh/2+t.boxVy*std::min(dt,.45));
        const double nx=dx/std::max(.025,(o.bw+t.last.bw)/2),ny=dy/std::max(.025,(o.bh+t.last.bh)/2);
        const double distance=std::hypot(nx,ny);
        if(distance>.65 || std::abs(o.x-(t.ax.p+t.ax.v*std::min(dt,.45)))>1.5 || std::abs(o.z-(t.az.p+t.az.v*std::min(dt,.45)))>8.)return 1e4;
        return .9*distance+.15*std::abs(std::log(ratio));
    }
    // Returns renderer IDs for measured observations; server IDs remain untouched.
    std::vector<int> update(const std::vector<Observation> &obs,Time frameNs,Time now) {
        std::vector<int> ids(obs.size(),0);if(frameNs<=lastFrameNs_)return ids;
        lastFrameNs_=frameNs;
        std::vector<bool> valid(obs.size(),false);
        for(size_t i=0;i<obs.size();++i) {
            const auto &o=obs[i];
            valid[i]=o.serverId>0 && o.measuredNs>0 && o.measuredNs<=now &&
                std::isfinite(o.confidence) && o.confidence>=.1 && o.confidence<=1. &&
                std::isfinite(o.bx) && std::isfinite(o.by) &&
                std::isfinite(o.bw) && std::isfinite(o.bh) && o.bw>0 && o.bh>0 &&
                std::isfinite(o.x) && std::isfinite(o.z);
        }
        prune(now);
        std::vector<bool> used(tracks.size(),false);
        // A known measured identity wins before geometric reacquisition.
        for(size_t i=0;i<obs.size();++i)for(size_t j=0;j<tracks.size();++j)
            if(valid[i] && !used[j] && obs[i].serverId==tracks[j].serverId && family(obs[i].cls)==family(tracks[j].cls)) {
                tracks[j].observe(obs[i],now);ids[i]=tracks[j].id;used[j]=true;break;
            }
        for(int tier=0;tier<2;++tier) {
            std::vector<size_t> rows,cols;
            for(size_t i=0;i<obs.size();++i)if(valid[i] && !ids[i] && (obs[i].confidence>=.35)==(tier==0))rows.push_back(i);
            for(size_t j=0;j<tracks.size();++j)if(!used[j])cols.push_back(j);
            std::vector<std::vector<double>> matrix(rows.size(),std::vector<double>(cols.size()+rows.size(),1.));
            for(size_t r=0;r<rows.size();++r)for(size_t c=0;c<cols.size();++c)matrix[r][c]=cost(tracks[cols[c]],obs[rows[r]]);
            const auto assignment=assign(matrix);
            for(size_t r=0;r<rows.size();++r) {
                const int c=assignment[r];if(c<0 || c>=int(cols.size()) || matrix[r][c]>=.8)continue;
                // Ambiguous nearby objects must not silently exchange identities.
                double alternative=1e4;for(size_t k=0;k<cols.size();++k)if(int(k)!=c)alternative=std::min(alternative,matrix[r][k]);
                if(alternative-matrix[r][c]<.12)continue;
                auto &t=tracks[cols[c]];t.observe(obs[rows[r]],now);ids[rows[r]]=t.id;used[cols[c]]=true;++reassociations;
            }
        }
        for(size_t i=0;i<obs.size();++i)if(valid[i] && !ids[i] && obs[i].serverId>0 && obs[i].confidence>=.35 && tracks.size()<64) {
            Track t;t.id=nextId_++;t.observe(obs[i],now);ids[i]=t.id;tracks.push_back(t);++births;
        }
        return ids;
    }
    void prune(Time now) {
        tracks.erase(std::remove_if(tracks.begin(),tracks.end(),[&](const Track &t){
            if(!t.expired(now))return false;
            if(!t.confirmed)++discardedTentative;
            return true;
        }),tracks.end());
    }
    void advance(Time now,double dt){prune(now);for(auto &t:tracks)t.advance(now,dt);}
};
}
