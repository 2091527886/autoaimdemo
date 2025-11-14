#ifndef TIMING_HPP
#define TIMING_HPP

#include <vector>
#include <tuple>
#include <limits>
#include <algorithm>
#include <string>
#include <cmath>
#include "Log.hpp"

struct Timing { double infer_ms=0, post_ms=0, loop_ms=0; std::string trt_throughput, trt_cpu_lat, trt_gpu_lat; };

struct FrameLat { unsigned long long idx=0; double cap_ms=0, det_ms=0, mou_ms=0, ctl_ms=0, ovl_ms=0, loop_ms=0; };

class LatAgg {
public:
    explicit LatAgg(size_t window):wnd(window?window:30){ buf.reserve(wnd);} 
    void push(const FrameLat& f){ buf.push_back(f); if(buf.size()>=wnd) printClear(); }
private:
    void printClear(){
        auto sc=[&](auto g){return stat(g);} ;
        auto [cap_avg,cap_min,cap_max]=sc([](const FrameLat& f){return f.cap_ms;});
        auto [det_avg,det_min,det_max]=sc([](const FrameLat& f){return f.det_ms;});
        auto [mou_avg,mou_min,mou_max]=sc([](const FrameLat& f){return f.mou_ms;});
        auto [ctl_avg,ctl_min,ctl_max]=sc([](const FrameLat& f){return f.ctl_ms;});
        auto [ovl_avg,ovl_min,ovl_max]=sc([](const FrameLat& f){return f.ovl_ms;});
        auto [loo_avg,loo_min,loo_max]=sc([](const FrameLat& f){return f.loop_ms;});
        lg::o()<<lg::fx2<<lg::ovr(P_L,buf.size())
            <<" | cap "<<cap_avg<<"/"<<cap_min<<"/"<<cap_max
            <<" | det "<<det_avg<<"/"<<det_min<<"/"<<det_max
            <<" | mou "<<mou_avg<<"/"<<mou_min<<"/"<<mou_max
            <<" | ctl "<<ctl_avg<<"/"<<ctl_min<<"/"<<ctl_max
            <<" | ovl "<<ovl_avg<<"/"<<ovl_min<<"/"<<ovl_max
            <<" | loop "<<loo_avg<<"/"<<loo_min<<"/"<<loo_max
            <<lg::endl<<lg::df; buf.clear(); }
    template<typename Getter>
    std::tuple<double,double,double> stat(Getter g) const{
        if(buf.empty()) return {0,0,0}; double sum=0,mn=std::numeric_limits<double>::infinity(),mx=0; 
        for(const auto& f:buf){ double v=g(f); sum+=v; mn=std::min(mn,v); mx=std::max(mx,v);} 
        return { sum/double(buf.size()), (std::isinf(mn)?0.0:mn), mx}; }
    size_t wnd; std::vector<FrameLat> buf; };

#endif
