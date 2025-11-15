#pragma once
#include "Log.hpp"
#include "Config.hpp"
#include "sh.hpp"
#include <string>
#include <tuple>
#include <algorithm>
#ifdef _WIN32
#include "pch.hpp"
#endif

namespace metrics {

class CaptureReporting {
 public:
  enum class Backend { DXGI, WGC };
 public:
  explicit CaptureReporting(const Config& cfg):cfg_(cfg){}
  void SetBackend(Backend b){ backend_=b; }
  void BootBanner(size_t stagingCount) const {
    if(!(cfg_.opt_report&&cfg_.opt_report_capture)) return;
    lg::o()<<P_B "Capture: "<<"ds=ON"<<" | stg="<<stagingCount<<" | evt=OFF"<<" | map=DNW"<<lg::endl;
    lg::o()<<P_B "Timing: "<<"step="<<(cfg_.capture_timing?"ON":"OFF")<<" | int="<<cfg_.capture_timing_every_n<<" | succ-only="<<(cfg_.capture_timing_on_success?"ON":"OFF")<<lg::endl;
  }
  void ReportStepTimings(double t_acquire_ms,double t_qi_ms,double t_copy_ms,double t_map_ms,double t_cvt_ms,double t_unmap_ms,double t_total_ms,size_t& stat_dirty_skips_member,size_t& stat_event_not_ready_member,size_t& stat_map_still_drawing_member,size_t& stat_success_member){
    stat_dirty_skips_+=stat_dirty_skips_member; stat_event_not_ready_+=stat_event_not_ready_member; stat_map_still_drawing_+=stat_map_still_drawing_member; stat_success_+=stat_success_member; stat_dirty_skips_member=stat_event_not_ready_member=stat_map_still_drawing_member=stat_success_member=0; opt_counter_++; counter_++; acc_acq_+=t_acquire_ms; acc_qi_+=t_qi_ms; acc_copy_+=t_copy_ms; acc_map_+=t_map_ms; acc_cvt_+=t_cvt_ms; acc_unmap_+=t_unmap_ms; acc_total_+=t_total_ms; }
  void drain(){
    bool print=cfg_.capture_timing_each_frame; if(!print){ int it=MX(1,cfg_.capture_timing_every_n); if(it<=1) print=(counter_>0); else if(counter_%(size_t)it==0) print=true; }
    if(!print||counter_==0) return; double n=(double)counter_;
    const char* l_acq=(backend_==Backend::WGC?"gf":"acq"),*l_qi=(backend_==Backend::WGC?"wa":"qi"),*l_cpy="cp",*l_map="mp",*l_cvt="cv",*l_unm="up",*l_tot="tt";
    lg::o()<<lg::fx3<<lg::ovr(P_CP,counter_)<<lg::fld(l_acq,(acc_acq_/n),"ms")<<lg::fld(l_qi,(acc_qi_/n),"ms")<<lg::fld(l_cpy,(acc_copy_/n),"ms")<<lg::fld(l_map,(acc_map_/n),"ms")<<lg::fld(l_cvt,(acc_cvt_/n),"ms")<<lg::fld(l_unm,(acc_unmap_/n),"ms")<<lg::fld(l_tot,(acc_total_/n),"ms")<<lg::endl<<lg::df;
    if(cfg_.opt_report&&cfg_.opt_report_capture){ bool p=false; int oi=MX(1,cfg_.opt_report_every_n); if(oi<=1)p=(opt_counter_>0); else if(opt_counter_%(size_t)oi==0)p=true; if(p){ size_t tot=(size_t)opt_counter_; auto pct=[&](size_t v){return tot? (100.0*double(v)/double(tot)):0.0;}; lg::o()<<lg::fx1<<lg::ovr(P_CO,tot)<<" | ds="<<stat_dirty_skips_<<" ("<<pct(stat_dirty_skips_)<<"%)"<<" | ev="<<stat_event_not_ready_<<" ("<<pct(stat_event_not_ready_)<<"%)"<<" | sd="<<stat_map_still_drawing_<<" ("<<pct(stat_map_still_drawing_)<<"%)"<<" | ok="<<stat_success_<<" ("<<pct(stat_success_)<<"%)"<<lg::endl<<lg::df; opt_counter_=0; stat_dirty_skips_=stat_event_not_ready_=stat_map_still_drawing_=stat_success_=0; } }
    counter_=0; acc_acq_=acc_qi_=acc_copy_=acc_map_=acc_cvt_=acc_unmap_=acc_total_=0.0; }
 private:
  const Config& cfg_; size_t opt_counter_=0, stat_dirty_skips_=0, stat_event_not_ready_=0, stat_map_still_drawing_=0, stat_success_=0; size_t counter_=0; double acc_acq_=0, acc_qi_=0, acc_copy_=0, acc_map_=0, acc_cvt_=0, acc_unmap_=0, acc_total_=0; Backend backend_=Backend::DXGI; };

struct SystemBootReporting {
  static void PrintOverlayBanner(const Config& cfg,bool started){ if(!cfg.opt_report||!cfg.opt_report_system) return; if(started){ lg::o()<<P_B "Overlay=ON (a="<<cfg.ovl_a<<", rgb="<<cfg.ovl_r<<","<<cfg.ovl_g<<","<<cfg.ovl_b<<", th="<<cfg.ovl_thick<<")"<<lg::endl; } else { lg::o()<<P_B "Overlay=OFF"<<lg::endl; } }
#ifdef _WIN32
  static void PrintSchedulingBanner(const Config& cfg,HANDLE /*hMmcss*/){ if(!cfg.opt_report||!cfg.opt_report_system) return; lg::wo()<<WP_B L"Scheduling: MMCSS=OFF"<<lg::wendl; }
#endif
};

class TrtPerfReporting { public: explicit TrtPerfReporting(const Config& cfg):cfg_(cfg){} bool ShouldReport(){ if(cfg_.perf_report){ if(cfg_.perf_each_frame) return true; perf_frame_counter_++; if(cfg_.perf_report_every_n<=1) return true; if(perf_frame_counter_%(size_t)cfg_.perf_report_every_n==0) return true; } return false; } private: const Config& cfg_; size_t perf_frame_counter_=0; };

} // namespace metrics
