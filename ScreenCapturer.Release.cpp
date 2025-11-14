#include "ScreenCapturer.hpp"
#include "ComHelpers.hpp"
#include "Config.hpp"
#include "Log.hpp"
#include "pch.hpp"
#include "sh.hpp"

ScreenCapturer::~ScreenCapturer(){
  for(size_t i=0;i<kStageN;++i){ SafeRelease(&stage[i]); }
  if(useWgc){ rel_wgc(); }
  SafeRelease(&dup);
  SafeRelease(&out1);
  SafeRelease(&out);
  SafeRelease(&adp);
  SafeRelease(&fac);
  SafeRelease(&ctx);
  SafeRelease(&dev);
}

void ScreenCapturer::dbgTick(){
  static size_t n=0;
  if(!cfg.capture_debug) return;
  n++;
  int m=std::max(1, cfg.capture_debug_every_n);
  if(n % (size_t)m == 0){
    lg::o() << P_C "dbg ds=" << statSkipDirty
            << " ew=" << statCopyPend
            << " sd=" << statStillDraw
            << " ok=" << statOk << lg::endl;
  }
}

void ScreenCapturer::release_view(ScreenCapturer::CapturedView& view){
  const bool do_tim = cfg.capture_timing && cfg.timing;
  if(!view.mapped) return;
  TSTART(t_unmap_s, do_tim);
  ctx->Unmap(stage[0], 0);
  double up=0.0; TELAPSE(up, do_tim, t_unmap_s);
  view.up_sum += up;
  if(cfg.capture_timing && cfg.capture_timing_on_success){
    double tt = do_tim ? msd(tn() - view.t_total_start).count() : 0.0;
    reportTimes(view.aq_sum, view.qi_sum, view.cp_sum, view.mp_sum, view.cv_sum, view.up_sum, tt);
  }
  view.mapped=false; view.data=nullptr;
}

void ScreenCapturer::reportTimes(double t_acquire_ms, double t_qi_ms, double t_copy_ms,
                                 double t_map_ms, double t_cvt_ms, double t_unmap_ms, double t_total_ms){
  reporter.ReportStepTimings(t_acquire_ms,t_qi_ms,t_copy_ms,t_map_ms,t_cvt_ms,t_unmap_ms,t_total_ms,
                             statSkipDirty, statCopyPend, statStillDraw, statOk);
}
