#include "ScreenCapturer.hpp"
#include "Config.hpp"
#include "Log.hpp"
#include "pch.hpp"
#include "ComHelpers.hpp"
#include "sh.hpp"
#include "sc_helpers.hpp"
using std::max;

bool ScreenCapturer::cap_dxgi_view(ScreenCapturer::CapturedView &view, const cv::Rect &crop_region) {
  const bool do_tim = cfg.capture_timing && cfg.timing;

  const int tries = max(1, 1 + cfg.cap_retry);

  auto report_fail = [&](double aq_add){ if (do_tim && !cfg.capture_timing_on_success){ double tt = msd(tn() - view.t_total_start).count(); rpt_add(view, tt, aq_add);} };
  bool success=false, fatal=false;
  auto body = [&]() -> bool {
    double aq=0.0, qi=0.0, cp=0.0, mp=0.0;

    IDXGIResource* pDesktopResource=nullptr;
    DXGI_OUTDUPL_FRAME_INFO frameInfo{};

    TSTART(t_acq_s, do_tim);
    HRESULT hr = dup->AcquireNextFrame(cfg.capture_timeout_ms_dxgi, &frameInfo, &pDesktopResource);
    TELAPSE(aq, do_tim, t_acq_s);

    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
      view.aq_sum += aq;
      return false;
    }
    if (FAILED(hr)) {
      report_fail(aq);
      fatal = true;
      return true;
    }

    ID3D11Texture2D* pAcquiredDesktopImage=nullptr;
    TSTART(t_qi_s, do_tim);
    hr = pDesktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void **)&pAcquiredDesktopImage);
    TELAPSE(qi, do_tim, t_qi_s);
    SafeRelease(&pDesktopResource);
    if (FAILED(hr)) {
      view.aq_sum += aq;
      view.qi_sum += qi;
      dup->ReleaseFrame();
      return false;
    }

    UINT dirty_size = 0;
    HRESULT hr_dirty = dup->GetFrameDirtyRects(0, nullptr, &dirty_size);
    if (SUCCEEDED(hr_dirty) && dirty_size == 0) {
      ++statSkipDirty;
      view.aq_sum += aq;
      view.qi_sum += qi;
      SafeRelease(&pAcquiredDesktopImage);
      dup->ReleaseFrame();
      return false;
    }

    D3D11_BOX sourceRegion = box_from_rect(crop_region);

    bool ok_copy = cpmp(ctx, stage[0], pAcquiredDesktopImage, &sourceRegion, view, do_tim, cp, mp, cfg.capture_flush_after_copy);

    SafeRelease(&pAcquiredDesktopImage);
    dup->ReleaseFrame();

    if (!ok_copy) {
      view.aq_sum += aq;
      view.qi_sum += qi;
      view.cp_sum += cp;
      view.mp_sum += mp;
      return false;
    }

    view.aq_sum += aq;
    view.qi_sum += qi;
    view.cp_sum += cp;
    view.mp_sum += mp;
    ++statOk;
    dbgTick();
    success = true;
    return true;
  };
  if (try_n(tries, body)) return true;
  if (fatal) return false;
  if (success) return true;
  if (cfg.capture_timing && !cfg.capture_timing_on_success){ double tt = do_tim ? msd(tn() - view.t_total_start).count() : 0.0; rpt(view, tt); }
  return false;
}
