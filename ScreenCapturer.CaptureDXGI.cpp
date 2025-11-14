#include "ScreenCapturer.hpp"
#include "Config.hpp"
#include "Log.hpp"
#include "pch.hpp"
#include "ComHelpers.hpp"
#include <chrono>

using clk = std::chrono::high_resolution_clock;
using tp = clk::time_point;
using msd = std::chrono::duration<double, std::milli>;
inline tp tn() { return clk::now(); }

bool ScreenCapturer::cap_dxgi_view(ScreenCapturer::CapturedView &view, const cv::Rect &crop_region) {
  const bool do_tim = cfg.capture_timing && cfg.timing;

  const int tries = std::max(1, 1 + cfg.cap_retry);

  auto try_loop = [&](auto &&body) -> bool {
    for (int attempt = 0; attempt < tries; ++attempt) {
      if (body())
        return true;
    }
    return false;
  };
  auto report_fail = [&](double aq_add) {
    if (do_tim && !cfg.capture_timing_on_success) {
      double tt = msd(tn() - view.t_total_start).count();
      reportTimes(view.aq_sum + aq_add, view.qi_sum, view.cp_sum, view.mp_sum, view.cv_sum, view.up_sum, tt);
    }
  };
  bool success = false, fatal = false;
  auto body = [&]() -> bool {
    double aq = 0.0, qi = 0.0, cp = 0.0, mp = 0.0; // cv(转换)为0

    IDXGIResource *pDesktopResource = nullptr;
    DXGI_OUTDUPL_FRAME_INFO frameInfo{};

    auto t_acq_s = do_tim ? tn() : tp{};
    HRESULT hr = dup->AcquireNextFrame(cfg.capture_timeout_ms_dxgi, &frameInfo, &pDesktopResource);
    auto t_acq_e = do_tim ? tn() : t_acq_s;
    if (do_tim)
      aq = msd(t_acq_e - t_acq_s).count();

    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
      view.aq_sum += aq;
      return false;
    }
    if (FAILED(hr)) {
      report_fail(aq);
      fatal = true;
      return true; // fail outer
    }

    ID3D11Texture2D *pAcquiredDesktopImage = nullptr;
    auto t_qi_s = do_tim ? tn() : tp{};
    hr = pDesktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void **)&pAcquiredDesktopImage);
    auto t_qi_e = do_tim ? tn() : t_qi_s;
    if (do_tim)
      qi = msd(t_qi_e - t_qi_s).count();
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

    D3D11_BOX sourceRegion{};
    sourceRegion.left = crop_region.x;
    sourceRegion.right = crop_region.x + crop_region.width;
    sourceRegion.top = crop_region.y;
    sourceRegion.bottom = crop_region.y + crop_region.height;
    sourceRegion.front = 0;
    sourceRegion.back = 1;

    auto copy_and_map = [&](ID3D11Texture2D *srcTex) -> bool {
      auto t_copy_s = do_tim ? tn() : tp{};
      ctx->CopySubresourceRegion(stage[0], 0, 0, 0, 0, srcTex, 0, &sourceRegion);
      auto t_copy_e = do_tim ? tn() : t_copy_s;
      if (do_tim)
        cp = msd(t_copy_e - t_copy_s).count();
      if (cfg.capture_flush_after_copy)
        ctx->Flush();

      D3D11_MAPPED_SUBRESOURCE mr;
      auto t_map_s = do_tim ? tn() : tp{};
      HRESULT hrm = ctx->Map(stage[0], 0, D3D11_MAP_READ, 0, &mr);
      auto t_map_e = do_tim ? tn() : t_map_s;
      if (do_tim)
        mp = msd(t_map_e - t_map_s).count();
      if (FAILED(hrm))
        return false;

      view.data = static_cast<unsigned char *>(mr.pData);
      view.pitch = mr.RowPitch;
      view.mapped = true;
      return true;
    };

    bool ok_copy = copy_and_map(pAcquiredDesktopImage);

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
    return true; // exit loop
  };
  if (try_loop(body)) {
    return true;
  }
  if (fatal) {
    return false;
  }
  if (success) {
    return true;
  }
  if (cfg.capture_timing && !cfg.capture_timing_on_success) {
    double tt = do_tim ? msd(tn() - view.t_total_start).count() : 0.0;
    reportTimes(view.aq_sum, view.qi_sum, view.cp_sum, view.mp_sum, view.cv_sum, view.up_sum, tt);
  }
  return false;
}
