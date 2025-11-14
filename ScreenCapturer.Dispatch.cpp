#include "ScreenCapturer.hpp"
#include "pch.hpp"
#include "sh.hpp"

bool ScreenCapturer::capture_view(ScreenCapturer::CapturedView &view, const cv::Rect &crop_region) {
  const bool do_tim = cfg.capture_timing && cfg.timing;

  view = CapturedView{};
  view.width = crop_region.width;
  view.height = crop_region.height;
  if (do_tim)
    view.t_total_start = tn();

  if (!useWgc) {
    return cap_dxgi_view(view, crop_region);
  } else {
    return cap_wgc_view(view, crop_region);
  }
}
