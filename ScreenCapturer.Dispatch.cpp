#include "ScreenCapturer.hpp"
#include "Config.hpp"
#include "pch.hpp"
#include <chrono>

using clk = std::chrono::high_resolution_clock;
using tp = clk::time_point;
using msd = std::chrono::duration<double, std::milli>;
inline tp tn() { return clk::now(); }

bool ScreenCapturer::capture_view(ScreenCapturer::CapturedView &view, const cv::Rect &crop_region) {
  const bool do_tim = cfg.capture_timing && cfg.timing;

  view = CapturedView{};
  view.width = crop_region.width;
  view.height = crop_region.height;
  view.timing_enabled = do_tim && cfg.capture_timing_on_success;
  if (do_tim)
    view.t_total_start = tn();

  if (!useWgc) {
    return cap_dxgi_view(view, crop_region);
  } else {
    return cap_wgc_view(view, crop_region);
  }
}
