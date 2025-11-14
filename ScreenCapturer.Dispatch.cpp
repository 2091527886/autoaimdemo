#include "ScreenCapturer.hpp"
#include "Config.hpp"
#include "Reporting.hpp"

#include <chrono>

bool ScreenCapturer::capture_view(ScreenCapturer::CapturedView& view, const cv::Rect& crop_region) {
    const bool do_tim = cfg.capture_timing && cfg.timing;
    using clock = std::chrono::high_resolution_clock;

    // reset view
    view = CapturedView{};
    view.width = crop_region.width;
    view.height = crop_region.height;
    view.timing_enabled = do_tim && cfg.capture_timing_on_success;
    if (do_tim) view.t_total_start = clock::now();

    if (!useWgc) {
        // DXGI path delegated
        return cap_dxgi_view(view, crop_region);
    } else {
        // WGC path delegated
        return cap_wgc_view(view, crop_region);
    }
}
