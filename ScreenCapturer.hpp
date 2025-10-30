#pragma once

#include <array>
#include <opencv2/opencv.hpp>
#include "Reporting.hpp"

class Config;

class ScreenCapturer {
public:
    // Reporting helper
    metrics::CaptureReporting capture_reporter_;
public:
    ScreenCapturer(const Config& cfg_ref, int crop_width, int crop_height);
    ~ScreenCapturer();

    ScreenCapturer(const ScreenCapturer&) = delete;
    ScreenCapturer& operator=(const ScreenCapturer&) = delete;

    bool CaptureFrame(cv::Mat& frame, const cv::Rect& crop_region);

    int getWidth() const { return width; }
    int getHeight() const { return height; }

private:
    bool try_serve_cache(cv::Mat& out);
    void accumulate_and_maybe_print(double t_acquire_ms, double t_qi_ms, double t_copy_ms,
                                    double t_map_ms, double t_cvt_ms, double t_unmap_ms, double t_total_ms);

private:
    const Config& cfg;

    static constexpr size_t kStagingCount = 2;
    std::array<struct ID3D11Texture2D*, kStagingCount> staging_{};
    std::array<struct ID3D11Query*,     kStagingCount> copy_done_queries_{};
    size_t frame_index_ = 0;
    cv::Mat last_bgr_cache_;

    size_t stat_dirty_skips_ = 0;
    size_t stat_event_not_ready_ = 0;
    size_t stat_map_still_drawing_ = 0;
    size_t stat_cache_reused_ = 0;
    size_t stat_success_ = 0;

    struct IDXGIFactory1* pFactory = nullptr;
    struct IDXGIAdapter1* pAdapter = nullptr;
    struct IDXGIOutput* pOutput = nullptr;
    struct IDXGIOutput1* pOutput1 = nullptr;
    struct ID3D11Device* pDevice = nullptr;
    struct ID3D11DeviceContext* pContext = nullptr;
    struct IDXGIOutputDuplication* pDuplicator = nullptr;
    int width = 0;
    int height = 0;
};
