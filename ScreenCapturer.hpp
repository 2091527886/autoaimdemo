#pragma once

#include <array>
#include <opencv2/opencv.hpp>
#include "Reporting.hpp"

class Config;

class ScreenCapturer {
public:
    // Reporting helper
    metrics::CaptureReporting reporter;
public:
    ScreenCapturer(const Config& cfg_ref, int crop_width, int crop_height);
    ~ScreenCapturer();

    ScreenCapturer(const ScreenCapturer&) = delete;
    ScreenCapturer& operator=(const ScreenCapturer&) = delete;

    bool capture(cv::Mat& frame, const cv::Rect& crop_region);

    int w() const { return width; }
    int h() const { return height; }

private:
    bool serveCache(cv::Mat& dst);
    void reportTimes(double t_acquire_ms, double t_qi_ms, double t_copy_ms,
                                    double t_map_ms, double t_cvt_ms, double t_unmap_ms, double t_total_ms);
    void dbgTick();

private:
    const Config& cfg;

    static constexpr size_t kStageN = 2;
    std::array<struct ID3D11Texture2D*, kStageN> stage{};
    std::array<struct ID3D11Query*,     kStageN> copyQ{};
    size_t idx = 0;
    cv::Mat cache;

    size_t statSkipDirty = 0;
    size_t statEvtWait = 0;
    size_t statStillDraw = 0;
    size_t statCacheRe = 0;
    size_t statOk = 0;

    struct IDXGIFactory1* fac = nullptr;
    struct IDXGIAdapter1* adp = nullptr;
    struct IDXGIOutput* out = nullptr;
    struct IDXGIOutput1* out1 = nullptr;
    struct ID3D11Device* dev = nullptr;
    struct ID3D11DeviceContext* ctx = nullptr;
    struct IDXGIOutputDuplication* dup = nullptr;
    int width = 0;
    int height = 0;
};
