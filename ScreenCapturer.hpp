#pragma once

#include <array>
#include <opencv2/core.hpp>
#include "Reporting.hpp"
#include <windows.h>
#include <chrono>

class Config;

class ScreenCapturer {
public:
    struct CapturedView {
        // mapped BGRA view into staging texture
        unsigned char* data = nullptr;
        int width = 0;
        int height = 0;
        size_t pitch = 0;
        // timing accumulators (reported on release)
        bool timing_enabled = false;
        double aq_sum = 0.0, qi_sum = 0.0, cp_sum = 0.0, mp_sum = 0.0, cv_sum = 0.0, up_sum = 0.0;
        std::chrono::high_resolution_clock::time_point t_total_start{};
        // ownership
        bool mapped = false;
    };

public:
    // Reporting helper
    metrics::CaptureReporting reporter;
    ScreenCapturer(const Config& cfg_ref, int crop_width, int crop_height);
    ~ScreenCapturer();

    ScreenCapturer(const ScreenCapturer&) = delete;
    ScreenCapturer& operator=(const ScreenCapturer&) = delete;

    // 零拷贝视图捕获（需调用 release_view 归还）
    bool capture_view(CapturedView& view, const cv::Rect& crop_region);
    void release_view(CapturedView& view);

    int w() const { return width; }
    int h() const { return height; }

private:
    void reportTimes(double t_acquire_ms, double t_qi_ms, double t_copy_ms,
                                    double t_map_ms, double t_cvt_ms, double t_unmap_ms, double t_total_ms);
    void dbgTick();

    // DXGI/WGC 后端拆分
    bool init_dxgi();
    bool init_wgc();
    bool cap_wgc_view(CapturedView& view, const cv::Rect& crop_region);
    void rel_wgc();

private:
    const Config& cfg;

    static constexpr size_t kStageN = 1;
    std::array<struct ID3D11Texture2D*, kStageN> stage{};
        size_t statSkipDirty = 0;
    size_t statCopyPend = 0;    // kept for reporting compatibility (always 0)
    size_t statStillDraw = 0;   // kept for reporting compatibility (always 0)
    size_t statOk = 0;

    // D3D/DXGI common
    struct IDXGIFactory1* fac = nullptr;
    struct IDXGIAdapter1* adp = nullptr;
    struct IDXGIOutput* out = nullptr;
    struct IDXGIOutput1* out1 = nullptr;
    struct ID3D11Device* dev = nullptr;
    struct ID3D11DeviceContext* ctx = nullptr;

    // DDA only
    struct IDXGIOutputDuplication* dup = nullptr;

    // WGC impl hidden (declared in .cpp)
    struct WgcCtx;
    WgcCtx* wgc = nullptr;
    bool useWgc = false;

    int width = 0;
    int height = 0;

};

