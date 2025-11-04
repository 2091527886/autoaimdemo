#include "ScreenCapturer.hpp"
#include "Config.hpp"
#include "SystemTuner.hpp"
#include "ComHelpers.hpp"
#include "Reporting.hpp"
#include "Log.hpp"

#include <array>
#include <chrono>
#include <stdexcept>
#include <algorithm>

// Windows & DirectX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

ScreenCapturer::ScreenCapturer(const Config& cfg_ref, int crop_width, int crop_height)
    : cfg(cfg_ref), reporter(cfg_ref) {
    HRESULT hr;

    // DXGI factory
    hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&fac);
    if (FAILED(hr)) throw std::runtime_error("Failed to create DXGI Factory.");

    // Adapter select by cfg.cap_gpu (>=0)
    UINT aidx = (cfg.cap_gpu >= 0) ? static_cast<UINT>(cfg.cap_gpu) : 0u;
    if (FAILED(fac->EnumAdapters1(aidx, &adp))) throw std::runtime_error("Failed to enumerate adapters.");

    // Optional enum logging
    if (cfg.cap_enum) {
        lg::o() << P_C "Adapters:" << lg::endl;
        for (UINT i=0;;++i){
            IDXGIAdapter1* a=nullptr; if (fac->EnumAdapters1(i,&a)!=S_OK) break; DXGI_ADAPTER_DESC1 d; a->GetDesc1(&d);
            lg::o() << P_C "["<<i<<"]" << lg::endl; a->Release();
        }
        lg::o() << P_C "Outputs on adapter "<<aidx<<":" << lg::endl;
        for (UINT i=0;;++i){ IDXGIOutput* o=nullptr; if (adp->EnumOutputs(i,&o)!=S_OK) break; DXGI_OUTPUT_DESC od; o->GetDesc(&od);
            RECT r=od.DesktopCoordinates; lg::o() << P_C "["<<i<<"] " << r.right-r.left << "x" << r.bottom-r.top << lg::endl; o->Release(); }
    }

    // Output selection by cfg.cap_out
    if (FAILED(adp->EnumOutputs(cfg.cap_out, &out))) throw std::runtime_error("Failed to enumerate outputs.");

    DXGI_OUTPUT_DESC outputDesc; out->GetDesc(&outputDesc);
    width = outputDesc.DesktopCoordinates.right - outputDesc.DesktopCoordinates.left;
    height = outputDesc.DesktopCoordinates.bottom - outputDesc.DesktopCoordinates.top;

    // D3D11 device
    if (FAILED(D3D11CreateDevice(adp, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev, nullptr, &ctx)))
        throw std::runtime_error("Failed to create D3D11 device.");
    TuneDxgi(dev, cfg.dxgi_gpu_prio, cfg.dxgi_max_latency);

    // Backend init
    useWgc = (cfg.cap == 1);
    if (!useWgc) {
        init_dxgi();
        reporter.SetBackend(metrics::CaptureReporting::Backend::DXGI);
    } else {
        init_wgc();
        reporter.SetBackend(metrics::CaptureReporting::Backend::WGC);
    }

    // staging & queries
    D3D11_TEXTURE2D_DESC desc; ZeroMemory(&desc, sizeof(desc));
    desc.Width = crop_width;
    desc.Height = crop_height;
    desc.MipLevels = 1; desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1; desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_STAGING; desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ; desc.MiscFlags = 0;

    for (size_t i = 0; i < kStageN; ++i) {
        HRESULT hr_local = dev->CreateTexture2D(&desc, NULL, &stage[i]);
        if (FAILED(hr_local)) throw std::runtime_error("Failed to create staging texture[" + std::to_string(i) + "]");
        D3D11_QUERY_DESC qd{}; qd.Query = D3D11_QUERY_EVENT; qd.MiscFlags = 0;
        hr_local = dev->CreateQuery(&qd, &copyQ[i]);
        if (FAILED(hr_local)) throw std::runtime_error("Failed to create EVENT query for staging[" + std::to_string(i) + "]");
    }

    LOG_IF(cfg.log_init_verbose, lg::o() << P_C "Ready " << (useWgc?"WGC ":"DDA ") << width << "x" << height << ", staging=" << kStageN << lg::endl);
    reporter.BootBanner(kStageN);
}

ScreenCapturer::~ScreenCapturer() {
    for (size_t i = 0; i < kStageN; ++i) {
        SafeRelease(&stage[i]);
        SafeRelease(&copyQ[i]);
    }
    if (useWgc) {
        rel_wgc();
    }
    if (fev) { CloseHandle(fev); fev = nullptr; }
    SafeRelease(&dup);
    SafeRelease(&out1);
    SafeRelease(&out);
    SafeRelease(&adp);
    SafeRelease(&fac);
    SafeRelease(&ctx);
    SafeRelease(&dev);
}

void ScreenCapturer::dbgTick() {
    static size_t n = 0;
    if (!cfg.capture_debug) return;
    n++;
    int m = std::max(1, cfg.capture_debug_every_n);
    if (n % static_cast<size_t>(m) == 0) {
        lg::o() << P_C "dbg ds=" << statSkipDirty
                << " ew=" << statEvtWait
                << " sd=" << statStillDraw
                << " cr=" << statCacheRe
                << " ok=" << statOk << lg::endl;
        // 不清零，保持累计视角
    }
}

bool ScreenCapturer::capture(cv::Mat& frame, const cv::Rect& crop_region) {
    if (useWgc) {
        return cap_wgc(frame, crop_region);
    }
    return cap_dxgi(frame, crop_region);
}

bool ScreenCapturer::serveCache(cv::Mat& dst) {
    if (!cache.empty()) {
        if (cache.rows == dst.rows && cache.cols == dst.cols && cache.type() == dst.type()) {
            cache.copyTo(dst);
            return true;
        } else {
            dst = cache.clone();
            return true;
        }
    }
    return false;
}

void ScreenCapturer::reportTimes(double t_acquire_ms, double t_qi_ms, double t_copy_ms,
                                    double t_map_ms, double t_cvt_ms, double t_unmap_ms, double t_total_ms) {
    reporter.ReportStepTimings(
        t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms,
        statSkipDirty, statEvtWait, statStillDraw, statCacheRe, statOk
    );
}
