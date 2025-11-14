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

    // select by cfg.cap_gpu
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
    }

    LOG_IF(cfg.log_init_verbose, lg::o() << P_C "Ready " << (useWgc?"WGC ":"DDA ") << width << "x" << height << ", staging=" << kStageN << lg::endl);
    reporter.BootBanner(kStageN);
}

ScreenCapturer::~ScreenCapturer() {
    for (size_t i = 0; i < kStageN; ++i) {
        SafeRelease(&stage[i]);
    }
    if (useWgc) {
        rel_wgc();
    }
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
                << " ew=" << statCopyPend
                << " sd=" << statStillDraw
                << " ok=" << statOk << lg::endl;
        // 不清零，保持累计视角
    }
}

bool ScreenCapturer::capture_view(ScreenCapturer::CapturedView& view, const cv::Rect& crop_region) {
    const bool do_tim = cfg.capture_timing && cfg.timing;
    using clock = std::chrono::high_resolution_clock;

    // reset view
    view = CapturedView{};
    view.width = crop_region.width;
    view.height = crop_region.height;
    view.timing_enabled = do_tim && cfg.capture_timing_on_success;
    if (do_tim) view.t_total_start = clock::now();

    const int tries = std::max(1, 1 + cfg.cap_retry);

    if (!useWgc) {
        // DXGI path
        for (int attempt = 0; attempt < tries; ++attempt) {
            double aq = 0.0, qi = 0.0, cp = 0.0, mp = 0.0, up = 0.0; // cv(转换)为0

            IDXGIResource* pDesktopResource = nullptr;
            DXGI_OUTDUPL_FRAME_INFO frameInfo{};

            auto t_acq_s = do_tim ? clock::now() : clock::time_point{};
            HRESULT hr = dup->AcquireNextFrame(cfg.capture_timeout_ms_dxgi, &frameInfo, &pDesktopResource);
            auto t_acq_e = do_tim ? clock::now() : t_acq_s;
            if (do_tim) aq = std::chrono::duration<double, std::milli>(t_acq_e - t_acq_s).count();

            if (hr == DXGI_ERROR_WAIT_TIMEOUT) { view.aq_sum += aq; continue; }
            if (FAILED(hr)) {
                if (do_tim && !cfg.capture_timing_on_success) {
                    double tt = std::chrono::duration<double, std::milli>(clock::now() - view.t_total_start).count();
                    reportTimes(view.aq_sum + aq, view.qi_sum, view.cp_sum, view.mp_sum, view.cv_sum, view.up_sum, tt);
                }
                return false;
            }

            ID3D11Texture2D* pAcquiredDesktopImage = nullptr;
            auto t_qi_s = do_tim ? clock::now() : clock::time_point{};
            hr = pDesktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&pAcquiredDesktopImage);
            auto t_qi_e = do_tim ? clock::now() : t_qi_s;
            if (do_tim) qi = std::chrono::duration<double, std::milli>(t_qi_e - t_qi_s).count();
            SafeRelease(&pDesktopResource);
            if (FAILED(hr)) { view.aq_sum += aq; view.qi_sum += qi; dup->ReleaseFrame(); continue; }

            UINT dirty_size = 0;
            HRESULT hr_dirty = dup->GetFrameDirtyRects(0, nullptr, &dirty_size);
            if (SUCCEEDED(hr_dirty) && dirty_size == 0) {
                ++statSkipDirty; view.aq_sum += aq; view.qi_sum += qi; SafeRelease(&pAcquiredDesktopImage); dup->ReleaseFrame(); continue;
            }

            D3D11_BOX sourceRegion{};
            sourceRegion.left   = crop_region.x;
            sourceRegion.right  = crop_region.x + crop_region.width;
            sourceRegion.top    = crop_region.y;
            sourceRegion.bottom = crop_region.y + crop_region.height;
            sourceRegion.front  = 0; sourceRegion.back = 1;

            auto t_copy_s = do_tim ? clock::now() : clock::time_point{};
            ctx->CopySubresourceRegion(stage[0], 0, 0, 0, 0, pAcquiredDesktopImage, 0, &sourceRegion);
            auto t_copy_e = do_tim ? clock::now() : t_copy_s;
            if (do_tim) cp = std::chrono::duration<double, std::milli>(t_copy_e - t_copy_s).count();
            if (cfg.capture_flush_after_copy) ctx->Flush();

            SafeRelease(&pAcquiredDesktopImage);
            dup->ReleaseFrame();

            D3D11_MAPPED_SUBRESOURCE mr;
            auto t_map_s = do_tim ? clock::now() : clock::time_point{};
            hr = ctx->Map(stage[0], 0, D3D11_MAP_READ, 0, &mr);
            auto t_map_e = do_tim ? clock::now() : t_map_s;
            if (do_tim) mp = std::chrono::duration<double, std::milli>(t_map_e - t_map_s).count();
            if (FAILED(hr)) { view.aq_sum += aq; view.qi_sum += qi; view.cp_sum += cp; view.mp_sum += mp; continue; }

            // fill view and keep mapped
            view.data = static_cast<unsigned char*>(mr.pData);
            view.pitch = mr.RowPitch;
            view.mapped = true;

            // accumulate and success (no unmap here)
            view.aq_sum += aq; view.qi_sum += qi; view.cp_sum += cp; view.mp_sum += mp;
            ++statOk; dbgTick();
            return true;
        }
        if (cfg.capture_timing && !cfg.capture_timing_on_success) {
            double tt = do_tim ? std::chrono::duration<double, std::milli>(clock::now() - view.t_total_start).count() : 0.0;
            reportTimes(view.aq_sum, view.qi_sum, view.cp_sum, view.mp_sum, view.cv_sum, view.up_sum, tt);
        }
        return false;
    } else {
        // WGC path delegated
        return cap_wgc_view(view, crop_region);
    }
}

void ScreenCapturer::release_view(ScreenCapturer::CapturedView& view) {
    const bool do_tim = cfg.capture_timing && cfg.timing;
    using clock = std::chrono::high_resolution_clock;

    if (!view.mapped) return;
    auto t_unmap_s = do_tim ? clock::now() : clock::time_point{};
    ctx->Unmap(stage[0], 0);
    auto t_unmap_e = do_tim ? clock::now() : t_unmap_s;
    double up = do_tim ? std::chrono::duration<double, std::milli>(t_unmap_e - t_unmap_s).count() : 0.0;
    view.up_sum += up;

    if (cfg.capture_timing && cfg.capture_timing_on_success) {
        double tt = do_tim ? std::chrono::duration<double, std::milli>(t_unmap_e - view.t_total_start).count() : 0.0;
        reportTimes(view.aq_sum, view.qi_sum, view.cp_sum, view.mp_sum, view.cv_sum, view.up_sum, tt);
    }

    view.mapped = false;
    view.data = nullptr;
}

void ScreenCapturer::reportTimes(double t_acquire_ms, double t_qi_ms, double t_copy_ms,
                                    double t_map_ms, double t_cvt_ms, double t_unmap_ms, double t_total_ms) {
    reporter.ReportStepTimings(
        t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms,
        statSkipDirty, statCopyPend, statStillDraw, statOk
    );
}
