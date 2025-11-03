#include "ScreenCapturer.hpp"
#include "Config.hpp"
#include "SystemTuner.hpp"
#include "ComHelpers.hpp"
#include "Reporting.hpp"

#include <array>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include "Log.hpp"

// Windows & DirectX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

ScreenCapturer::ScreenCapturer(const Config& cfg_ref, int crop_width, int crop_height)
    : cfg(cfg_ref), reporter(cfg_ref) {
    HRESULT hr;

    hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&fac);
    if (FAILED(hr)) throw std::runtime_error("Failed to create DXGI Factory.");
    {
        HRESULT hr_enum_adapters = fac->EnumAdapters1(0, &adp);
        if (FAILED(hr_enum_adapters)) throw std::runtime_error("Failed to enumerate adapters.");
    }
    {
        HRESULT hr_enum_outputs = adp->EnumOutputs(0, &out);
        if (FAILED(hr_enum_outputs)) throw std::runtime_error("Failed to enumerate outputs.");
    }

    DXGI_OUTPUT_DESC outputDesc;
    out->GetDesc(&outputDesc);
    width = outputDesc.DesktopCoordinates.right - outputDesc.DesktopCoordinates.left;
    height = outputDesc.DesktopCoordinates.bottom - outputDesc.DesktopCoordinates.top;

    {
        HRESULT hr_create_device = D3D11CreateDevice(adp, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev, nullptr, &ctx);
        if (FAILED(hr_create_device)) {
            throw std::runtime_error("Failed to create D3D11 device.");
        }
        // Raise DXGI GPU priority and reduce frame latency for this device
        TuneDxgi(dev, cfg.dxgi_gpu_prio, cfg.dxgi_max_latency);
    }
    {
        HRESULT hr_qi_output1 = out->QueryInterface(__uuidof(IDXGIOutput1), (void**)&out1);
        if (FAILED(hr_qi_output1)) {
            throw std::runtime_error("Failed to query IDXGIOutput1.");
        }
    }
    hr = out1->DuplicateOutput(dev, &dup);
    if (FAILED(hr)) {
        lg::ok_or_hrfail_if(true, P_C, "DuplicateOutput", hr, SUCCEEDED(hr));
        throw std::runtime_error("Failed to create output duplication.");
    }

    // 创建多缓冲 staging 纹理与事件查询
    D3D11_TEXTURE2D_DESC desc;
    desc.Width = crop_width;
    desc.Height = crop_height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;

    HRESULT hr_local;
    for (size_t i = 0; i < kStageN; ++i) {
        hr_local = dev->CreateTexture2D(&desc, NULL, &stage[i]);
        if (FAILED(hr_local)) {
            throw std::runtime_error("Failed to create staging texture[" + std::to_string(i) + "]");
        }
        D3D11_QUERY_DESC qd{}; qd.Query = D3D11_QUERY_EVENT; qd.MiscFlags = 0;
        hr_local = dev->CreateQuery(&qd, &copyQ[i]);
        if (FAILED(hr_local)) {
            throw std::runtime_error("Failed to create EVENT query for staging[" + std::to_string(i) + "]");
        }
    }
    LOG_IF(cfg.log_init_verbose, lg::o() << P_C "Ready " << width << "x" << height << ", staging=" << kStageN << lg::endl);

    // 捕获链路启动横幅（仅在启用报告时）
    reporter.BootBanner(kStageN);
}

ScreenCapturer::~ScreenCapturer() {
    for (size_t i = 0; i < kStageN; ++i) {
        SafeRelease(&stage[i]);
        SafeRelease(&copyQ[i]);
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
                << " ew=" << statEvtWait
                << " sd=" << statStillDraw
                << " cr=" << statCacheRe
                << " ok=" << statOk << lg::endl;
        // 不清零，保持累计视角
    }
}

bool ScreenCapturer::capture(cv::Mat& frame, const cv::Rect& crop_region) {
    auto t_total_start = std::chrono::high_resolution_clock::now();
    double t_acquire_ms = 0.0, t_qi_ms = 0.0, t_copy_ms = 0.0, t_map_ms = 0.0, t_cvt_ms = 0.0, t_unmap_ms = 0.0, t_total_ms = 0.0;

    // Minimal helper: compute and report on failure paths (logic unchanged)
    auto report_if_fail = [&]() {
        if (cfg.capture_timing && !cfg.capture_timing_on_success) {
            t_total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now() - t_total_start).count();
            reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
        }
    };

    IDXGIResource* pDesktopResource = nullptr;
    DXGI_OUTDUPL_FRAME_INFO frameInfo;

    auto t_acq_s = std::chrono::high_resolution_clock::now();
    HRESULT hr = dup->AcquireNextFrame(cfg.capture_timeout_ms, &frameInfo, &pDesktopResource);
    auto t_acq_e = std::chrono::high_resolution_clock::now();
    t_acquire_ms = std::chrono::duration<double, std::milli>(t_acq_e - t_acq_s).count();

    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
        if (serveCache(frame)) {
            ++statCacheRe;
            return true;
        }
        report_if_fail();
        return false;
    }
    if (FAILED(hr)) {
        lg::ok_or_hrfail_if(true, P_C, "AcquireNextFrame", hr, SUCCEEDED(hr));
        report_if_fail();
        return false;
    }

    ID3D11Texture2D* pAcquiredDesktopImage = nullptr;
    auto t_qi_s = std::chrono::high_resolution_clock::now();
    hr = pDesktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&pAcquiredDesktopImage);
    auto t_qi_e = std::chrono::high_resolution_clock::now();
    t_qi_ms = std::chrono::duration<double, std::milli>(t_qi_e - t_qi_s).count();
    SafeRelease(&pDesktopResource);
    if (FAILED(hr)) {
        report_if_fail();
        dup->ReleaseFrame();
        return false;
    }

    UINT dirty_size = 0;
    HRESULT hr_dirty = dup->GetFrameDirtyRects(0, nullptr, &dirty_size);
    if (SUCCEEDED(hr_dirty) && dirty_size == 0) {
        bool served = serveCache(frame);
        SafeRelease(&pAcquiredDesktopImage);
        dup->ReleaseFrame();
        if (served) {
            ++statSkipDirty;
            ++statCacheRe;
            return true;
        }
        report_if_fail();
        return false;
    }

    D3D11_BOX sourceRegion;
    sourceRegion.left = crop_region.x;
    sourceRegion.right = crop_region.x + crop_region.width;
    sourceRegion.top = crop_region.y;
    sourceRegion.bottom = crop_region.y + crop_region.height;
    sourceRegion.front = 0;
    sourceRegion.back = 1;

    const size_t curr = idx % kStageN;
    const size_t prev = (idx + kStageN - 1) % kStageN;

    auto t_copy_s = std::chrono::high_resolution_clock::now();
    ctx->CopySubresourceRegion(stage[curr], 0, 0, 0, 0, pAcquiredDesktopImage, 0, &sourceRegion);
    auto t_copy_e = std::chrono::high_resolution_clock::now();
    t_copy_ms = std::chrono::duration<double, std::milli>(t_copy_e - t_copy_s).count();

    ctx->End(copyQ[curr]);

    SafeRelease(&pAcquiredDesktopImage);
    dup->ReleaseFrame();

    BOOL copy_done = FALSE;
    HRESULT qhr = ctx->GetData(copyQ[prev], &copy_done, sizeof(copy_done), 0);
    if (qhr != S_OK || !copy_done) {
        ++statEvtWait;
        bool served = serveCache(frame);
        if (served) ++statCacheRe;
        t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
    if (cfg.capture_timing && !cfg.capture_timing_on_success) { 
            reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
        }
        idx++;
        return served;
    }

    D3D11_MAPPED_SUBRESOURCE mappedResource;
    auto t_map_s = std::chrono::high_resolution_clock::now();
    hr = ctx->Map(stage[prev], 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mappedResource);
    auto t_map_e = std::chrono::high_resolution_clock::now();
    t_map_ms = std::chrono::duration<double, std::milli>(t_map_e - t_map_s).count();
    if (hr == DXGI_ERROR_WAS_STILL_DRAWING) {
        ++statStillDraw;
        bool served = serveCache(frame);
        if (served) ++statCacheRe;
        t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
    if (cfg.capture_timing && !cfg.capture_timing_on_success) { 
            reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
        }
        idx++;
        return served;
    }
    if (FAILED(hr)) {
        report_if_fail();
        idx++;
        return false;
    }

    cv::Mat bgra_frame(crop_region.height, crop_region.width, CV_8UC4, mappedResource.pData, mappedResource.RowPitch);
    auto t_cvt_s = std::chrono::high_resolution_clock::now();
    cv::cvtColor(bgra_frame, frame, cv::COLOR_BGRA2BGR);
    frame.copyTo(cache);
    auto t_cvt_e = std::chrono::high_resolution_clock::now();
    t_cvt_ms = std::chrono::duration<double, std::milli>(t_cvt_e - t_cvt_s).count();

    auto t_unmap_s = std::chrono::high_resolution_clock::now();
    ctx->Unmap(stage[prev], 0);
    auto t_unmap_e = std::chrono::high_resolution_clock::now();
    t_unmap_ms = std::chrono::duration<double, std::milli>(t_unmap_e - t_unmap_s).count();

    t_total_ms = std::chrono::duration<double, std::milli>(t_unmap_e - t_total_start).count();
    idx++;
    if (cfg.capture_timing) {
        reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
    }
    ++statOk;
    dbgTick();
    return true;
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
