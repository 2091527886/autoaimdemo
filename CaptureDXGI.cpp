#include "ScreenCapturer.hpp"
#include "Config.hpp"
#include "Log.hpp"
#include "ComHelpers.hpp"

#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <chrono>
#include <stdexcept>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

bool ScreenCapturer::init_dxgi() {
    // DDA init (moved from ctor, logic unchanged)
    if (FAILED(out->QueryInterface(__uuidof(IDXGIOutput1), (void**)&out1))) {
        throw std::runtime_error("Failed to query IDXGIOutput1.");
    }
    HRESULT hr = out1->DuplicateOutput(dev, &dup);
    if (FAILED(hr)) {
        lg::ok_or_hrfail_if(true, P_C, "DuplicateOutput", hr, SUCCEEDED(hr));
        throw std::runtime_error("Failed to create output duplication.");
    }
    return true;
}

bool ScreenCapturer::cap_dxgi(cv::Mat& frame, const cv::Rect& crop_region) {
    using clock = std::chrono::high_resolution_clock;
    auto t_total_start = clock::now();
    double t_acquire_ms = 0.0, t_qi_ms = 0.0, t_copy_ms = 0.0, t_map_ms = 0.0, t_cvt_ms = 0.0, t_unmap_ms = 0.0, t_total_ms = 0.0;

    auto report_if_fail = [&]() {
        if (cfg.capture_timing && !cfg.capture_timing_on_success) {
            t_total_ms = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
            reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
        }
    };

    IDXGIResource* pDesktopResource = nullptr;
    DXGI_OUTDUPL_FRAME_INFO frameInfo;

    auto t_acq_s = clock::now();
    HRESULT hr = dup->AcquireNextFrame(cfg.capture_timeout_ms, &frameInfo, &pDesktopResource);
    auto t_acq_e = clock::now();
    t_acquire_ms = std::chrono::duration<double, std::milli>(t_acq_e - t_acq_s).count();

    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
        if (serveCache(frame)) {
            ++statCacheRe;
            if (cfg.capture_timing && cfg.capture_timing_on_success) {
                t_total_ms = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
                reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
            }
            return true;
        }
        if (cfg.capture_timing && !cfg.capture_timing_on_success) {
            t_total_ms = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
            reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
        }
        return false;
    }
    if (FAILED(hr)) {
        lg::ok_or_hrfail_if(true, P_C, "AcquireNextFrame", hr, SUCCEEDED(hr));
        if (cfg.capture_timing && !cfg.capture_timing_on_success) {
            t_total_ms = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
            reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
        }
        return false;
    }

    ID3D11Texture2D* pAcquiredDesktopImage = nullptr;
    auto t_qi_s = clock::now();
    hr = pDesktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&pAcquiredDesktopImage);
    auto t_qi_e = clock::now();
    t_qi_ms = std::chrono::duration<double, std::milli>(t_qi_e - t_qi_s).count();
    SafeRelease(&pDesktopResource);
    if (FAILED(hr)) {
        if (cfg.capture_timing && !cfg.capture_timing_on_success) {
            t_total_ms = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
            reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
        }
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
            if (cfg.capture_timing && cfg.capture_timing_on_success) {
                t_total_ms = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
                reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
            }
            return true;
        }
        if (cfg.capture_timing && !cfg.capture_timing_on_success) {
            t_total_ms = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
            reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
        }
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

    auto t_copy_s = clock::now();
    ctx->CopySubresourceRegion(stage[curr], 0, 0, 0, 0, pAcquiredDesktopImage, 0, &sourceRegion);
    auto t_copy_e = clock::now();
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
        t_total_ms = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
        if (cfg.capture_timing) {
            if (served) {
                if (cfg.capture_timing_on_success) {
                    reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
                }
            } else {
                if (!cfg.capture_timing_on_success) {
                    reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
                }
            }
        }
        idx++;
        return served;
    }

    D3D11_MAPPED_SUBRESOURCE mappedResource;
    auto t_map_s = clock::now();
    hr = ctx->Map(stage[prev], 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mappedResource);
    auto t_map_e = clock::now();
    t_map_ms = std::chrono::duration<double, std::milli>(t_map_e - t_map_s).count();
    if (hr == DXGI_ERROR_WAS_STILL_DRAWING) {
        ++statStillDraw;
        bool served = serveCache(frame);
        if (served) ++statCacheRe;
        t_total_ms = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
        if (cfg.capture_timing) {
            if (served) {
                if (cfg.capture_timing_on_success) {
                    reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
                }
            } else {
                if (!cfg.capture_timing_on_success) {
                    reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
                }
            }
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
    auto t_cvt_s = clock::now();
    cv::cvtColor(bgra_frame, frame, cv::COLOR_BGRA2BGR);
    frame.copyTo(cache);
    auto t_cvt_e = clock::now();
    t_cvt_ms = std::chrono::duration<double, std::milli>(t_cvt_e - t_cvt_s).count();

    auto t_unmap_s = clock::now();
    ctx->Unmap(stage[prev], 0);
    auto t_unmap_e = clock::now();
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
