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
    const bool do_tim = cfg.capture_timing && cfg.timing;
    using clock = std::chrono::high_resolution_clock;
    auto t_total_start = do_tim ? clock::now() : clock::time_point{};
    double aq = 0.0, qi = 0.0, cp = 0.0, mp = 0.0, cv = 0.0, up = 0.0, tt = 0.0;

    auto report_if_fail = [&]() {
        if (do_tim && !cfg.capture_timing_on_success) {
            tt = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
            reportTimes(aq, qi, cp, mp, cv, up, tt);
        }
    };

    IDXGIResource* pDesktopResource = nullptr;
    DXGI_OUTDUPL_FRAME_INFO frameInfo;

    auto t_acq_s = do_tim ? clock::now() : clock::time_point{}; // aq
    HRESULT hr = dup->AcquireNextFrame(cfg.capture_timeout_ms_dxgi, &frameInfo, &pDesktopResource);
    auto t_acq_e = do_tim ? clock::now() : t_acq_s;
    if (do_tim) aq = std::chrono::duration<double, std::milli>(t_acq_e - t_acq_s).count();

    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
        if (serveCache(frame)) {
            ++statCacheRe;
            if (cfg.capture_timing && cfg.capture_timing_on_success) {
                tt = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
                reportTimes(aq, qi, cp, mp, cv, up, tt);
            }
            return true;
        }
        if (cfg.capture_timing && !cfg.capture_timing_on_success) {
            tt = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
            reportTimes(aq, qi, cp, mp, cv, up, tt);
        }
        return false;
    }
    if (FAILED(hr)) {
        lg::ok_or_hrfail_if(true, P_C, "AcquireNextFrame", hr, SUCCEEDED(hr));
        if (cfg.capture_timing && !cfg.capture_timing_on_success) {
            tt = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
            reportTimes(aq, qi, cp, mp, cv, up, tt);
        }
        return false;
    }

    ID3D11Texture2D* pAcquiredDesktopImage = nullptr;
    auto t_qi_s = do_tim ? clock::now() : clock::time_point{};
    hr = pDesktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&pAcquiredDesktopImage);
    auto t_qi_e = do_tim ? clock::now() : t_qi_s;
    if (do_tim) qi = std::chrono::duration<double, std::milli>(t_qi_e - t_qi_s).count();
    SafeRelease(&pDesktopResource);
    if (FAILED(hr)) {
        if (cfg.capture_timing && !cfg.capture_timing_on_success) {
            tt = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
            reportTimes(aq, qi, cp, mp, cv, up, tt);
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
                tt = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
                reportTimes(aq, qi, cp, mp, cv, up, tt);
            }
            return true;
        }
        if (cfg.capture_timing && !cfg.capture_timing_on_success) {
            tt = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
            reportTimes(aq, qi, cp, mp, cv, up, tt);
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

    auto t_copy_s = do_tim ? clock::now() : clock::time_point{};
    ctx->CopySubresourceRegion(stage[curr], 0, 0, 0, 0, pAcquiredDesktopImage, 0, &sourceRegion);
    auto t_copy_e = do_tim ? clock::now() : t_copy_s;
    if (do_tim) cp = std::chrono::duration<double, std::milli>(t_copy_e - t_copy_s).count();

    ctx->End(copyQ[curr]);

    SafeRelease(&pAcquiredDesktopImage);
    dup->ReleaseFrame();

    BOOL copy_done = FALSE;
    HRESULT qhr = ctx->GetData(copyQ[prev], &copy_done, sizeof(copy_done), 0);
    if (qhr != S_OK || !copy_done) {
        ++statCopyPend;
        bool served = serveCache(frame);
        if (served) ++statCacheRe;
        tt = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
        if (cfg.capture_timing) {
            if (served) {
                if (cfg.capture_timing_on_success) {
                    reportTimes(aq, qi, cp, mp, cv, up, tt);
                }
            } else {
                if (!cfg.capture_timing_on_success) {
                    reportTimes(aq, qi, cp, mp, cv, up, tt);
                }
            }
        }
        idx++;
        return served;
    }

    D3D11_MAPPED_SUBRESOURCE mappedResource;
    auto t_map_s = do_tim ? clock::now() : clock::time_point{};
    hr = ctx->Map(stage[prev], 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mappedResource);
    auto t_map_e = do_tim ? clock::now() : t_map_s;
    if (do_tim) mp = std::chrono::duration<double, std::milli>(t_map_e - t_map_s).count();
    if (hr == DXGI_ERROR_WAS_STILL_DRAWING) {
        ++statStillDraw;
        bool served = serveCache(frame);
        if (served) ++statCacheRe;
        tt = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
        if (cfg.capture_timing) {
            if (served) {
                if (cfg.capture_timing_on_success) {
                    reportTimes(aq, qi, cp, mp, cv, up, tt);
                }
            } else {
                if (!cfg.capture_timing_on_success) {
                    reportTimes(aq, qi, cp, mp, cv, up, tt);
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
    auto t_cvt_s = do_tim ? clock::now() : clock::time_point{};
    cv::cvtColor(bgra_frame, frame, cv::COLOR_BGRA2BGR);
    frame.copyTo(cache);
    auto t_cvt_e = do_tim ? clock::now() : t_cvt_s;
    if (do_tim) cv = std::chrono::duration<double, std::milli>(t_cvt_e - t_cvt_s).count();

    auto t_unmap_s = do_tim ? clock::now() : clock::time_point{};
    ctx->Unmap(stage[prev], 0);
    auto t_unmap_e = do_tim ? clock::now() : t_unmap_s;
    if (do_tim) up = std::chrono::duration<double, std::milli>(t_unmap_e - t_unmap_s).count();

    tt = std::chrono::duration<double, std::milli>(t_unmap_e - t_total_start).count();
    idx++;
    if (cfg.capture_timing) {
        reportTimes(aq, qi, cp, mp, cv, up, tt);
    }
    ++statOk;
    dbgTick();
    return true;
}
