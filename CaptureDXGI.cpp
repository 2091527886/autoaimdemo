#include "ScreenCapturer.hpp"
#include "Config.hpp"
#include "Log.hpp"
#include "ComHelpers.hpp"

#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <chrono>
#include <stdexcept>
#include <algorithm>

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

    // 累计分段计时（多次尝试累加）
    double aq_sum = 0.0, qi_sum = 0.0, cp_sum = 0.0, mp_sum = 0.0, cv_sum = 0.0, up_sum = 0.0;
    auto t_total_start = do_tim ? clock::now() : clock::time_point{};

    // 总尝试次数（首次+重试）
    const int tries = std::max(1, 1 + cfg.cap_retry);

    for (int attempt = 0; attempt < tries; ++attempt) {
        double aq = 0.0, qi = 0.0, cp = 0.0, mp = 0.0, cvc = 0.0, up = 0.0;

        IDXGIResource* pDesktopResource = nullptr;
        DXGI_OUTDUPL_FRAME_INFO frameInfo{};

        // Acquire
        auto t_acq_s = do_tim ? clock::now() : clock::time_point{};
        HRESULT hr = dup->AcquireNextFrame(cfg.capture_timeout_ms_dxgi, &frameInfo, &pDesktopResource);
        auto t_acq_e = do_tim ? clock::now() : t_acq_s;
        if (do_tim) aq = std::chrono::duration<double, std::milli>(t_acq_e - t_acq_s).count();

        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
            aq_sum += aq; // 计入获取等待
            // 无新帧，立即重试
            continue;
        }
        if (FAILED(hr)) {
            // 致命失败：结束并上报失败
            aq_sum += aq;
            if (do_tim && !cfg.capture_timing_on_success) {
                double tt = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
                reportTimes(aq_sum, qi_sum, cp_sum, mp_sum, cv_sum, up_sum, tt);
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
            // 本次尝试失败，释放并重试
            qi_sum += qi;
            dup->ReleaseFrame();
            continue;
        }

        // 脏矩形为0：跳过（无新内容），立即重试
        UINT dirty_size = 0;
        HRESULT hr_dirty = dup->GetFrameDirtyRects(0, nullptr, &dirty_size);
        if (SUCCEEDED(hr_dirty) && dirty_size == 0) {
            ++statSkipDirty;
            qi_sum += qi;
            SafeRelease(&pAcquiredDesktopImage);
            dup->ReleaseFrame();
            continue;
        }

        // Copy
        D3D11_BOX sourceRegion{};
        sourceRegion.left   = crop_region.x;
        sourceRegion.right  = crop_region.x + crop_region.width;
        sourceRegion.top    = crop_region.y;
        sourceRegion.bottom = crop_region.y + crop_region.height;
        sourceRegion.front  = 0;
        sourceRegion.back   = 1;

        auto t_copy_s = do_tim ? clock::now() : clock::time_point{};
        ctx->CopySubresourceRegion(stage[0], 0, 0, 0, 0, pAcquiredDesktopImage, 0, &sourceRegion);
        auto t_copy_e = do_tim ? clock::now() : t_copy_s;
        if (do_tim) cp = std::chrono::duration<double, std::milli>(t_copy_e - t_copy_s).count();

        if (cfg.capture_flush_after_copy) {
            ctx->Flush();
        }

        SafeRelease(&pAcquiredDesktopImage);
        dup->ReleaseFrame();

        // Map (blocking)
        D3D11_MAPPED_SUBRESOURCE mr;
        auto t_map_s = do_tim ? clock::now() : clock::time_point{};
        hr = ctx->Map(stage[0], 0, D3D11_MAP_READ, 0, &mr);
        auto t_map_e = do_tim ? clock::now() : t_map_s;
        if (do_tim) mp = std::chrono::duration<double, std::milli>(t_map_e - t_map_s).count();
        if (FAILED(hr)) {
            aq_sum += aq; qi_sum += qi; cp_sum += cp; mp_sum += mp;
            continue;
        }

        // Convert
        cv::Mat bgra(crop_region.height, crop_region.width, CV_8UC4, mr.pData, mr.RowPitch);
        auto t_cvt_s = do_tim ? clock::now() : clock::time_point{};
        frame = bgra.clone(); // keep BGRA, avoid CPU-side color conversion
        auto t_cvt_e = do_tim ? clock::now() : t_cvt_s;
        if (do_tim) cvc = std::chrono::duration<double, std::milli>(t_cvt_e - t_cvt_s).count();

        // Unmap
        auto t_unmap_s = do_tim ? clock::now() : clock::time_point{};
        ctx->Unmap(stage[0], 0);
        auto t_unmap_e = do_tim ? clock::now() : t_unmap_s;
        if (do_tim) up = std::chrono::duration<double, std::milli>(t_unmap_e - t_unmap_s).count();

        // 累计本次尝试
        aq_sum += aq; qi_sum += qi; cp_sum += cp; mp_sum += mp; cv_sum += cvc; up_sum += up;

        // 成功
        double tt = do_tim ? std::chrono::duration<double, std::milli>(t_unmap_e - t_total_start).count() : 0.0;
        if (cfg.capture_timing && cfg.capture_timing_on_success) {
            reportTimes(aq_sum, qi_sum, cp_sum, mp_sum, cv_sum, up_sum, tt);
        }
        ++statOk;
        dbgTick();
        return true;
    }

    // 全部尝试失败
    if (cfg.capture_timing && !cfg.capture_timing_on_success) {
        double tt = do_tim ? std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count() : 0.0;
        reportTimes(aq_sum, qi_sum, cp_sum, mp_sum, cv_sum, up_sum, tt);
    }
    return false;
}
