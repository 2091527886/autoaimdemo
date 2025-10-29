#include "ScreenCapturer.hpp"
#include "Config.hpp"
#include "DxgiTuning.hpp"
#include "ComHelpers.hpp"
#include "metrics/Reporting.hpp"

#include <array>
#include <chrono>
#include <stdexcept>
#include <iostream>

// Windows & DirectX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

ScreenCapturer::ScreenCapturer(const Config& cfg_ref, int crop_width, int crop_height)
    : cfg(cfg_ref), capture_reporter_(cfg_ref) {
    if (cfg.log_init_verbose) std::cout << "--- Initializing D3D for screen capture... ---" << std::endl;
    if (cfg.log_init_verbose) {
        std::cout << "[CAPTURE] Crop size: " << crop_width << "x" << crop_height << std::endl;
    }
    HRESULT hr;

    hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&pFactory);
    if (cfg.log_init_verbose) {
        std::cout << "[CAPTURE] CreateDXGIFactory1 hr=0x"
                  << std::hex << std::uppercase << static_cast<unsigned long>(hr)
                  << std::dec << std::nouppercase << std::endl;
    }
    if (FAILED(hr)) throw std::runtime_error("Failed to create DXGI Factory.");
    {
        HRESULT hr_enum_adapters = pFactory->EnumAdapters1(0, &pAdapter);
        if (cfg.log_init_verbose) {
            std::cout << "[CAPTURE] EnumAdapters1 hr=0x"
                      << std::hex << std::uppercase << static_cast<unsigned long>(hr_enum_adapters)
                      << std::dec << std::nouppercase << std::endl;
        }
        if (FAILED(hr_enum_adapters)) throw std::runtime_error("Failed to enumerate adapters.");
    }
    {
        HRESULT hr_enum_outputs = pAdapter->EnumOutputs(0, &pOutput);
        if (cfg.log_init_verbose) {
            std::cout << "[CAPTURE] EnumOutputs hr=0x"
                      << std::hex << std::uppercase << static_cast<unsigned long>(hr_enum_outputs)
                      << std::dec << std::nouppercase << std::endl;
        }
        if (FAILED(hr_enum_outputs)) throw std::runtime_error("Failed to enumerate outputs.");
    }

    DXGI_OUTPUT_DESC outputDesc;
    pOutput->GetDesc(&outputDesc);
    width = outputDesc.DesktopCoordinates.right - outputDesc.DesktopCoordinates.left;
    height = outputDesc.DesktopCoordinates.bottom - outputDesc.DesktopCoordinates.top;
    if (cfg.log_init_verbose) {
        std::cout << "[CAPTURE] Output desktop size: " << width << "x" << height << std::endl;
    }

    {
        HRESULT hr_create_device = D3D11CreateDevice(pAdapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &pDevice, nullptr, &pContext);
        if (cfg.log_init_verbose) {
            std::cout << "[CAPTURE] D3D11CreateDevice hr=0x"
                      << std::hex << std::uppercase << static_cast<unsigned long>(hr_create_device)
                      << std::dec << std::nouppercase << std::endl;
        }
        if (FAILED(hr_create_device)) {
            throw std::runtime_error("Failed to create D3D11 device.");
        }
        // Raise DXGI GPU priority and reduce frame latency for this device
        RaiseDxgiGpuPriorityAndLatency(pDevice, cfg.dxgi_gpu_prio, cfg.dxgi_max_latency);
    }
    {
        HRESULT hr_qi_output1 = pOutput->QueryInterface(__uuidof(IDXGIOutput1), (void**)&pOutput1);
        if (cfg.log_init_verbose) {
            std::cout << "[CAPTURE] QueryInterface(IDXGIOutput1) hr=0x"
                      << std::hex << std::uppercase << static_cast<unsigned long>(hr_qi_output1)
                      << std::dec << std::nouppercase << std::endl;
        }
        if (FAILED(hr_qi_output1)) {
            throw std::runtime_error("Failed to query IDXGIOutput1.");
        }
    }
    hr = pOutput1->DuplicateOutput(pDevice, &pDuplicator);
    if (cfg.log_init_verbose) {
        std::cout << "[CAPTURE] DuplicateOutput hr=" << (FAILED(hr) ? "FAILED" : "OK") << std::endl;
    }
    if (FAILED(hr)) {
        std::cerr << "DuplicateOutput failed. HRESULT=0x"
                  << std::hex << std::uppercase << static_cast<unsigned long>(hr)
                  << std::dec << std::nouppercase << std::endl;
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
    for (size_t i = 0; i < kStagingCount; ++i) {
        hr_local = pDevice->CreateTexture2D(&desc, NULL, &staging_[i]);
        if (FAILED(hr_local)) {
            throw std::runtime_error("Failed to create staging texture[" + std::to_string(i) + "]");
        }
        D3D11_QUERY_DESC qd{}; qd.Query = D3D11_QUERY_EVENT; qd.MiscFlags = 0;
        hr_local = pDevice->CreateQuery(&qd, &copy_done_queries_[i]);
        if (FAILED(hr_local)) {
            throw std::runtime_error("Failed to create EVENT query for staging[" + std::to_string(i) + "]");
        }
    }
    std::cout << "--- Screen capture initialized successfully (" << width << "x" << height << ") with " << kStagingCount << " staging buffers ---" << std::endl;

    // 捕获链路启动横幅（仅在启用报告时）
    capture_reporter_.BootBanner(kStagingCount);
}

ScreenCapturer::~ScreenCapturer() {
    std::cout << "--- Cleaning up D3D resources... ---" << std::endl;
    for (size_t i = 0; i < kStagingCount; ++i) {
        SafeRelease(&staging_[i]);
        SafeRelease(&copy_done_queries_[i]);
    }
    SafeRelease(&pDuplicator);
    SafeRelease(&pOutput1);
    SafeRelease(&pOutput);
    SafeRelease(&pAdapter);
    SafeRelease(&pFactory);
    SafeRelease(&pContext);
    SafeRelease(&pDevice);
}

bool ScreenCapturer::CaptureFrame(cv::Mat& frame, const cv::Rect& crop_region) {
    auto t_total_start = std::chrono::high_resolution_clock::now();
    double t_acquire_ms = 0.0, t_qi_ms = 0.0, t_copy_ms = 0.0, t_map_ms = 0.0, t_cvt_ms = 0.0, t_unmap_ms = 0.0, t_total_ms = 0.0;

    IDXGIResource* pDesktopResource = nullptr;
    DXGI_OUTDUPL_FRAME_INFO frameInfo;

    auto t_acq_s = std::chrono::high_resolution_clock::now();
    HRESULT hr = pDuplicator->AcquireNextFrame(cfg.capture_timeout_ms, &frameInfo, &pDesktopResource);
    auto t_acq_e = std::chrono::high_resolution_clock::now();
    t_acquire_ms = std::chrono::duration<double, std::milli>(t_acq_e - t_acq_s).count();

    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
        if (try_serve_cache(frame)) {
            ++stat_cache_reused_;
            return true;
        }
    if (cfg.capture_timing && !cfg.capture_timing_on_success) { 
            t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
            accumulate_and_maybe_print(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
        }
        return false;
    }
    if (FAILED(hr)) {
        std::cerr << "AcquireNextFrame failed. HRESULT: 0x" << std::hex << hr << std::endl;
    if (cfg.capture_timing && !cfg.capture_timing_on_success) { 
            t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
            accumulate_and_maybe_print(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
        }
        return false;
    }

    ID3D11Texture2D* pAcquiredDesktopImage = nullptr;
    auto t_qi_s = std::chrono::high_resolution_clock::now();
    hr = pDesktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&pAcquiredDesktopImage);
    auto t_qi_e = std::chrono::high_resolution_clock::now();
    t_qi_ms = std::chrono::duration<double, std::milli>(t_qi_e - t_qi_s).count();
    SafeRelease(&pDesktopResource);
    if (FAILED(hr)) {
    if (cfg.capture_timing && !cfg.capture_timing_on_success) { 
            t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
            accumulate_and_maybe_print(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
        }
        return false;
    }

    UINT dirty_size = 0;
    HRESULT hr_dirty = pDuplicator->GetFrameDirtyRects(0, nullptr, &dirty_size);
    if (SUCCEEDED(hr_dirty) && dirty_size == 0) {
        bool served = try_serve_cache(frame);
        SafeRelease(&pDesktopResource);
        SafeRelease(&pAcquiredDesktopImage);
        pDuplicator->ReleaseFrame();
        if (served) {
            ++stat_dirty_skips_;
            ++stat_cache_reused_;
            return true;
        }
        t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
    if (cfg.capture_timing && !cfg.capture_timing_on_success) { 
            accumulate_and_maybe_print(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
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

    const size_t curr = frame_index_ % kStagingCount;
    const size_t prev = (frame_index_ + kStagingCount - 1) % kStagingCount;

    auto t_copy_s = std::chrono::high_resolution_clock::now();
    pContext->CopySubresourceRegion(staging_[curr], 0, 0, 0, 0, pAcquiredDesktopImage, 0, &sourceRegion);
    auto t_copy_e = std::chrono::high_resolution_clock::now();
    t_copy_ms = std::chrono::duration<double, std::milli>(t_copy_e - t_copy_s).count();

    pContext->End(copy_done_queries_[curr]);

    SafeRelease(&pAcquiredDesktopImage);
    pDuplicator->ReleaseFrame();

    BOOL copy_done = FALSE;
    HRESULT qhr = pContext->GetData(copy_done_queries_[prev], &copy_done, sizeof(copy_done), 0);
    if (qhr != S_OK || !copy_done) {
        ++stat_event_not_ready_;
        bool served = try_serve_cache(frame);
        if (served) ++stat_cache_reused_;
        t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
    if (cfg.capture_timing && !cfg.capture_timing_on_success) { 
            accumulate_and_maybe_print(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
        }
        frame_index_++;
        return served;
    }

    D3D11_MAPPED_SUBRESOURCE mappedResource;
    auto t_map_s = std::chrono::high_resolution_clock::now();
    hr = pContext->Map(staging_[prev], 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mappedResource);
    auto t_map_e = std::chrono::high_resolution_clock::now();
    t_map_ms = std::chrono::duration<double, std::milli>(t_map_e - t_map_s).count();
    if (hr == DXGI_ERROR_WAS_STILL_DRAWING) {
        ++stat_map_still_drawing_;
        bool served = try_serve_cache(frame);
        if (served) ++stat_cache_reused_;
        t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
    if (cfg.capture_timing && !cfg.capture_timing_on_success) { 
            accumulate_and_maybe_print(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
        }
        frame_index_++;
        return served;
    }
    if (FAILED(hr)) {
        t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
    if (cfg.capture_timing && !cfg.capture_timing_on_success) { 
            accumulate_and_maybe_print(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
        }
        frame_index_++;
        return false;
    }

    cv::Mat bgra_frame(crop_region.height, crop_region.width, CV_8UC4, mappedResource.pData, mappedResource.RowPitch);
    auto t_cvt_s = std::chrono::high_resolution_clock::now();
    cv::cvtColor(bgra_frame, frame, cv::COLOR_BGRA2BGR);
    frame.copyTo(last_bgr_cache_);
    auto t_cvt_e = std::chrono::high_resolution_clock::now();
    t_cvt_ms = std::chrono::duration<double, std::milli>(t_cvt_e - t_cvt_s).count();

    auto t_unmap_s = std::chrono::high_resolution_clock::now();
    pContext->Unmap(staging_[prev], 0);
    auto t_unmap_e = std::chrono::high_resolution_clock::now();
    t_unmap_ms = std::chrono::duration<double, std::milli>(t_unmap_e - t_unmap_s).count();

    t_total_ms = std::chrono::duration<double, std::milli>(t_unmap_e - t_total_start).count();
    frame_index_++;
    if (cfg.capture_timing && !cfg.capture_timing_on_success) { 
        accumulate_and_maybe_print(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
    }
    ++stat_success_;
    return true;
}

bool ScreenCapturer::try_serve_cache(cv::Mat& out) {
    if (!last_bgr_cache_.empty()) {
        if (last_bgr_cache_.rows == out.rows && last_bgr_cache_.cols == out.cols && last_bgr_cache_.type() == out.type()) {
            last_bgr_cache_.copyTo(out);
            return true;
        } else {
            out = last_bgr_cache_.clone();
            return true;
        }
    }
    return false;
}

void ScreenCapturer::accumulate_and_maybe_print(double t_acquire_ms, double t_qi_ms, double t_copy_ms,
                                    double t_map_ms, double t_cvt_ms, double t_unmap_ms, double t_total_ms) {
    capture_reporter_.ReportStepTimings(
        t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms,
        stat_dirty_skips_, stat_event_not_ready_, stat_map_still_drawing_, stat_cache_reused_, stat_success_
    );
}
