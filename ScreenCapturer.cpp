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

// WinRT (WGC)
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <winrt/base.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

namespace {
    // Helper: get DXGI interface from WinRT object
    template<typename T>
    winrt::com_ptr<T> GetDXGIInterfaceFromObject(winrt::Windows::Foundation::IInspectable const& object) {
        winrt::com_ptr<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess> access = object.as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
        winrt::com_ptr<T> result;
        winrt::check_hresult(access->GetInterface(winrt::guid_of<T>(), result.put_void()));
        return result;
    }
    // Helper: create WinRT D3D device from DXGI device
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice CreateDirect3DDevice(IDXGIDevice* dxgiDevice) {
        winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice device{ nullptr };
        winrt::com_ptr<IInspectable> inspectable;
        winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice, reinterpret_cast<IInspectable**>(inspectable.put())));
        device = inspectable.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();
        return device;
    }
}

struct ScreenCapturer::WgcCtx {
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice wdev{ nullptr };
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem item{ nullptr };
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool pool{ nullptr };
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession sess{ nullptr };
};

ScreenCapturer::ScreenCapturer(const Config& cfg_ref, int crop_width, int crop_height)
    : cfg(cfg_ref), reporter(cfg_ref) {
    // Initialize WinRT apartment (MT) for WGC
    winrt::init_apartment(winrt::apartment_type::multi_threaded);
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
        // DDA
        if (FAILED(out->QueryInterface(__uuidof(IDXGIOutput1), (void**)&out1))) {
            throw std::runtime_error("Failed to query IDXGIOutput1.");
        }
        hr = out1->DuplicateOutput(dev, &dup);
        if (FAILED(hr)) {
            lg::ok_or_hrfail_if(true, P_C, "DuplicateOutput", hr, SUCCEEDED(hr));
            throw std::runtime_error("Failed to create output duplication.");
        }
    } else {
        // WGC
        wgc = new WgcCtx();
        // Create WinRT device from DXGI device
        winrt::com_ptr<IDXGIDevice> dxgiDev;
        dev->QueryInterface(__uuidof(IDXGIDevice), dxgiDev.put_void());
        wgc->wdev = CreateDirect3DDevice(dxgiDev.get());
        // Capture item from HMONITOR
        winrt::com_ptr<IGraphicsCaptureItemInterop> interop =
            winrt::get_activation_factory<winrt::Windows::Graphics::Capture::GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
        winrt::Windows::Graphics::Capture::GraphicsCaptureItem item{ nullptr };
        winrt::check_hresult(interop->CreateForMonitor(outputDesc.Monitor, winrt::guid_of<winrt::Windows::Graphics::Capture::GraphicsCaptureItem>(), winrt::put_abi(item)));
        wgc->item = item;
        // Frame pool and session
        auto size = winrt::Windows::Graphics::SizeInt32{ width, height };
        wgc->pool = winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::Create(
            wgc->wdev,
            winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized,
            2, size);
        wgc->sess = wgc->pool.CreateCaptureSession(wgc->item);
        wgc->sess.IsCursorCaptureEnabled(cfg.cap_cursor);
        wgc->sess.IsBorderRequired(cfg.cap_border);
        wgc->sess.StartCapture();
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
    if (useWgc && wgc) {
        // WGC resources auto-release via WinRT; explicit session stop by dtor
        delete wgc; wgc = nullptr;
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
    if (useWgc) {
        // WGC path: pull frame from frame pool, copy to staging, map and convert
        auto t_total_start = std::chrono::high_resolution_clock::now();
        double t_acquire_ms = 0.0, t_qi_ms = 0.0, t_copy_ms = 0.0, t_map_ms = 0.0, t_cvt_ms = 0.0, t_unmap_ms = 0.0, t_total_ms = 0.0;

        auto t_acq_s = std::chrono::high_resolution_clock::now();
        auto frm = wgc->pool.TryGetNextFrame();
        auto t_acq_e = std::chrono::high_resolution_clock::now();
        t_acquire_ms = std::chrono::duration<double, std::milli>(t_acq_e - t_acq_s).count();

        if (!frm) {
            // timeout or no frame yet -> serve cache
            bool served = serveCache(frame);
            if (!served && cfg.capture_timing && !cfg.capture_timing_on_success) {
                t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
                reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
            }
            return served;
        }

        // Get surface as ID3D11Texture2D
        auto surf = frm.Surface();
        auto tex = GetDXGIInterfaceFromObject<ID3D11Texture2D>(surf);

        D3D11_BOX src; src.left = crop_region.x; src.top = crop_region.y;
        src.right = crop_region.x + crop_region.width; src.bottom = crop_region.y + crop_region.height;
        src.front = 0; src.back = 1;

        const size_t curr = idx % kStageN;
        const size_t prev = (idx + kStageN - 1) % kStageN;

        auto t_copy_s = std::chrono::high_resolution_clock::now();
        ctx->CopySubresourceRegion(stage[curr], 0, 0, 0, 0, tex.get(), 0, &src);
        auto t_copy_e = std::chrono::high_resolution_clock::now();
        t_copy_ms = std::chrono::duration<double, std::milli>(t_copy_e - t_copy_s).count();

        ctx->End(copyQ[curr]);

        BOOL copy_done = FALSE;
        HRESULT qhr = ctx->GetData(copyQ[prev], &copy_done, sizeof(copy_done), 0);
        if (qhr != S_OK || !copy_done) {
            ++statEvtWait;
            bool served = serveCache(frame);
            if (served) ++statCacheRe;
            t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
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

        D3D11_MAPPED_SUBRESOURCE mr;
        auto t_map_s = std::chrono::high_resolution_clock::now();
        HRESULT hrm = ctx->Map(stage[prev], 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mr);
        auto t_map_e = std::chrono::high_resolution_clock::now();
        t_map_ms = std::chrono::duration<double, std::milli>(t_map_e - t_map_s).count();
        if (hrm == DXGI_ERROR_WAS_STILL_DRAWING) {
            ++statStillDraw;
            bool served = serveCache(frame);
            if (served) ++statCacheRe;
            t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
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
        if (FAILED(hrm)) {
            idx++;
            return false;
        }

        cv::Mat bgra(crop_region.height, crop_region.width, CV_8UC4, mr.pData, mr.RowPitch);
        auto t_cvt_s = std::chrono::high_resolution_clock::now();
        cv::cvtColor(bgra, frame, cv::COLOR_BGRA2BGR);
        frame.copyTo(cache);
        auto t_cvt_e = std::chrono::high_resolution_clock::now();
        t_cvt_ms = std::chrono::duration<double, std::milli>(t_cvt_e - t_cvt_s).count();

        auto t_unmap_s = std::chrono::high_resolution_clock::now();
        ctx->Unmap(stage[prev], 0);
        auto t_unmap_e = std::chrono::high_resolution_clock::now();
        t_unmap_ms = std::chrono::duration<double, std::milli>(t_unmap_e - t_unmap_s).count();

        double t_total_ms_loc = std::chrono::duration<double, std::milli>(t_unmap_e - t_total_start).count();
        idx++;
        if (cfg.capture_timing) reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms_loc);
        ++statOk;
        dbgTick();
        return true;
    }

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
            if (cfg.capture_timing && cfg.capture_timing_on_success) {
                t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
                reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
            }
            return true;
        }
        if (cfg.capture_timing && !cfg.capture_timing_on_success) {
            t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
            reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
        }
        return false;
    }
    if (FAILED(hr)) {
        lg::ok_or_hrfail_if(true, P_C, "AcquireNextFrame", hr, SUCCEEDED(hr));
        if (cfg.capture_timing && !cfg.capture_timing_on_success) {
            t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
            reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
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
                t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
                reportTimes(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
            }
            return true;
        }
        if (cfg.capture_timing && !cfg.capture_timing_on_success) {
            t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
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
    auto t_map_s = std::chrono::high_resolution_clock::now();
    hr = ctx->Map(stage[prev], 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mappedResource);
    auto t_map_e = std::chrono::high_resolution_clock::now();
    t_map_ms = std::chrono::duration<double, std::milli>(t_map_e - t_map_s).count();
    if (hr == DXGI_ERROR_WAS_STILL_DRAWING) {
        ++statStillDraw;
        bool served = serveCache(frame);
        if (served) ++statCacheRe;
        t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
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
