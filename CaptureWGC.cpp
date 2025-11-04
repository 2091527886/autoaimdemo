#include "ScreenCapturer.hpp"
#include "Config.hpp"
#include "ComHelpers.hpp"
#include "Log.hpp"

#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <chrono>

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
    template<typename T>
    winrt::com_ptr<T> GetDXGIInterfaceFromObject(winrt::Windows::Foundation::IInspectable const& object) {
        winrt::com_ptr<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess> access = object.as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
        winrt::com_ptr<T> result;
        winrt::check_hresult(access->GetInterface(winrt::guid_of<T>(), result.put_void()));
        return result;
    }
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
    winrt::event_token tok{};
};


bool ScreenCapturer::init_wgc() {
    // Initialize WinRT apartment (moved from ctor)
    winrt::init_apartment(winrt::apartment_type::multi_threaded);

    // HMONITOR from current output
    DXGI_OUTPUT_DESC od; out->GetDesc(&od);

    wgc = new WgcCtx();

    // Create WinRT device from DXGI device
    winrt::com_ptr<IDXGIDevice> dxgiDev;
    dev->QueryInterface(__uuidof(IDXGIDevice), dxgiDev.put_void());
    wgc->wdev = CreateDirect3DDevice(dxgiDev.get());

    // Capture item from HMONITOR
    winrt::com_ptr<IGraphicsCaptureItemInterop> interop =
        winrt::get_activation_factory<winrt::Windows::Graphics::Capture::GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem item{ nullptr };
    winrt::check_hresult(interop->CreateForMonitor(od.Monitor, winrt::guid_of<winrt::Windows::Graphics::Capture::GraphicsCaptureItem>(), winrt::put_abi(item)));
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

    // Create auto-reset frame event
    if (!fev) fev = CreateEventW(nullptr, FALSE, FALSE, nullptr);

    // Subscribe FrameArrived -> SetEvent(fev)
    wgc->tok = wgc->pool.FrameArrived([this](auto&&, auto&&){
        if (fev) SetEvent(fev);
    });

    wgc->sess.StartCapture();
    return true;
}

void ScreenCapturer::rel_wgc() {
    if (wgc) {
        if (wgc->pool && wgc->tok.value) {
            wgc->pool.FrameArrived(wgc->tok);
        }
        // session/pool will be released with wgc destruction
        delete wgc; wgc = nullptr;
    }
}

bool ScreenCapturer::cap_wgc(cv::Mat& frame, const cv::Rect& crop_region) {
    using clock = std::chrono::high_resolution_clock;
    auto t_total_start = clock::now();
    double t_acquire_ms = 0.0, t_qi_ms = 0.0, t_copy_ms = 0.0, t_map_ms = 0.0, t_cvt_ms = 0.0, t_unmap_ms = 0.0, t_total_ms = 0.0;

    auto t_acq_s = clock::now();
    auto frm = wgc->pool.TryGetNextFrame();
    auto t_acq_e = clock::now();
    t_acquire_ms = std::chrono::duration<double, std::milli>(t_acq_e - t_acq_s).count();
    // For WGC, treat second field as 'wa' (wait for frame event) collected in CapAsync
    double t_wait_ms = get_wait_ms();

    if (!frm) {
        bool served = serveCache(frame);
        if (!served && cfg.capture_timing && !cfg.capture_timing_on_success) {
            t_total_ms = std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count();
            // Map to WGC labels: gf=t_acquire_ms, wa=t_wait_ms
            reportTimes(t_acquire_ms, t_wait_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
        }
        return served;
    }

    auto surf = frm.Surface();
    auto tex = GetDXGIInterfaceFromObject<ID3D11Texture2D>(surf);

    D3D11_BOX src; src.left = crop_region.x; src.top = crop_region.y;
    src.right = crop_region.x + crop_region.width; src.bottom = crop_region.y + crop_region.height;
    src.front = 0; src.back = 1;

    const size_t curr = idx % kStageN;
    const size_t prev = (idx + kStageN - 1) % kStageN;

    auto t_copy_s = clock::now();
    ctx->CopySubresourceRegion(stage[curr], 0, 0, 0, 0, tex.get(), 0, &src);
    auto t_copy_e = clock::now();
    t_copy_ms = std::chrono::duration<double, std::milli>(t_copy_e - t_copy_s).count(); // cp

    ctx->End(copyQ[curr]);

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

    D3D11_MAPPED_SUBRESOURCE mr;
    auto t_map_s = clock::now();
    HRESULT hrm = ctx->Map(stage[prev], 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mr);
    auto t_map_e = clock::now();
    t_map_ms = std::chrono::duration<double, std::milli>(t_map_e - t_map_s).count(); // mp
    if (hrm == DXGI_ERROR_WAS_STILL_DRAWING) {
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
    if (FAILED(hrm)) {
        idx++;
        return false;
    }

    cv::Mat bgra(crop_region.height, crop_region.width, CV_8UC4, mr.pData, mr.RowPitch);
    auto t_cvt_s = clock::now();
    cv::cvtColor(bgra, frame, cv::COLOR_BGRA2BGR);
    frame.copyTo(cache);
    auto t_cvt_e = clock::now();
    t_cvt_ms = std::chrono::duration<double, std::milli>(t_cvt_e - t_cvt_s).count(); // cv

    auto t_unmap_s = clock::now();
    ctx->Unmap(stage[prev], 0);
    auto t_unmap_e = clock::now();
    t_unmap_ms = std::chrono::duration<double, std::milli>(t_unmap_e - t_unmap_s).count(); // up

    double t_total_ms_loc = std::chrono::duration<double, std::milli>(t_unmap_e - t_total_start).count();
    idx++;
    if (cfg.capture_timing) reportTimes(t_acquire_ms, t_wait_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms_loc); // gf, wa, cp, mp, cv, up, tt
    ++statOk;
    dbgTick();
    return true;
}
