#include "ScreenCapturer.hpp"
#include "Config.hpp"
#include "ComHelpers.hpp"
#include "Log.hpp"

#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <chrono>
#include <algorithm>

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
        1, size);
    wgc->sess = wgc->pool.CreateCaptureSession(wgc->item);
    wgc->sess.IsCursorCaptureEnabled(cfg.cap_cursor);
    wgc->sess.IsBorderRequired(cfg.cap_border);

    wgc->sess.StartCapture();
    return true;
}

void ScreenCapturer::rel_wgc() {
    if (wgc) {
        // session/pool will be released with wgc destruction
        delete wgc; wgc = nullptr;
    }
}

bool ScreenCapturer::cap_wgc(cv::Mat& frame, const cv::Rect& crop_region) {
    const bool do_tim = cfg.capture_timing && cfg.timing;
    using clock = std::chrono::high_resolution_clock;

    // 累计分段计时（多次尝试累加）
    double aq_sum = 0.0, qi_sum = 0.0, cp_sum = 0.0, mp_sum = 0.0, cv_sum = 0.0, up_sum = 0.0;
    auto t_total_start = do_tim ? clock::now() : clock::time_point{};

    const int tries = std::max(1, 1 + cfg.cap_retry);

    for (int attempt = 0; attempt < tries; ++attempt) {
        double aq = 0.0, qi = 0.0, cp = 0.0, mp = 0.0, cvc = 0.0, up = 0.0;

        // Acquire (TryGetNextFrame)
        auto t_acq_s = do_tim ? clock::now() : clock::time_point{};
        auto frm = wgc->pool.TryGetNextFrame();
        auto t_acq_e = do_tim ? clock::now() : t_acq_s;
        if (do_tim) aq = std::chrono::duration<double, std::milli>(t_acq_e - t_acq_s).count();

        if (!frm) {
            // 无帧，立即重试
            aq_sum += aq;
            continue;
        }

        auto surf = frm.Surface();
        auto tex = GetDXGIInterfaceFromObject<ID3D11Texture2D>(surf);

        // Copy
        D3D11_BOX src{}; src.left = crop_region.x; src.top = crop_region.y;
        src.right = crop_region.x + crop_region.width; src.bottom = crop_region.y + crop_region.height;
        src.front = 0; src.back = 1;

        auto t_copy_s = do_tim ? clock::now() : clock::time_point{};
        ctx->CopySubresourceRegion(stage[0], 0, 0, 0, 0, tex.get(), 0, &src);
        auto t_copy_e = do_tim ? clock::now() : t_copy_s;
        if (do_tim) cp = std::chrono::duration<double, std::milli>(t_copy_e - t_copy_s).count();

        if (cfg.capture_flush_after_copy) {
            ctx->Flush();
        }

        // Map (blocking)
        D3D11_MAPPED_SUBRESOURCE mr;
        auto t_map_s = do_tim ? clock::now() : clock::time_point{};
        HRESULT hrm = ctx->Map(stage[0], 0, D3D11_MAP_READ, 0, &mr);
        auto t_map_e = do_tim ? clock::now() : t_map_s;
        if (do_tim) mp = std::chrono::duration<double, std::milli>(t_map_e - t_map_s).count();
        if (FAILED(hrm)) {
            aq_sum += aq; cp_sum += cp; mp_sum += mp;
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

        aq_sum += aq; qi_sum += qi; cp_sum += cp; mp_sum += mp; cv_sum += cvc; up_sum += up;

        // 成功上报并返回
        double tt = do_tim ? std::chrono::duration<double, std::milli>(t_unmap_e - t_total_start).count() : 0.0;
        if (cfg.capture_timing && cfg.capture_timing_on_success) {
            reportTimes(aq_sum, qi_sum, cp_sum, mp_sum, cv_sum, up_sum, tt);
        }
        ++statOk;
        dbgTick();
        return true;
    }

    if (cfg.capture_timing && !cfg.capture_timing_on_success) {
        double tt = do_tim ? std::chrono::duration<double, std::milli>(clock::now() - t_total_start).count() : 0.0;
        reportTimes(aq_sum, qi_sum, cp_sum, mp_sum, cv_sum, up_sum, tt);
    }
    return false;
}

bool ScreenCapturer::cap_wgc_view(ScreenCapturer::CapturedView& view, const cv::Rect& crop_region) {
    const bool do_tim = cfg.capture_timing && cfg.timing;
    using clock = std::chrono::high_resolution_clock;

    view = CapturedView{};
    view.width = crop_region.width;
    view.height = crop_region.height;
    if (do_tim) view.t_total_start = clock::now();

    const int tries = std::max(1, 1 + cfg.cap_retry);

    for (int attempt = 0; attempt < tries; ++attempt) {
        double aq = 0.0, cp = 0.0, mp = 0.0;

        auto t_acq_s = do_tim ? clock::now() : clock::time_point{};
        auto frm = wgc->pool.TryGetNextFrame();
        auto t_acq_e = do_tim ? clock::now() : t_acq_s;
        if (do_tim) aq = std::chrono::duration<double, std::milli>(t_acq_e - t_acq_s).count();
        if (!frm) { view.aq_sum += aq; continue; }

        auto surf = frm.Surface();
        auto tex = GetDXGIInterfaceFromObject<ID3D11Texture2D>(surf);

        D3D11_BOX src{}; src.left = crop_region.x; src.top = crop_region.y;
        src.right = crop_region.x + crop_region.width; src.bottom = crop_region.y + crop_region.height;
        src.front = 0; src.back = 1;

        auto t_copy_s = do_tim ? clock::now() : clock::time_point{};
        ctx->CopySubresourceRegion(stage[0], 0, 0, 0, 0, tex.get(), 0, &src);
        auto t_copy_e = do_tim ? clock::now() : t_copy_s;
        if (do_tim) cp = std::chrono::duration<double, std::milli>(t_copy_e - t_copy_s).count();
        if (cfg.capture_flush_after_copy) ctx->Flush();

        D3D11_MAPPED_SUBRESOURCE mr;
        auto t_map_s = do_tim ? clock::now() : clock::time_point{};
        HRESULT hrm = ctx->Map(stage[0], 0, D3D11_MAP_READ, 0, &mr);
        auto t_map_e = do_tim ? clock::now() : t_map_s;
        if (do_tim) mp = std::chrono::duration<double, std::milli>(t_map_e - t_map_s).count();
        if (FAILED(hrm)) { view.aq_sum += aq; view.cp_sum += cp; view.mp_sum += mp; continue; }

        view.data = static_cast<unsigned char*>(mr.pData);
        view.pitch = mr.RowPitch;
        view.mapped = true;

        view.aq_sum += aq; view.cp_sum += cp; view.mp_sum += mp;
        ++statOk; dbgTick();
        return true;
    }

    if (cfg.capture_timing && !cfg.capture_timing_on_success) {
        double tt = do_tim ? std::chrono::duration<double, std::milli>(clock::now() - view.t_total_start).count() : 0.0;
        reportTimes(view.aq_sum, view.qi_sum, view.cp_sum, view.mp_sum, view.cv_sum, view.up_sum, tt);
    }
    return false;
}
