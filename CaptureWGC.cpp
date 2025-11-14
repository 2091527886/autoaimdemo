#include "Config.hpp"
#include "Log.hpp"
#include "ScreenCapturer.hpp"
#include "pch.hpp"
#include <chrono>

using clk = std::chrono::high_resolution_clock;
using tp = clk::time_point;
using msd = std::chrono::duration<double, std::milli>;
inline tp tn() { return clk::now(); }

#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/base.h>

namespace {
template <typename T>
winrt::com_ptr<T> GetDXGIInterfaceFromObject(winrt::Windows::Foundation::IInspectable const &object) {
  winrt::com_ptr<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess> access = object.as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
  winrt::com_ptr<T> result;
  winrt::check_hresult(access->GetInterface(winrt::guid_of<T>(), result.put_void()));
  return result;
}
winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice CreateDirect3DDevice(IDXGIDevice *dxgiDevice) {
  winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice device{nullptr};
  winrt::com_ptr<IInspectable> inspectable;
  winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice, reinterpret_cast<IInspectable **>(inspectable.put())));
  device = inspectable.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();
  return device;
}
} // namespace

struct ScreenCapturer::WgcCtx {
  winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice wdev{nullptr};
  winrt::Windows::Graphics::Capture::GraphicsCaptureItem item{nullptr};
  winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool pool{nullptr};
  winrt::Windows::Graphics::Capture::GraphicsCaptureSession sess{nullptr};
};

bool ScreenCapturer::init_wgc() {
  winrt::init_apartment(winrt::apartment_type::multi_threaded);

  DXGI_OUTPUT_DESC od;
  out->GetDesc(&od);

  wgc = new WgcCtx();

  winrt::com_ptr<IDXGIDevice> dxgiDev;
  dev->QueryInterface(__uuidof(IDXGIDevice), dxgiDev.put_void());
  wgc->wdev = CreateDirect3DDevice(dxgiDev.get());

  winrt::com_ptr<IGraphicsCaptureItemInterop> interop =
      winrt::get_activation_factory<winrt::Windows::Graphics::Capture::GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
  winrt::Windows::Graphics::Capture::GraphicsCaptureItem item{nullptr};
  winrt::check_hresult(interop->CreateForMonitor(od.Monitor, winrt::guid_of<winrt::Windows::Graphics::Capture::GraphicsCaptureItem>(), winrt::put_abi(item)));
  wgc->item = item;

  auto size = winrt::Windows::Graphics::SizeInt32{width, height};
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
    delete wgc;
    wgc = nullptr;
  }
}

bool ScreenCapturer::cap_wgc_view(ScreenCapturer::CapturedView &view, const cv::Rect &crop_region) {
  const bool do_tim = cfg.capture_timing && cfg.timing;

  view = CapturedView{};
  view.width = crop_region.width;
  view.height = crop_region.height;
  if (do_tim)
    view.t_total_start = tn();

  const int tries = std::max(1, 1 + cfg.cap_retry);

  auto try_loop = [&](auto &&body) -> bool {
    for (int attempt = 0; attempt < tries; ++attempt)
      if (body())
        return true;
    return false;
  };
  if (try_loop([&]() {
        double aq = 0.0, cp = 0.0, mp = 0.0;
        auto t_acq_s = do_tim ? tn() : tp{};
        auto frm = wgc->pool.TryGetNextFrame();
        auto t_acq_e = do_tim ? tn() : t_acq_s;
        if (do_tim)
          aq = msd(t_acq_e - t_acq_s).count();
        if (!frm) {
          view.aq_sum += aq;
          return false;
        }

        auto surf = frm.Surface();
        auto tex = GetDXGIInterfaceFromObject<ID3D11Texture2D>(surf);

        D3D11_BOX src{};
        src.left = crop_region.x;
        src.top = crop_region.y;
        src.right = crop_region.x + crop_region.width;
        src.bottom = crop_region.y + crop_region.height;
        src.front = 0;
        src.back = 1;

        auto copy_and_map = [&](ID3D11Texture2D *srcTex) -> bool {
          auto t_copy_s = do_tim ? tn() : tp{};
          ctx->CopySubresourceRegion(stage[0], 0, 0, 0, 0, srcTex, 0, &src);
          auto t_copy_e = do_tim ? tn() : t_copy_s;
          if (do_tim)
            cp = msd(t_copy_e - t_copy_s).count();
          if (cfg.capture_flush_after_copy)
            ctx->Flush();

          D3D11_MAPPED_SUBRESOURCE mr;
          auto t_map_s = do_tim ? tn() : tp{};
          HRESULT hrm = ctx->Map(stage[0], 0, D3D11_MAP_READ, 0, &mr);
          auto t_map_e = do_tim ? tn() : t_map_s;
          if (do_tim)
            mp = msd(t_map_e - t_map_s).count();
          if (FAILED(hrm))
            return false;

          view.data = static_cast<unsigned char *>(mr.pData);
          view.pitch = mr.RowPitch;
          view.mapped = true;
          return true;
        };

        if (!copy_and_map(tex.get())) {
          view.aq_sum += aq;
          view.cp_sum += cp;
          view.mp_sum += mp;
          return false;
        }

        view.aq_sum += aq;
        view.cp_sum += cp;
        view.mp_sum += mp;
        ++statOk;
        dbgTick();
        return true;
      })) {
    return true;
  }

  if (cfg.capture_timing && !cfg.capture_timing_on_success) {
    double tt = do_tim ? msd(tn() - view.t_total_start).count() : 0.0;
    reportTimes(view.aq_sum, view.qi_sum, view.cp_sum, view.mp_sum, view.cv_sum, view.up_sum, tt);
  }
  return false;
}
