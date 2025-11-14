#include "Log.hpp"
#include "ScreenCapturer.hpp"
#include "pch.hpp"
#include "sh.hpp"

#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/base.h>
#include "sc_helpers.hpp"

namespace wg = winrt::Windows::Graphics; namespace wgd = wg::DirectX::Direct3D11; namespace wgc = wg::Capture;
using std::max;

namespace {
template <typename T>
winrt::com_ptr<T> GetDXGIInterfaceFromObject(winrt::Windows::Foundation::IInspectable const &object) {
  winrt::com_ptr<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess> access = object.as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
  winrt::com_ptr<T> result; winrt::check_hresult(access->GetInterface(winrt::guid_of<T>(), result.put_void()));
  return result;
}
wgd::IDirect3DDevice CreateDirect3DDevice(IDXGIDevice *dxgiDevice) {
  wgd::IDirect3DDevice device{nullptr};
  winrt::com_ptr<IInspectable> inspectable;
  winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice, reinterpret_cast<IInspectable **>(inspectable.put())));
  device = inspectable.as<wgd::IDirect3DDevice>();
  return device;
}
} // namespace

struct ScreenCapturer::WgcCtx {
  wgd::IDirect3DDevice wdev{nullptr};
  wgc::GraphicsCaptureItem item{nullptr};
  wgc::Direct3D11CaptureFramePool pool{nullptr};
  wgc::GraphicsCaptureSession sess{nullptr};
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
      winrt::get_activation_factory<wgc::GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
  wgc::GraphicsCaptureItem item{nullptr};
  winrt::check_hresult(interop->CreateForMonitor(od.Monitor, winrt::guid_of<wgc::GraphicsCaptureItem>(), winrt::put_abi(item)));
  wgc->item = item;

  auto size = winrt::Windows::Graphics::SizeInt32{width, height};
  wgc->pool = wgc::Direct3D11CaptureFramePool::Create(
      wgc->wdev,
      wg::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized,
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

  const int tries = max(1, 1 + cfg.cap_retry);

  if (try_n(tries, [&](){
        double aq=0.0, cp=0.0, mp=0.0;
        TSTART(t_acq_s, do_tim);
        auto frm = wgc->pool.TryGetNextFrame();
        TELAPSE(aq, do_tim, t_acq_s);
        if (!frm) {
          view.aq_sum += aq;
          return false;
        }

        auto surf = frm.Surface();
        auto tex = GetDXGIInterfaceFromObject<ID3D11Texture2D>(surf);

        D3D11_BOX src = box_from_rect(crop_region);

        auto copy_and_map = [&](ID3D11Texture2D *srcTex)->bool{
          TSTART(t_copy_s, do_tim);
          ctx->CopySubresourceRegion(stage[0], 0, 0, 0, 0, srcTex, 0, &src);
          TELAPSE(cp, do_tim, t_copy_s);
          if (cfg.capture_flush_after_copy) ctx->Flush();

          if(!map_stage(ctx, stage[0], view, do_tim, mp)) return false;
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
