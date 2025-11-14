#include "ScreenCapturer.hpp"
#include "ComHelpers.hpp"
#include "Config.hpp"
#include "Log.hpp"
#include "Reporting.hpp"
#include "SystemTuner.hpp"
#include "pch.hpp"
#include <chrono>

// shorthands
using clk = std::chrono::high_resolution_clock;
using tp = clk::time_point;
using msd = std::chrono::duration<double, std::milli>;
inline tp tn() { return clk::now(); }

ScreenCapturer::ScreenCapturer(const Config &cfg_ref, int crop_width, int crop_height)
    : cfg(cfg_ref), reporter(cfg_ref) {
  HRESULT hr;

  hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **)&fac);
  if (FAILED(hr))
    throw std::runtime_error("Failed to create DXGI Factory.");

  UINT aidx = (cfg.cap_gpu >= 0) ? static_cast<UINT>(cfg.cap_gpu) : 0u;
  if (FAILED(fac->EnumAdapters1(aidx, &adp)))
    throw std::runtime_error("Failed to enumerate adapters.");

  if (cfg.cap_enum) {
    lg::o() << P_C "Adapters:" << lg::endl;
    for (UINT i = 0;; ++i) {
      IDXGIAdapter1 *a = nullptr;
      if (fac->EnumAdapters1(i, &a) != S_OK)
        break;
      DXGI_ADAPTER_DESC1 d;
      a->GetDesc1(&d);
      lg::o() << P_C "[" << i << "]" << lg::endl;
      a->Release();
    }
    lg::o() << P_C "Outputs on adapter " << aidx << ":" << lg::endl;
    for (UINT i = 0;; ++i) {
      IDXGIOutput *o = nullptr;
      if (adp->EnumOutputs(i, &o) != S_OK)
        break;
      DXGI_OUTPUT_DESC od;
      o->GetDesc(&od);
      RECT r = od.DesktopCoordinates;
      lg::o() << P_C "[" << i << "] " << r.right - r.left << "x" << r.bottom - r.top << lg::endl;
      o->Release();
    }
  }

  if (FAILED(adp->EnumOutputs(cfg.cap_out, &out)))
    throw std::runtime_error("Failed to enumerate outputs.");

  DXGI_OUTPUT_DESC outputDesc;
  out->GetDesc(&outputDesc);
  width = outputDesc.DesktopCoordinates.right - outputDesc.DesktopCoordinates.left;
  height = outputDesc.DesktopCoordinates.bottom - outputDesc.DesktopCoordinates.top;

  if (FAILED(D3D11CreateDevice(adp, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &dev, nullptr, &ctx)))
    throw std::runtime_error("Failed to create D3D11 device.");
  TuneDxgi(dev, cfg.dxgi_gpu_prio, cfg.dxgi_max_latency);

  useWgc = (cfg.cap == 1);
  if (!useWgc) {
    init_dxgi();
    reporter.SetBackend(metrics::CaptureReporting::Backend::DXGI);
  } else {
    init_wgc();
    reporter.SetBackend(metrics::CaptureReporting::Backend::WGC);
  }

  D3D11_TEXTURE2D_DESC desc;
  ZeroMemory(&desc, sizeof(desc));
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

  for (size_t i = 0; i < kStageN; ++i) {
    HRESULT hr_local = dev->CreateTexture2D(&desc, NULL, &stage[i]);
    if (FAILED(hr_local))
      throw std::runtime_error("Failed to create staging texture[" + std::to_string(i) + "]");
  }

  LOG_IF(cfg.log_init_verbose, lg::o() << P_C "Ready " << (useWgc ? "WGC " : "DDA ") << width << "x" << height << ", staging=" << kStageN << lg::endl);
  reporter.BootBanner(kStageN);
}

ScreenCapturer::~ScreenCapturer() {
  for (size_t i = 0; i < kStageN; ++i) {
    SafeRelease(&stage[i]);
  }
  if (useWgc) {
    rel_wgc();
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
  if (!cfg.capture_debug)
    return;
  n++;
  int m = std::max(1, cfg.capture_debug_every_n);
  if (n % static_cast<size_t>(m) == 0) {
    lg::o() << P_C "dbg ds=" << statSkipDirty
            << " ew=" << statCopyPend
            << " sd=" << statStillDraw
            << " ok=" << statOk << lg::endl;
  }
}

void ScreenCapturer::release_view(ScreenCapturer::CapturedView &view) {
  const bool do_tim = cfg.capture_timing && cfg.timing;
  using clock = std::chrono::high_resolution_clock;

  if (!view.mapped)
    return;
  auto t_unmap_s = do_tim ? tn() : tp{};
  ctx->Unmap(stage[0], 0);
  auto t_unmap_e = do_tim ? tn() : t_unmap_s;
  double up = do_tim ? msd(t_unmap_e - t_unmap_s).count() : 0.0;
  view.up_sum += up;

  if (cfg.capture_timing && cfg.capture_timing_on_success) {
    double tt = do_tim ? msd(t_unmap_e - view.t_total_start).count() : 0.0;
    reportTimes(view.aq_sum, view.qi_sum, view.cp_sum, view.mp_sum, view.cv_sum, view.up_sum, tt);
  }

  view.mapped = false;
  view.data = nullptr;
}

void ScreenCapturer::reportTimes(double t_acquire_ms, double t_qi_ms, double t_copy_ms,
                                 double t_map_ms, double t_cvt_ms, double t_unmap_ms, double t_total_ms) {
  reporter.ReportStepTimings(
      t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms,
      statSkipDirty, statCopyPend, statStillDraw, statOk);
}
