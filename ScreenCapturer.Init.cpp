#include "ScreenCapturer.hpp"
#include "ComHelpers.hpp"
#include "Config.hpp"
#include "Log.hpp"
#include "Reporting.hpp"
#include "SystemTuner.hpp"
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
  if (!useWgc) { init_dxgi(); reporter.SetBackend(metrics::CaptureReporting::Backend::DXGI); }
  else { init_wgc(); reporter.SetBackend(metrics::CaptureReporting::Backend::WGC); }

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
