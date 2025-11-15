#include "Log.hpp"
#include "ScreenCapturer.hpp"
#include "pch.hpp"
bool ScreenCapturer::init_dxgi() {
  if (FAILED(out->QueryInterface(__uuidof(IDXGIOutput1), (void **)&out1))) {
    throw std::runtime_error("Failed to query IDXGIOutput1.");
  }
  HRESULT hr = out1->DuplicateOutput(dev, &dup);
  if (FAILED(hr)){ lg::hrfail(P_C, "DuplicateOutput", hr); throw std::runtime_error("Failed to create output duplication."); }
  else { lg::o() << P_C "DuplicateOutput OK" << lg::endl; }
  return true;
}
