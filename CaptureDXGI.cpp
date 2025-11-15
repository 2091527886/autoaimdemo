#include "Log.hpp"
#include "ScreenCapturer.hpp"
#include "pch.hpp"
bool ScreenCapturer::init_dxgi() {
  HR_THROW(out->QueryInterface(__uuidof(IDXGIOutput1), (void **)&out1), P_C, "QueryInterface(IDXGIOutput1)", throw std::runtime_error("Failed to query IDXGIOutput1."); );
  HR_THROW(out1->DuplicateOutput(dev, &dup), P_C, "DuplicateOutput", throw std::runtime_error("Failed to create output duplication."); );
  LO << P_C "DuplicateOutput OK" << END;
  return true;
}
