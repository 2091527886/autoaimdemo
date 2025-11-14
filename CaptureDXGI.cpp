#include "ScreenCapturer.hpp"
#include "Config.hpp"
#include "Log.hpp"
#include "ComHelpers.hpp"

#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <chrono>
#include <stdexcept>
#include <algorithm>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

bool ScreenCapturer::init_dxgi() {
    if (FAILED(out->QueryInterface(__uuidof(IDXGIOutput1), (void**)&out1))) {
        throw std::runtime_error("Failed to query IDXGIOutput1.");
    }
    HRESULT hr = out1->DuplicateOutput(dev, &dup);
    if (FAILED(hr)) {
        lg::ok_or_hrfail_if(true, P_C, "DuplicateOutput", hr, SUCCEEDED(hr));
        throw std::runtime_error("Failed to create output duplication.");
    }
    return true;
}

