#include "DxgiTuning.hpp"
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

void TuneDxgi(ID3D11Device* dev, int gpu_prio, int max_latency) {
    if (!dev) return;
    IDXGIDevice1* dxgi1 = nullptr;
    HRESULT hr = dev->QueryInterface(__uuidof(IDXGIDevice1), (void**)&dxgi1);
    if (FAILED(hr) || !dxgi1) {
        std::cerr << "QueryInterface(IDXGIDevice1) failed: 0x" << std::hex << (unsigned long)hr << std::dec << std::endl;
        return;
    }
    hr = dxgi1->SetGPUThreadPriority(gpu_prio);
    if (FAILED(hr)) {
        std::cerr << "SetGPUThreadPriority(7) failed: 0x" << std::hex << (unsigned long)hr << std::dec << std::endl;
    }
    hr = dxgi1->SetMaximumFrameLatency(max_latency);
    if (FAILED(hr)) {
        std::cerr << "SetMaximumFrameLatency(1) failed: 0x" << std::hex << (unsigned long)hr << std::dec << std::endl;
    }
    dxgi1->Release();
}
