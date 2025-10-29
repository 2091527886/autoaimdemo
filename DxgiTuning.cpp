#include "DxgiTuning.hpp"
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

void RaiseDxgiGpuPriorityAndLatency(ID3D11Device* d3dDevice, int gpu_thread_priority, int max_frame_latency) {
    if (!d3dDevice) return;
    IDXGIDevice1* dxgiDevice1 = nullptr;
    HRESULT hr = d3dDevice->QueryInterface(__uuidof(IDXGIDevice1), (void**)&dxgiDevice1);
    if (FAILED(hr) || !dxgiDevice1) {
        std::cerr << "QueryInterface(IDXGIDevice1) failed: 0x" << std::hex << (unsigned long)hr << std::dec << std::endl;
        return;
    }
    hr = dxgiDevice1->SetGPUThreadPriority(gpu_thread_priority);
    if (FAILED(hr)) {
        std::cerr << "SetGPUThreadPriority(7) failed: 0x" << std::hex << (unsigned long)hr << std::dec << std::endl;
    }
    hr = dxgiDevice1->SetMaximumFrameLatency(max_frame_latency);
    if (FAILED(hr)) {
        std::cerr << "SetMaximumFrameLatency(1) failed: 0x" << std::hex << (unsigned long)hr << std::dec << std::endl;
    }
    dxgiDevice1->Release();
}
