#pragma once
#include <d3d11.h>
#include <dxgi1_2.h>
#include <iostream>

// Raise DXGI GPU thread priority and set frame latency (migrated)
void RaiseDxgiGpuPriorityAndLatency(ID3D11Device* d3dDevice, int gpu_thread_priority = 7, int max_frame_latency = 1);
