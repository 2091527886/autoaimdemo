#pragma once
#include <d3d11.h>
#include <dxgi1_2.h>
#include <iostream>

// Raise DXGI GPU thread priority and set frame latency (migrated)
void TuneDxgi(ID3D11Device* dev, int gpu_prio = 7, int max_latency = 1);
