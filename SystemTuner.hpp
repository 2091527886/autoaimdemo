#pragma once

// Unified system scheduling & runtime optimization facade
#include "pch.hpp"
#include <cstdint>
#include "Log.hpp"

struct ID3D11Device;

// Priority
void SetProcPrioHigh();
void SetThreadPrioHighest();
void SetThreadMemPrioNormal(bool log = true);
void SetProcPrioBoost(bool enable, bool log = true);
void SetProcPrioBoostDisabled(bool log = true);

// Affinity
// Power throttle
void disablePowerThrottle(bool log = true);

// MMCSS
// DXGI tuning
void TuneDxgi(ID3D11Device* dev, int gpu_prio = 7, int max_latency = 1);
