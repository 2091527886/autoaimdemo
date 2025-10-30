#pragma once

// Unified system scheduling & runtime optimization facade
// NOTE: Logic unchanged from original modules. Only consolidated.

#include <windows.h>
#include <avrt.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <cstdint>
#include "Log.hpp"

// ---- Process/Thread Priority (from SystemPriority) ----
void SetProcPrioHigh();
void SetThreadPrioHighest();
void SetThreadMemPrioNormal(bool log = true);
void SetProcPrioBoost(bool enable, bool log = true);
void SetProcPrioBoostDisabled(bool log = true);

// ---- Thread Affinity (from ThreadAffinity) ----
bool setAffinity(uint64_t mask, bool log = true);

// ---- Power Throttling (from PowerThrottling) ----
void disablePowerThrottle(bool log = true);

// ---- MMCSS (from MmcssUtil) ----
HANDLE MmcssEnable(bool enable, const wchar_t* task, AVRT_PRIORITY prio, DWORD* outIdx = nullptr);
void MmcssDisable(HANDLE h);

// ---- DXGI tuning (from DxgiTuning) ----
void TuneDxgi(ID3D11Device* dev, int gpu_prio = 7, int max_latency = 1);
