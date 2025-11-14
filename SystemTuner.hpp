#pragma once

// Unified system scheduling & runtime optimization facade
#include <windows.h>
#include <avrt.h>
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
bool setAffinity(uint64_t mask, bool log = true);

// Power throttle
void disablePowerThrottle(bool log = true);

// MMCSS
HANDLE MmcssEnable(bool enable, const wchar_t* task, AVRT_PRIORITY prio, DWORD* outIdx = nullptr);
void MmcssDisable(HANDLE h);

// DXGI tuning
void TuneDxgi(ID3D11Device* dev, int gpu_prio = 7, int max_latency = 1);
