#include "pch.hpp"
#include "SystemTuner.hpp"

void SetProcPrioHigh() {
    BOOL ok = SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    lg::ok_or_werr(P_B, "ProcPrio(HIGH)", ok);
}

void SetThreadPrioHighest() {
    BOOL ok = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    lg::ok_or_werr(P_B, "ThreadPrio(HIGHEST)", ok);
}

void SetThreadMemPrioNormal(bool log) {
    MEMORY_PRIORITY_INFORMATION memPri{};
    memPri.MemoryPriority = MEMORY_PRIORITY_NORMAL;
    bool ok = SetThreadInformation(GetCurrentThread(), ThreadMemoryPriority, &memPri, sizeof(memPri)) ? true : false;
    lg::ok_or_werr_if(log, P_B, "ThreadMemPrio=NORMAL", ok);
}

void SetProcPrioBoost(bool enable, bool log) {
    BOOL ok = SetProcessPriorityBoost(GetCurrentProcess(), enable ? TRUE : FALSE);
    lg::ok_or_werr_if(log, P_B, enable ? "ProcPrioBoost(TRUE)" : "ProcPrioBoost(FALSE)", ok);
}

void SetProcPrioBoostDisabled(bool log) {
    SetProcPrioBoost(false, log);
}

bool setAffinity(uint64_t mask, bool log) {
    if (!mask) return false;
    DWORD_PTR pMask=0, sMask=0;
    {
        BOOL ok = GetProcessAffinityMask(GetCurrentProcess(), &pMask, &sMask);
        lg::ok_or_werr_if(log, P_A, "GetProcessAffinityMask", ok);
        if (!ok) return false;
    }
    DWORD_PTR fMask = (DWORD_PTR)(mask & (uint64_t)pMask);
    if (!fMask) {
        LOG_IF(log, lg::e() << P_A "empty after intersect with process mask" << lg::endl);
        return false;
    }
    {
        DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), fMask);
        BOOL ok = prev != 0;
        lg::ok_or_werr_if(log, P_A, "SetThreadAffinityMask", ok);
        if (!ok) return false;
    }
    LOG_IF(log, lg::o() << P_A "Thread pinned. mask=0x" << lg::hexu << (uint64_t)fMask << lg::deff << lg::endl);
    return true;
}

void disablePowerThrottle(bool log) {
   PROCESS_POWER_THROTTLING_STATE pState = {};
   pState.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
   pState.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
   pState.StateMask = 0;
   {
       BOOL ok = SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &pState, sizeof(pState));
       lg::ok_or_werr_if(log, P_B, "ProcPowerThrottling=OFF", ok);
   }

   THREAD_POWER_THROTTLING_STATE tState = {};
   tState.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
   tState.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
   tState.StateMask = 0;
   {
       BOOL ok = SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &tState, sizeof(tState));
       lg::ok_or_werr_if(log, P_B, "ThreadPowerThrottling=OFF", ok);
   }

   {
       BOOL ok = SetThreadPriorityBoost(GetCurrentThread(), FALSE);
       lg::ok_or_werr_if(log, P_B, "ThreadPriorityBoost(FALSE)", ok);
   }
}

HANDLE MmcssEnable(bool enable, const wchar_t* task, AVRT_PRIORITY prio, DWORD* outIdx) {
    if (!enable) return nullptr;
    if (!task) task = L"Games";
    DWORD idx = 0;
    HANDLE h = AvSetMmThreadCharacteristicsW(task, &idx);
    if (!h) {
        lg::ok_or_werr(P_MS, "AvSetMmThreadCharacteristicsW", false);
        return nullptr;
    }
    {
        BOOL ok = AvSetMmThreadPriority(h, prio);
        lg::ok_or_werr(P_MS, "AvSetMmThreadPriority", ok);
    }
    if (outIdx) *outIdx = idx;
    lg::wo() << WP_MS L"Enabled for thread. Task=" << task << L", priority=" << (int)prio << lg::wendl;
    return h;
}

void MmcssDisable(HANDLE h) {
    if (h) AvRevertMmThreadCharacteristics(h);
}

void TuneDxgi(ID3D11Device* dev, int gpu_prio, int max_latency) {
    if (!dev) return;
    IDXGIDevice1* dxgi1 = nullptr;
    HRESULT hr = dev->QueryInterface(__uuidof(IDXGIDevice1), (void**)&dxgi1);
    if (FAILED(hr) || !dxgi1) {
        lg::hrfail(P_E, "QueryInterface(IDXGIDevice1)", hr);
        return;
    }
    hr = dxgi1->SetGPUThreadPriority(gpu_prio);
    if (FAILED(hr)) {
        lg::hrfail(P_E, "SetGPUThreadPriority(7)", hr);
    }
    hr = dxgi1->SetMaximumFrameLatency(max_latency);
    if (FAILED(hr)) {
        lg::hrfail(P_E, "SetMaximumFrameLatency(1)", hr);
    }
    dxgi1->Release();
}
