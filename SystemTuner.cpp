#include "SystemTuner.hpp"

#pragma comment(lib, "Avrt.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

// ---- SystemPriority.cpp ----
void SetProcPrioHigh() {
    if (!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)) {
        lg::e() << "SetPriorityClass(HIGH) failed: " << lg::winerr() << std::endl;
    }
}

void SetThreadPrioHighest() {
    if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST)) {
        lg::e() << "SetThreadPriority(HIGHEST) failed: " << lg::winerr() << std::endl;
    }
}

void SetThreadMemPrioNormal(bool log) {
    MEMORY_PRIORITY_INFORMATION memPri{};
    memPri.MemoryPriority = MEMORY_PRIORITY_NORMAL;
    if (!SetThreadInformation(GetCurrentThread(), ThreadMemoryPriority, &memPri, sizeof(memPri))) {
        LOG_IF(log, lg::e() << LOGP_BOOT "ThreadMemoryPriority=NORMAL failed: " << lg::winerr() << std::endl);
    } else if (log) {
        lg::o() << "[BOOT] ThreadMemoryPriority=NORMAL OK" << std::endl;
    }
}

void SetProcPrioBoost(bool enable, bool log) {
    if (!SetProcessPriorityBoost(GetCurrentProcess(), enable ? TRUE : FALSE)) {
        LOG_IF(log, lg::e() << (enable ? "[BOOT] ProcPriorityBoost(TRUE) failed: " : "[BOOT] ProcPriorityBoost(FALSE) failed: ") << lg::winerr() << std::endl);
    } else if (log) {
        lg::o() << (enable ? "[BOOT] ProcPriorityBoost(TRUE) OK" : "[BOOT] ProcPriorityBoost(FALSE) OK") << std::endl;
    }
}

void SetProcPrioBoostDisabled(bool log) {
    SetProcPrioBoost(false, log);
}

// ---- ThreadAffinity.cpp ----
bool setAffinity(uint64_t mask, bool log) {
    if (!mask) return false;
    DWORD_PTR pMask=0, sMask=0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &pMask, &sMask)) {
        LOG_IF(log, lg::e() << LOGP_AFF "GetProcessAffinityMask failed: " << lg::winerr() << std::endl);
        return false;
    }
    DWORD_PTR fMask = (DWORD_PTR)(mask & (uint64_t)pMask);
    if (!fMask) {
        LOG_IF(log, lg::e() << LOGP_AFF "empty after intersect with process mask" << std::endl);
        return false;
    }
    if (!SetThreadAffinityMask(GetCurrentThread(), fMask)) {
        LOG_IF(log, lg::e() << LOGP_AFF "SetThreadAffinityMask failed: " << lg::winerr() << std::endl);
        return false;
    }
    LOG_IF(log, lg::o() << "[AFF] Thread pinned. mask=0x" << lg::hexu << (uint64_t)fMask << lg::deff << std::endl);
    return true;
}

// ---- PowerThrottling.cpp ----
void disablePowerThrottle(bool log) {
   PROCESS_POWER_THROTTLING_STATE pState = {};
   pState.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
   pState.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
   pState.StateMask = 0;
   if (!SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &pState, sizeof(pState))) {
       LOG_IF(log, lg::e() << LOGP_BOOT "ProcPowerThrottling=OFF failed: " << lg::winerr() << std::endl);
   } else if (log) {
       lg::o() << "[BOOT] ProcPowerThrottling=OFF OK" << std::endl;
   }

   THREAD_POWER_THROTTLING_STATE tState = {};
   tState.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
   tState.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
   tState.StateMask = 0;
   if (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &tState, sizeof(tState))) {
       LOG_IF(log, lg::e() << LOGP_BOOT "ThreadPowerThrottling=OFF failed: " << lg::winerr() << std::endl);
   } else if (log) {
       lg::o() << "[BOOT] ThreadPowerThrottling=OFF OK" << std::endl;
   }

   if (!SetThreadPriorityBoost(GetCurrentThread(), FALSE)) {
       LOG_IF(log, lg::e() << LOGP_BOOT "ThreadPriorityBoost(FALSE) failed: " << lg::winerr() << std::endl);
   } else if (log) {
       lg::o() << "[BOOT] ThreadPriorityBoost(FALSE) OK" << std::endl;
   }
}

// ---- MmcssUtil.cpp ----
HANDLE MmcssEnable(bool enable, const wchar_t* task, AVRT_PRIORITY prio, DWORD* outIdx) {
    if (!enable) return nullptr;
    if (!task) task = L"Games";
    DWORD idx = 0;
    HANDLE h = AvSetMmThreadCharacteristicsW(task, &idx);
    if (!h) {
        lg::e() << "[MMCSS] AvSetMmThreadCharacteristicsW failed: " << lg::winerr() << std::endl;
        return nullptr;
    }
    if (!AvSetMmThreadPriority(h, prio)) {
        lg::e() << "[MMCSS] AvSetMmThreadPriority failed: " << lg::winerr() << std::endl;
    }
    if (outIdx) *outIdx = idx;
    lg::wo() << L"[MMCSS] Enabled for thread. Task=" << task << L", priority=" << (int)prio << std::endl;
    return h;
}

void MmcssDisable(HANDLE h) {
    if (h) AvRevertMmThreadCharacteristics(h);
}

// ---- DxgiTuning.cpp ----
void TuneDxgi(ID3D11Device* dev, int gpu_prio, int max_latency) {
    if (!dev) return;
    IDXGIDevice1* dxgi1 = nullptr;
    HRESULT hr = dev->QueryInterface(__uuidof(IDXGIDevice1), (void**)&dxgi1);
    if (FAILED(hr) || !dxgi1) {
        lg::e() << "QueryInterface(IDXGIDevice1) failed: " << lg::hrx(hr) << std::endl;
        return;
    }
    hr = dxgi1->SetGPUThreadPriority(gpu_prio);
    if (FAILED(hr)) {
        lg::e() << "SetGPUThreadPriority(7) failed: " << lg::hrx(hr) << std::endl;
    }
    hr = dxgi1->SetMaximumFrameLatency(max_latency);
    if (FAILED(hr)) {
        lg::e() << "SetMaximumFrameLatency(1) failed: " << lg::hrx(hr) << std::endl;
    }
    dxgi1->Release();
}
