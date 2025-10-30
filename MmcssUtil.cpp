#include "MmcssUtil.hpp"
#include "Log.hpp"
#pragma comment(lib, "Avrt.lib")

HANDLE MmcssEnable(bool enable, const wchar_t* task, AVRT_PRIORITY prio, DWORD* outIdx) {
    if (!enable) return nullptr;
    if (!task) task = L"Games";
    DWORD idx = 0;
    HANDLE h = AvSetMmThreadCharacteristicsW(task, &idx);
    if (!h) {
        lg::e() << "[MMCSS] AvSetMmThreadCharacteristicsW failed: " << GetLastError() << std::endl;
        return nullptr;
    }
    if (!AvSetMmThreadPriority(h, prio)) {
        lg::e() << "[MMCSS] AvSetMmThreadPriority failed: " << GetLastError() << std::endl;
    }
    if (outIdx) *outIdx = idx;
    lg::wo() << L"[MMCSS] Enabled for thread. Task=" << task << L", priority=" << (int)prio << std::endl;
    return h;
}

void MmcssDisable(HANDLE h) {
    if (h) AvRevertMmThreadCharacteristics(h);
}
