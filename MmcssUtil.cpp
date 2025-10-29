#include "MmcssUtil.hpp"
#pragma comment(lib, "Avrt.lib")

HANDLE EnableMmcssIfConfigured(bool enable, const wchar_t* task_name, AVRT_PRIORITY priority, DWORD* out_task_index) {
    if (!enable) return nullptr;
    if (!task_name) task_name = L"Games";
    DWORD task_index = 0;
    HANDLE h = AvSetMmThreadCharacteristicsW(task_name, &task_index);
    if (!h) {
        std::cerr << "[MMCSS] AvSetMmThreadCharacteristicsW failed: " << GetLastError() << std::endl;
        return nullptr;
    }
    if (!AvSetMmThreadPriority(h, priority)) {
        std::cerr << "[MMCSS] AvSetMmThreadPriority failed: " << GetLastError() << std::endl;
    }
    if (out_task_index) *out_task_index = task_index;
    std::wcout << L"[MMCSS] Enabled for thread. Task=" << task_name << L", priority=" << (int)priority << std::endl;
    return h;
}

void DisableMmcss(HANDLE h) {
    if (h) AvRevertMmThreadCharacteristics(h);
}
