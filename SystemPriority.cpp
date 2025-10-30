#include "SystemPriority.hpp"
#include "Log.hpp"

void SetProcPrioHigh() {
    if (!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)) {
        lg::e() << "SetPriorityClass(HIGH) failed: " << GetLastError() << std::endl;
    }
}

void SetThreadPrioHighest() {
    if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST)) {
        lg::e() << "SetThreadPriority(HIGHEST) failed: " << GetLastError() << std::endl;
    }
}

void SetThreadMemPrioNormal(bool log) {
    MEMORY_PRIORITY_INFORMATION memPri{};
    memPri.MemoryPriority = MEMORY_PRIORITY_NORMAL;
    if (!SetThreadInformation(GetCurrentThread(), ThreadMemoryPriority, &memPri, sizeof(memPri))) {
        if (log) lg::e() << "[BOOT] ThreadMemoryPriority=NORMAL failed: " << GetLastError() << std::endl;
    } else if (log) {
        lg::o() << "[BOOT] ThreadMemoryPriority=NORMAL OK" << std::endl;
    }
}

void SetProcPrioBoost(bool enable, bool log) {
    if (!SetProcessPriorityBoost(GetCurrentProcess(), enable ? TRUE : FALSE)) {
        if (log) lg::e() << (enable ? "[BOOT] ProcPriorityBoost(TRUE) failed: " : "[BOOT] ProcPriorityBoost(FALSE) failed: ") << GetLastError() << std::endl;
    } else if (log) {
        lg::o() << (enable ? "[BOOT] ProcPriorityBoost(TRUE) OK" : "[BOOT] ProcPriorityBoost(FALSE) OK") << std::endl;
    }
}

void SetProcPrioBoostDisabled(bool log) {
    SetProcPrioBoost(false, log);
}
