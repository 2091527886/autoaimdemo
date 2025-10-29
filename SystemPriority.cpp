#include "SystemPriority.hpp"

void SetProcPrioHigh() {
    if (!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)) {
        std::cerr << "SetPriorityClass(HIGH) failed: " << GetLastError() << std::endl;
    }
}

void SetThreadPrioHighest() {
    if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST)) {
        std::cerr << "SetThreadPriority(HIGHEST) failed: " << GetLastError() << std::endl;
    }
}

void SetThreadMemPrioNormal(bool log) {
    MEMORY_PRIORITY_INFORMATION memPri{};
    memPri.MemoryPriority = MEMORY_PRIORITY_NORMAL;
    if (!SetThreadInformation(GetCurrentThread(), ThreadMemoryPriority, &memPri, sizeof(memPri))) {
        if (log) std::cerr << "[BOOT] ThreadMemoryPriority=NORMAL failed: " << GetLastError() << std::endl;
    } else if (log) {
        std::cout << "[BOOT] ThreadMemoryPriority=NORMAL OK" << std::endl;
    }
}

void SetProcPrioBoost(bool enable, bool log) {
    if (!SetProcessPriorityBoost(GetCurrentProcess(), enable ? TRUE : FALSE)) {
        if (log) std::cerr << (enable ? "[BOOT] ProcPriorityBoost(TRUE) failed: " : "[BOOT] ProcPriorityBoost(FALSE) failed: ") << GetLastError() << std::endl;
    } else if (log) {
        std::cout << (enable ? "[BOOT] ProcPriorityBoost(TRUE) OK" : "[BOOT] ProcPriorityBoost(FALSE) OK") << std::endl;
    }
}

void SetProcPrioBoostDisabled(bool log) {
    SetProcPrioBoost(false, log);
}
