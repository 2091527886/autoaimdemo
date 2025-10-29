#include "SystemPriority.hpp"

void RaiseProcessPrioritySafely() {
    if (!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)) {
        std::cerr << "SetPriorityClass(HIGH) failed: " << GetLastError() << std::endl;
    }
}

void RaiseThisThreadPrioritySafely() {
    if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST)) {
        std::cerr << "SetThreadPriority(HIGHEST) failed: " << GetLastError() << std::endl;
    }
}

void SetThisThreadMemoryPriorityNormal(bool verbose) {
    MEMORY_PRIORITY_INFORMATION mpi{};
    mpi.MemoryPriority = MEMORY_PRIORITY_NORMAL;
    if (!SetThreadInformation(GetCurrentThread(), ThreadMemoryPriority, &mpi, sizeof(mpi))) {
        if (verbose) std::cerr << "[BOOT] ThreadMemoryPriority=NORMAL failed: " << GetLastError() << std::endl;
    } else if (verbose) {
        std::cout << "[BOOT] ThreadMemoryPriority=NORMAL OK" << std::endl;
    }
}

void DisableProcessPriorityBoost(bool verbose) {
    if (!SetProcessPriorityBoost(GetCurrentProcess(), FALSE)) {
        if (verbose) std::cerr << "[BOOT] ProcPriorityBoost(FALSE) failed: " << GetLastError() << std::endl;
    } else if (verbose) {
        std::cout << "[BOOT] ProcPriorityBoost(FALSE) OK" << std::endl;
    }
}
