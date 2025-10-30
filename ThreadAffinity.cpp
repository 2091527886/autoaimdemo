#include "ThreadAffinity.hpp"

uint64_t coreMask(const std::vector<unsigned>& cores) {
    uint64_t mask = 0;
    for (unsigned core : cores) if (core < 64) mask |= (1ull << core);
    return mask;
}

bool setAffinity(uint64_t mask, bool log) {
    if (!mask) return false;
    DWORD_PTR pMask=0, sMask=0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &pMask, &sMask)) {
        if (log) std::cerr << "[AFF] GetProcessAffinityMask failed: " << GetLastError() << std::endl;
        return false;
    }
    DWORD_PTR fMask = (DWORD_PTR)(mask & (uint64_t)pMask);
    if (!fMask) {
        if (log) std::cerr << "[AFF] empty after intersect with process mask" << std::endl;
        return false;
    }
    if (!SetThreadAffinityMask(GetCurrentThread(), fMask)) {
        if (log) std::cerr << "[AFF] SetThreadAffinityMask failed: " << GetLastError() << std::endl;
        return false;
    }
    if (log) std::cout << "[AFF] Thread pinned. mask=0x" << std::hex << (uint64_t)fMask << std::dec << std::endl;
    return true;
}
