#include "ThreadAffinity.hpp"

uint64_t BuildMaskFromCores(const std::vector<unsigned>& cores) {
    uint64_t m = 0;
    for (unsigned c : cores) if (c < 64) m |= (1ull << c);
    return m;
}

bool ApplyThreadAffinityMask(uint64_t desiredMask, bool verbose) {
    if (!desiredMask) return false;
    DWORD_PTR procMask=0, sysMask=0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &procMask, &sysMask)) {
        if (verbose) std::cerr << "[AFF] GetProcessAffinityMask failed: " << GetLastError() << std::endl;
        return false;
    }
    DWORD_PTR finalMask = (DWORD_PTR)(desiredMask & (uint64_t)procMask);
    if (!finalMask) {
        if (verbose) std::cerr << "[AFF] empty after intersect with process mask" << std::endl;
        return false;
    }
    if (!SetThreadAffinityMask(GetCurrentThread(), finalMask)) {
        if (verbose) std::cerr << "[AFF] SetThreadAffinityMask failed: " << GetLastError() << std::endl;
        return false;
    }
    if (verbose) std::cout << "[AFF] Thread pinned. mask=0x" << std::hex << (uint64_t)finalMask << std::dec << std::endl;
    return true;
}
