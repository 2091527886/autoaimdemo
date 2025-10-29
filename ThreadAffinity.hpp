#pragma once
#include <windows.h>
#include <vector>
#include <cstdint>
#include <iostream>

// Affinity helpers (migrated from Config.hpp)
uint64_t BuildMaskFromCores(const std::vector<unsigned>& cores);
bool ApplyThreadAffinityMask(uint64_t desiredMask, bool verbose=true);
