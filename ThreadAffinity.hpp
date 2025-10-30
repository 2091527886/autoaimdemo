#pragma once
#include <windows.h>
#include <vector>
#include <cstdint>
#include <iostream>

// Affinity helpers (migrated from Config.hpp)
uint64_t coreMask(const std::vector<unsigned>& cores);
bool setAffinity(uint64_t mask, bool log=true);
