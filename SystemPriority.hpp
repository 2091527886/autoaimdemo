#pragma once
#include <windows.h>
#include <winnt.h>
#include <iostream>

// Process and thread priority utilities (migrated from main.cpp, no logic changes)
void SetProcPrioHigh();
void SetThreadPrioHighest();
void SetThreadMemPrioNormal(bool log=true);
void SetProcPrioBoost(bool enable, bool log=true);
void SetProcPrioBoostDisabled(bool log=true);
