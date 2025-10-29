#pragma once
#include <windows.h>
#include <winnt.h>
#include <iostream>

// Process and thread priority utilities (migrated from main.cpp, no logic changes)
void RaiseProcessPrioritySafely();
void RaiseThisThreadPrioritySafely();
void SetThisThreadMemoryPriorityNormal(bool verbose=true);
void DisableProcessPriorityBoost(bool verbose=true);
