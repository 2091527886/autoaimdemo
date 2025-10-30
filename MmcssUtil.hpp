#pragma once
#include <windows.h>
#include <avrt.h>
#include <iostream>

// MMCSS utilities (migrated)
HANDLE MmcssEnable(bool enable, const wchar_t* task, AVRT_PRIORITY prio, DWORD* outIdx=nullptr);
void MmcssDisable(HANDLE h);
