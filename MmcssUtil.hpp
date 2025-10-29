#pragma once
#include <windows.h>
#include <avrt.h>
#include <iostream>

// MMCSS utilities (migrated)
HANDLE EnableMmcssIfConfigured(bool enable, const wchar_t* task_name, AVRT_PRIORITY priority, DWORD* out_task_index=nullptr);
void DisableMmcss(HANDLE h);
