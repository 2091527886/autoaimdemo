#pragma once
#include <windows.h>
#include <atomic>

extern std::atomic<bool> g_exit_req;
extern std::atomic<bool> g_exit_done;

BOOL WINAPI CtrlHandler(DWORD type);
