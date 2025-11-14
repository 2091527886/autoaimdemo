#include "CtrlHandler.hpp"

std::atomic<bool> g_exit_req{false};
std::atomic<bool> g_exit_done{false};

BOOL WINAPI CtrlHandler(DWORD type) {
    switch (type) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
        g_exit_req.store(true, std::memory_order_relaxed);
        return TRUE;
    case CTRL_CLOSE_EVENT:
    case CTRL_LOGOFF_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        g_exit_req.store(true, std::memory_order_relaxed);
        // wait up to ~2s for graceful exit
        for (int i = 0; i < 200; ++i) { // ~2s
            if (g_exit_done.load(std::memory_order_relaxed)) return TRUE;
            Sleep(10);
        }
        return FALSE; // not finished in time -> let system terminate
    default:
        return FALSE;
    }
}
