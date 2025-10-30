#pragma once
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#endif

// Lightweight helpers to reduce log boilerplate while keeping original texts/logic
// - Keep using lg::o()/lg::e()/lg::wo()
// - Add minimal manipulators/macros for prefixes, hex formatting, HRESULT, and Win32 errors

// Short alias (narrow)
#define P_B   "[BOOT] "
#define P_A   "[AFF] "
#define P_C   "[CAPTURE] "
#define P_I   "[INFO] "
#define P_D   "[DEBUG] "
#define P_E   "[ERROR] "
#define P_F   "[FATAL] "
#define P_T   "[TRT-YOLO PERF] "
#define P_L   "[LAT] "
#define P_CP  "[CAPTURE PERF] "
#define P_CO  "[CAPTURE OPT] "
#define P_MS  "[MMCSS] "
#define P_M   "[Mouse] "
#ifdef _WIN32
// Short alias (wide)
#define WP_B  L"[BOOT] "
#define WP_MS L"[MMCSS] "
#endif

// Conditional logging wrapper: LOG_IF(cond, stream_expr)
#define LOG_IF(cond, expr) do { if (cond) { expr; } } while(0)

namespace lg {
    inline std::ostream& o() noexcept { return std::cout; }
    inline std::ostream& e() noexcept { return std::cerr; }
    inline std::wostream& wo() noexcept { return std::wcout; }

    // Stream manipulators for hex uppercase and default reset
    inline std::ostream& hexu(std::ostream& os) { os << std::hex << std::uppercase; return os; }
    inline std::ostream& deff(std::ostream& os) { os << std::dec << std::nouppercase; return os; }

    // HRESULT/hex printer helper
    struct hrx_t { unsigned long v; };
    inline std::ostream& operator<<(std::ostream& os, const hrx_t& h) {
        os << "0x" << hexu << h.v << deff; return os;
    }

    inline hrx_t hrx(unsigned long v) { return hrx_t{ v }; }
#ifdef _WIN32
    inline hrx_t hrx(HRESULT hr) { return hrx_t{ static_cast<unsigned long>(hr) }; }
    inline unsigned long winerr() { return static_cast<unsigned long>(GetLastError()); }
#endif
}
