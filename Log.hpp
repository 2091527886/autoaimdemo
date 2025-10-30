#pragma once
#include <iostream>
#include <iomanip>
#include <string>

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

    // Newline/flush wrapper manipulators
    inline std::ostream& endl(std::ostream& os) { return os << std::endl; }
    inline std::wostream& wendl(std::wostream& os) { return os << std::endl; }
    // Lightweight single newline (no flush), for prefixing with a blank line
    inline std::ostream& nl(std::ostream& os) { return os << '\n'; }

    // Stream manipulators for hex uppercase and default reset
    inline std::ostream& hexu(std::ostream& os) { os << std::hex << std::uppercase; return os; }
    inline std::ostream& deff(std::ostream& os) { os << std::dec << std::nouppercase; return os; }

    // Fixed-point precision helpers (short, consistent)
    inline std::ostream& fx1(std::ostream& os) { os << std::fixed << std::setprecision(1); return os; }
    inline std::ostream& fx2(std::ostream& os) { os << std::fixed << std::setprecision(2); return os; }
    inline std::ostream& fx3(std::ostream& os) { os << std::fixed << std::setprecision(3); return os; }
    inline std::ostream& df(std::ostream& os)  { os << std::dec << std::nouppercase; return os; }

    // "over N frames" chunk with tag prefix
    struct ovr_t { const char* tag; size_t n; };
    inline ovr_t ovr(const char* tag, size_t n) { return { tag, n }; }
    inline std::ostream& operator<<(std::ostream& os, const ovr_t& v) {
        os << v.tag << v.n << "f"; return os;
    }

    // Field helper: prints " | name=value<suffix>"
    struct fld_t { const char* name; double v; const char* suf; };
    inline fld_t fld(const char* name, double v, const char* suf = "") { return { name, v, suf }; }
    inline std::ostream& operator<<(std::ostream& os, const fld_t& f) {
        os << " | " << f.name << "=" << f.v << f.suf; return os;
    }

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

    // Multi-line helper for same tag prefix
    inline void ml(const char* tag, const std::string& a, const std::string& b, const std::string& c) {
        o() << tag << a << '\n' << tag << b << '\n' << tag << c << std::endl;
    }

    // Forward decls needed by ok_or_werr
    void werr(const char* tag, const char* msg);

    // Success/Failure unified logging
    inline void ok_or_werr(const char* tag, const char* what, bool ok) {
        if (ok) {
            o() << tag << what << " OK" << std::endl;
        } else {
            std::string msg = std::string(what) + " failed";
            werr(tag, msg.c_str());
        }
    }
    inline void ok_or_werr_if(bool log, const char* tag, const char* what, bool ok) {
        if (log) ok_or_werr(tag, what, ok);
    }
    // Success/Failure with dynamic message
    inline void ok_or_werr_msg(bool log, const char* tag, const std::string& what, bool ok) {
        if (!log) return;
        if (ok) {
            o() << tag << what << " OK" << std::endl;
        } else {
            werr(tag, (what + " failed").c_str());
        }
    }

    // Win32 error helpers (short)
    inline void werr(const char* tag, const char* msg) {
    #ifdef _WIN32
        e() << tag << msg << ": " << winerr() << std::endl;
    #else
        e() << tag << msg << std::endl;
    #endif
    }
    inline void werr_if(bool cond, const char* tag, const char* msg) {
        if (cond) werr(tag, msg);
    }

    // HRESULT failure helper
    inline void hrfail(const char* tag, const char* api, unsigned long hr) {
        e() << tag << api << " failed: " << hrx(hr) << std::endl;
    }
#ifdef _WIN32
    inline void hrfail(const char* tag, const char* api, HRESULT hr) {
        e() << tag << api << " failed: " << hrx(hr) << std::endl;
    }
    // Success/Failure for HRESULT-producing operations
    inline void ok_or_hrfail_if(bool log, const char* tag, const char* api, HRESULT hr, bool ok) {
        if (log) {
            if (ok) {
                o() << tag << api << " OK" << std::endl;
            } else {
                hrfail(tag, api, hr);
            }
        }
    }
#endif
}
