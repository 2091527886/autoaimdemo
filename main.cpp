#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <memory>
#include <sstream> // For std::ostringstream
#include <thread>  // For std::this_thread
#include <limits>
#include <tuple>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
// Windows & DirectX
#include <windows.h>
#include <winnt.h> // for MEMORY_PRIORITY_INFORMATION and MEMORY_PRIORITY_* constants
#include <d3d11.h>
#include <dxgi1_2.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#include <avrt.h>
#pragma comment(lib, "Avrt.lib")

// --- Priority helpers (process/thread/DXGI) ---
static inline void RaiseProcessPrioritySafely() {
    if (!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)) {
        std::cerr << "SetPriorityClass(HIGH) failed: " << GetLastError() << std::endl;
    }
}

static inline void RaiseThisThreadPrioritySafely() {
    if (!SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST)) {
        std::cerr << "SetThreadPriority(HIGHEST) failed: " << GetLastError() << std::endl;
    }
}

static inline void RaiseDxgiGpuPriorityAndLatency(ID3D11Device* d3dDevice, int gpu_thread_priority = 7, int max_frame_latency = 1) {
    if (!d3dDevice) return;
    IDXGIDevice1* dxgiDevice1 = nullptr;
    HRESULT hr = d3dDevice->QueryInterface(__uuidof(IDXGIDevice1), (void**)&dxgiDevice1);
    if (FAILED(hr) || !dxgiDevice1) {
        std::cerr << "QueryInterface(IDXGIDevice1) failed: 0x" << std::hex << (unsigned long)hr << std::dec << std::endl;
        return;
    }
    hr = dxgiDevice1->SetGPUThreadPriority(gpu_thread_priority);
    if (FAILED(hr)) {
        std::cerr << "SetGPUThreadPriority(7) failed: 0x" << std::hex << (unsigned long)hr << std::dec << std::endl;
    }
    hr = dxgiDevice1->SetMaximumFrameLatency(max_frame_latency);
    if (FAILED(hr)) {
        std::cerr << "SetMaximumFrameLatency(1) failed: 0x" << std::hex << (unsigned long)hr << std::dec << std::endl;
    }
    dxgiDevice1->Release();
}

// --- Power throttling/background-mode helpers ---

// 将当前线程内存优先级显式设置为 NORMAL（保证不被降级）
static inline void SetThisThreadMemoryPriorityNormal(bool verbose=true) {
    MEMORY_PRIORITY_INFORMATION mpi{};
    mpi.MemoryPriority = MEMORY_PRIORITY_NORMAL;
    if (!SetThreadInformation(GetCurrentThread(), ThreadMemoryPriority, &mpi, sizeof(mpi))) {
        if (verbose) std::cerr << "[BOOT] ThreadMemoryPriority=NORMAL failed: " << GetLastError() << std::endl;
    } else if (verbose) {
        std::cout << "[BOOT] ThreadMemoryPriority=NORMAL OK" << std::endl;
    }
}

// 禁用进程 PriorityBoost，避免在系统内存压力下被降低优先级
static inline void DisableProcessPriorityBoost(bool verbose=true) {
    if (!SetProcessPriorityBoost(GetCurrentProcess(), FALSE)) {
        if (verbose) std::cerr << "[BOOT] ProcPriorityBoost(FALSE) failed: " << GetLastError() << std::endl;
    } else if (verbose) {
        std::cout << "[BOOT] ProcPriorityBoost(FALSE) OK" << std::endl;
    }
}


// Forward declare Config for use in OverlayAnimator
class Config;

// Declaration only; definition moved after Config to avoid forward-declare issues
class OverlayAnimator {
public:
    OverlayAnimator();
    ~OverlayAnimator();
    bool start(const Config& cfg);
    void stop();
private:
    static LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
    void thread_proc(const Config& cfg);
    std::atomic<bool> running_ { false };
    std::thread worker_;
    HWND hwnd_ = nullptr;
};
static inline HANDLE EnableMmcssIfConfigured(bool enable, const wchar_t* task_name, AVRT_PRIORITY priority, DWORD* out_task_index=nullptr) {
    if (!enable) return nullptr;
    if (!task_name) task_name = L"Games";
    DWORD task_index = 0;
    HANDLE h = AvSetMmThreadCharacteristicsW(task_name, &task_index);
    if (!h) {
        std::cerr << "[MMCSS] AvSetMmThreadCharacteristicsW failed: " << GetLastError() << std::endl;
        return nullptr;
    }
    if (!AvSetMmThreadPriority(h, priority)) {
        std::cerr << "[MMCSS] AvSetMmThreadPriority failed: " << GetLastError() << std::endl;
    }
    if (out_task_index) *out_task_index = task_index;
    std::wcout << L"[MMCSS] Enabled for thread. Task=" << task_name << L", priority=" << (int)priority << std::endl;
    return h;
}
static inline void DisableMmcss(HANDLE h) {
    if (h) AvRevertMmThreadCharacteristics(h);
}
static inline void DisablePowerThrottlingForProcessAndThread(bool verbose=true) {
   // Disable process-wide execution speed throttling
   PROCESS_POWER_THROTTLING_STATE pstate = {};
   pstate.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
   pstate.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
   pstate.StateMask = 0; // clear EXECUTION_SPEED -> no throttling
   if (!SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &pstate, sizeof(pstate))) {
       if (verbose) std::cerr << "[BOOT] ProcPowerThrottling=OFF failed: " << GetLastError() << std::endl;
   } else if (verbose) {
       std::cout << "[BOOT] ProcPowerThrottling=OFF OK" << std::endl;
   }

   // Disable thread execution speed throttling for current thread
   THREAD_POWER_THROTTLING_STATE tstate = {};
   tstate.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
   tstate.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
   tstate.StateMask = 0; // clear EXECUTION_SPEED
   if (!SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &tstate, sizeof(tstate))) {
       if (verbose) std::cerr << "[BOOT] ThreadPowerThrottling=OFF failed: " << GetLastError() << std::endl;
   } else if (verbose) {
       std::cout << "[BOOT] ThreadPowerThrottling=OFF OK" << std::endl;
   }

   // Prevent Windows from automatically reducing this thread's priority during heavy paging
   if (!SetThreadPriorityBoost(GetCurrentThread(), FALSE)) {
       if (verbose) std::cerr << "[BOOT] ThreadPriorityBoost(FALSE) failed: " << GetLastError() << std::endl;
   } else if (verbose) {
       std::cout << "[BOOT] ThreadPriorityBoost(FALSE) OK" << std::endl;
   }
}


// TensorRT-YOLO
#include <trtyolo.hpp>
#pragma comment(lib, "trtyolo.lib")

// OpenCV
#include <opencv2/opencv.hpp>
#ifdef _DEBUG
#pragma comment(lib, "opencv_world4110d.lib") // 请根据你的OpenCV版本修改
#else
#pragma comment(lib, "opencv_world4110.lib") // 请根据你的OpenCV版本修改
#endif

// Input Simulator (确保头文件在项目路径中)
#include "InputSimulator.hpp"
#include "Config.hpp"

// Forward Declarations
struct Detection;
struct TimingDetails;
class Config;
class ScreenCapturer;
class ObjectDetector;
class MouseController;
class AimAssistant;

// --- 辅助宏 ---
template<class T>
void SafeRelease(T** ppT) {
    if (*ppT) {
        (*ppT)->Release();
        *ppT = nullptr;
    }
}

// --- 数据结构定义 ---

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

struct TimingDetails {
    double inference_ms = 0;
    double postprocess_ms = 0;
    double total_loop_ms = 0;
};

// 单帧延迟记录
struct FrameLatency {
    double capture_ms = 0.0;
    double detect_ms  = 0.0;
    double mouse_ms   = 0.0;
    double loop_ms    = 0.0;
};

// 简单的每 N 帧聚合并打印统计（平均/最小/最大），使用 std::chrono 的毫秒单位
class LatencyAggregator {
public:
    explicit LatencyAggregator(size_t window) : window_(window ? window : 30) {
        buffer_.reserve(window_);
    }

    void add(const FrameLatency& f) {
        buffer_.push_back(f);
        if (buffer_.size() >= window_) {
            print_and_clear();
        }
    }

    void force_print() {
        if (!buffer_.empty()) print_and_clear();
    }

private:
    void print_and_clear() {
        auto [cap_avg, cap_min, cap_max] = stats([](const FrameLatency& f){ return f.capture_ms; });
        auto [det_avg, det_min, det_max] = stats([](const FrameLatency& f){ return f.detect_ms;  });
        auto [mou_avg, mou_min, mou_max] = stats([](const FrameLatency& f){ return f.mouse_ms;   });
        auto [loo_avg, loo_min, loo_max] = stats([](const FrameLatency& f){ return f.loop_ms;    });

        std::cout << std::fixed << std::setprecision(2)
                  << "[LAT] over " << buffer_.size() << " frames"
                  << " | capture ms avg/min/max: " << cap_avg << "/" << cap_min << "/" << cap_max
                  << " | detect: " << det_avg << "/" << det_min << "/" << det_max
                  << " | mouse: "  << mou_avg << "/" << mou_min << "/" << mou_max
                  << " | loop: "   << loo_avg << "/" << loo_min << "/" << loo_max
                  << std::endl;

        buffer_.clear();
    }

    template<typename Getter>
    std::tuple<double,double,double> stats(Getter g) const {
        if (buffer_.empty()) return {0.0, 0.0, 0.0};
        double sum = 0.0, mn = std::numeric_limits<double>::infinity(), mx = 0.0;
        for (const auto& f : buffer_) {
            const double v = g(f);
            sum += v;
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        return { sum / static_cast<double>(buffer_.size()), (std::isinf(mn)?0.0:mn), mx };
    }

    size_t window_;
    std::vector<FrameLatency> buffer_;
};

// --- PID 控制器 ---
class PidController {
public:
    PidController() = default;
    PidController(double kp, double ki, double kd, double out_limit, double integ_limit)
        : kp_(kp), ki_(ki), kd_(kd), out_limit_(std::abs(out_limit)), integ_limit_(std::abs(integ_limit)) {}

    void SetGains(double kp, double ki, double kd) { kp_ = kp; ki_ = ki; kd_ = kd; }
    void SetLimits(double out_limit, double integ_limit) {
        out_limit_ = std::abs(out_limit);
        integ_limit_ = std::abs(integ_limit);
    }

    void Reset() { prev_error_ = 0.0; integral_ = 0.0; initialized_ = false; }

    double Update(double error, double dt_s) {
        if (dt_s <= 0.0) dt_s = 1.0 / 60.0; // fallback
        if (!initialized_) { prev_error_ = error; initialized_ = true; }

        // 误差导数
        double derivative = (error - prev_error_) / dt_s;

        // 积分项
        integral_ += error * dt_s;
        // 积分限幅
        if (integ_limit_ > 0.0) {
            if (integral_ > integ_limit_) integral_ = integ_limit_;
            else if (integral_ < -integ_limit_) integral_ = -integ_limit_;
        }

        double output = kp_ * error + ki_ * integral_ + kd_ * derivative;

        // 输出限幅 + 抗饱和：若饱和则在下次积分前自然恢复（简单策略）
        if (out_limit_ > 0.0) {
            if (output > out_limit_) output = out_limit_;
            else if (output < -out_limit_) output = -out_limit_;
        }

        prev_error_ = error;
        return output;
    }

private:
    double kp_ = 0.4;
    double ki_ = 0.0;
    double kd_ = 0.08;
    double out_limit_ = 50.0;    // 每帧输出像素限幅
    double integ_limit_ = 200.0; // 积分项限幅

    double prev_error_ = 0.0;
    double integral_ = 0.0;
    bool initialized_ = false;
};

// --- 1. 配置中心 (Config) ---

// 亲和工具：构建掩码 + 应用到当前线程（与进程亲和取交集）
// class Config moved to Config.hpp
// Only use it via #include "Config.hpp"

// --- 2. 鼠标控制器 (MouseController) ---
// === OverlayAnimator definition (after Config) ===
OverlayAnimator::OverlayAnimator() = default;
OverlayAnimator::~OverlayAnimator() { stop(); }

bool OverlayAnimator::start(const Config& cfg) {
    if (running_.load()) return true;
    if (!cfg.enable_overlay_animator) return false;
    running_.store(true);
    worker_ = std::thread([this, &cfg]{ this->thread_proc(cfg); });
    return true;
}
void OverlayAnimator::stop() {
    if (!running_.load()) return;
    running_.store(false);
    if (hwnd_) PostMessage(hwnd_, WM_CLOSE, 0, 0);
    if (worker_.joinable()) worker_.join();
    hwnd_ = nullptr;
}
LRESULT CALLBACK OverlayAnimator::WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}
void OverlayAnimator::thread_proc(const Config& cfg) {
    // 线程启动即绑定（如开启）
    if (cfg.enable_thread_affinity && cfg.overlay_thread_affinity_mask) {
        ApplyThreadAffinityMask(cfg.overlay_thread_affinity_mask,
            cfg.enable_optimization_report && cfg.enable_system_opt_report);
    }
    HINSTANCE hInst = GetModuleHandle(NULL);
    const wchar_t* cls = L"RovoOverlayAnimatorWnd";
    WNDCLASSEXW wc{ sizeof(WNDCLASSEXW) };
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = cls;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassExW(&wc);

    int sz = std::max(1, cfg.overlay_pixel_size);
    int x = cfg.overlay_pos_x;
    int y = cfg.overlay_pos_y;
    HWND hWnd = CreateWindowExW(WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
                                cls, L"", WS_POPUP, x, y, sz, sz, nullptr, nullptr, hInst, nullptr);
    if (!hWnd) { running_.store(false); return; }
    // Don't activate
    ShowWindow(hWnd, SW_SHOWNA);
    hwnd_ = hWnd;

    // Prepare a 1x1 or 2x2 DIB section for per-pixel alpha updates
    HDC screenDC = GetDC(nullptr);
    HDC memDC = CreateCompatibleDC(screenDC);
    BITMAPV5HEADER bi{};
    bi.bV5Size = sizeof(bi);
    bi.bV5Width = sz; bi.bV5Height = -sz; // top-down
    bi.bV5Planes = 1; bi.bV5BitCount = 32; bi.bV5Compression = BI_BITFIELDS;
    bi.bV5RedMask = 0x00FF0000; bi.bV5GreenMask = 0x0000FF00; bi.bV5BlueMask = 0x000000FF; bi.bV5AlphaMask = 0xFF000000;
    void* bits = nullptr;
    HBITMAP hbmp = CreateDIBSection(screenDC, (BITMAPINFO*)&bi, DIB_RGB_COLORS, &bits, nullptr, 0);
    HGDIOBJ old = SelectObject(memDC, hbmp);

    BLENDFUNCTION bf{ AC_SRC_OVER, 0, 255, AC_SRC_ALPHA };
    SIZE wndSize{ sz, sz };
    POINT srcPos{ 0, 0 };

    // High-frequency timer using waitable timer at 100ns units
    HANDLE hTimer = CreateWaitableTimerExW(NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (!hTimer) hTimer = CreateWaitableTimerW(NULL, FALSE, NULL);

    // Animation loop: toggle pixel alpha/color every interval
    uint32_t t = 0;
    while (running_.load()) {
        // write pixels
        uint32_t c1 = 0x01010101u * ((t & 1) ? 0x01 : 0x02); // tiny change in alpha/color
        for (int i = 0; i < sz * sz; ++i) {
            ((uint32_t*)bits)[i] = c1;
        }
        POINT wndPos{ x, y };
        UpdateLayeredWindow(hWnd, screenDC, &wndPos, &wndSize, memDC, &srcPos, 0, &bf, ULW_ALPHA);
        ++t;

        // wait for next tick (waitable timer + optional spin to improve accuracy)
        LARGE_INTEGER due;
        due.QuadPart = - (cfg.overlay_update_interval_100ns); // relative, negative, in 100ns units
        uint64_t spin_us = (uint64_t)std::max<long long>(0, cfg.overlay_spin_budget_us);
        if (hTimer && SetWaitableTimer(hTimer, &due, 0, nullptr, nullptr, FALSE)) {
            if (spin_us > 0) {
                // coarse wait then spin for the last few microseconds
                WaitForSingleObject(hTimer, INFINITE);
                auto spin_end = std::chrono::high_resolution_clock::now() + std::chrono::microseconds(spin_us);
                while (std::chrono::high_resolution_clock::now() < spin_end) {
                    // optional pause hint
                    YieldProcessor();
                }
            } else {
                WaitForSingleObject(hTimer, INFINITE);
            }
        } else {
            // Fallback: minimal sleep
            Sleep(0);
        }
    }
    if (hTimer) CloseHandle(hTimer);
    // Cleanup
    SelectObject(memDC, old);
    DeleteObject(hbmp);
    DeleteDC(memDC);
    ReleaseDC(nullptr, screenDC);
    DestroyWindow(hWnd);
}

class MouseController {
public:
    MouseController() {
        std::cout << "--- Initializing Mouse Simulation Driver... ---" << std::endl;
        hMouseDll = LoadLibrary(L"IbInputSimulator.dll");
        if (!hMouseDll) {
            throw std::runtime_error("Failed to load IbInputSimulator.dll! Ensure it's in the EXE directory.");
        }

        IbSendInit_ptr = (pIbSendInit)GetProcAddress(hMouseDll, "IbSendInit");
        IbSendDestroy_ptr = (pIbSendDestroy)GetProcAddress(hMouseDll, "IbSendDestroy");
        IbSendMouseMove_ptr = (pIbSendMouseMove)GetProcAddress(hMouseDll, "IbSendMouseMove");

        if (!IbSendInit_ptr || !IbSendDestroy_ptr || !IbSendMouseMove_ptr) {
            FreeLibrary(hMouseDll);
            hMouseDll = nullptr;
            throw std::runtime_error("Failed to find required functions in IbInputSimulator.dll.");
        }
//Logitech SendInput
        Send::Error error = IbSendInit_ptr(Send::SendType::Logitech, static_cast<Send::InitFlags>(0), nullptr);
        if (error != Send::Error::Success) { 
            FreeLibrary(hMouseDll);
            hMouseDll = nullptr;
            throw std::runtime_error("Failed to initialize mouse driver! Code: " + std::to_string(static_cast<int>(error)) + ". Run as Administrator.");
        }
        std::cout << "--- Mouse driver initialized successfully! ---" << std::endl;
    }

    ~MouseController() {
        if (hMouseDll) {
            if (IbSendDestroy_ptr) IbSendDestroy_ptr();
            FreeLibrary(hMouseDll);
            std::cout << "--- Mouse driver resources cleaned up. ---" << std::endl;
        }
    }

    MouseController(const MouseController&) = delete;
    MouseController& operator=(const MouseController&) = delete;

    void MoveRelative(int dx, int dy) {
        if (IbSendMouseMove_ptr) {
            bool ok = IbSendMouseMove_ptr(dx, dy, Send::MoveMode::Relative);
            if (!ok) {
                std::cerr << "[ERROR] IbSendMouseMove failed (dx=" << dx << ", dy=" << dy << ")" << std::endl;
            }
        }
    }

private:
    typedef Send::Error(__stdcall* pIbSendInit)(Send::SendType, Send::InitFlags, void*);
    typedef void(__stdcall* pIbSendDestroy)();
    typedef bool(__stdcall* pIbSendMouseMove)(int, int, Send::MoveMode);

    HMODULE hMouseDll = nullptr;
    pIbSendInit IbSendInit_ptr = nullptr;
    pIbSendDestroy IbSendDestroy_ptr = nullptr;
    pIbSendMouseMove IbSendMouseMove_ptr = nullptr;
};

// --- 3. 屏幕捕捉器 (ScreenCapturer) - 优化版本 ---
class ScreenCapturer {
public:
    ScreenCapturer(const Config& cfg_ref, int crop_width, int crop_height) : cfg(cfg_ref) {
        if (cfg.verbose_init_log) std::cout << "--- Initializing D3D for screen capture... ---" << std::endl;
        if (cfg.verbose_init_log) {
            std::cout << "[CAPTURE] Crop size: " << crop_width << "x" << crop_height << std::endl;
        }
        HRESULT hr;

        hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&pFactory);
        if (cfg.verbose_init_log) {
            std::cout << "[CAPTURE] CreateDXGIFactory1 hr=0x"
                      << std::hex << std::uppercase << static_cast<unsigned long>(hr)
                      << std::dec << std::nouppercase << std::endl;
        }
        if (FAILED(hr)) throw std::runtime_error("Failed to create DXGI Factory.");
        {
            HRESULT hr_enum_adapters = pFactory->EnumAdapters1(0, &pAdapter);
            if (cfg.verbose_init_log) {
                std::cout << "[CAPTURE] EnumAdapters1 hr=0x"
                          << std::hex << std::uppercase << static_cast<unsigned long>(hr_enum_adapters)
                          << std::dec << std::nouppercase << std::endl;
            }
            if (FAILED(hr_enum_adapters)) throw std::runtime_error("Failed to enumerate adapters.");
        }
        {
            HRESULT hr_enum_outputs = pAdapter->EnumOutputs(0, &pOutput);
            if (cfg.verbose_init_log) {
                std::cout << "[CAPTURE] EnumOutputs hr=0x"
                          << std::hex << std::uppercase << static_cast<unsigned long>(hr_enum_outputs)
                          << std::dec << std::nouppercase << std::endl;
            }
            if (FAILED(hr_enum_outputs)) throw std::runtime_error("Failed to enumerate outputs.");
        }

        DXGI_OUTPUT_DESC outputDesc;
        pOutput->GetDesc(&outputDesc);
        width = outputDesc.DesktopCoordinates.right - outputDesc.DesktopCoordinates.left;
        height = outputDesc.DesktopCoordinates.bottom - outputDesc.DesktopCoordinates.top;
        if (cfg.verbose_init_log) {
            std::cout << "[CAPTURE] Output desktop size: " << width << "x" << height << std::endl;
        }

        {
            HRESULT hr_create_device = D3D11CreateDevice(pAdapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &pDevice, nullptr, &pContext);
            if (cfg.verbose_init_log) {
                std::cout << "[CAPTURE] D3D11CreateDevice hr=0x"
                          << std::hex << std::uppercase << static_cast<unsigned long>(hr_create_device)
                          << std::dec << std::nouppercase << std::endl;
            }
            if (FAILED(hr_create_device)) {
                throw std::runtime_error("Failed to create D3D11 device.");
            }
            // Raise DXGI GPU priority and reduce frame latency for this device
            RaiseDxgiGpuPriorityAndLatency(pDevice, cfg.dxgi_gpu_thread_priority, cfg.dxgi_max_frame_latency);
        }
        {
            HRESULT hr_qi_output1 = pOutput->QueryInterface(__uuidof(IDXGIOutput1), (void**)&pOutput1);
            if (cfg.verbose_init_log) {
                std::cout << "[CAPTURE] QueryInterface(IDXGIOutput1) hr=0x"
                          << std::hex << std::uppercase << static_cast<unsigned long>(hr_qi_output1)
                          << std::dec << std::nouppercase << std::endl;
            }
            if (FAILED(hr_qi_output1)) {
                throw std::runtime_error("Failed to query IDXGIOutput1.");
            }
        }
        hr = pOutput1->DuplicateOutput(pDevice, &pDuplicator);
        if (cfg.verbose_init_log) {
            std::cout << "[CAPTURE] DuplicateOutput hr=" << (FAILED(hr) ? "FAILED" : "OK") << std::endl;
        }
        if (FAILED(hr)) {
            std::cerr << "DuplicateOutput failed. HRESULT=0x"
                      << std::hex << std::uppercase << static_cast<unsigned long>(hr)
                      << std::dec << std::nouppercase << std::endl;
            throw std::runtime_error("Failed to create output duplication.");
        }

        // 创建多缓冲 staging 纹理与事件查询
        D3D11_TEXTURE2D_DESC desc;
        desc.Width = crop_width;
        desc.Height = crop_height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Usage = D3D11_USAGE_STAGING;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        desc.MiscFlags = 0;

        for (size_t i = 0; i < kStagingCount; ++i) {
            hr = pDevice->CreateTexture2D(&desc, NULL, &staging_[i]);
            if (FAILED(hr)) {
                throw std::runtime_error("Failed to create staging texture[" + std::to_string(i) + "]");
            }
            D3D11_QUERY_DESC qd{}; qd.Query = D3D11_QUERY_EVENT; qd.MiscFlags = 0;
            hr = pDevice->CreateQuery(&qd, &copy_done_queries_[i]);
            if (FAILED(hr)) {
                throw std::runtime_error("Failed to create EVENT query for staging[" + std::to_string(i) + "]");
            }
        }
        std::cout << "--- Screen capture initialized successfully (" << width << "x" << height << ") with " << kStagingCount << " staging buffers ---" << std::endl;

        // 捕获链路启动横幅（仅在启用报告时）
        if (cfg.enable_optimization_report && cfg.enable_capture_opt_report) {
            std::cout << "[BOOT] Capture pipeline: "
                      << "dirty-skip=ON"
                      << " | staging-buffers=" << kStagingCount
                      << " | event-query=ON"
                      << " | map=DO_NOT_WAIT"
                      << " | cache-reuse=ON"
                      << std::endl;
            std::cout << "[BOOT] Timing: "
                      << "step-timing=" << (cfg.enable_capture_step_timing?"ON":"OFF")
                      << " | interval=" << cfg.capture_timing_interval_frames
                      << " | success-only=" << (cfg.capture_timing_only_on_success?"ON":"OFF")
                      << std::endl;
        }
    }

    ~ScreenCapturer() {
        std::cout << "--- Cleaning up D3D resources... ---" << std::endl;
        for (size_t i = 0; i < kStagingCount; ++i) {
            SafeRelease(&staging_[i]);
            SafeRelease(&copy_done_queries_[i]);
        }
        SafeRelease(&pDuplicator);
        SafeRelease(&pOutput1);
        SafeRelease(&pOutput);
        SafeRelease(&pAdapter);
        SafeRelease(&pFactory);
        SafeRelease(&pContext);
        SafeRelease(&pDevice);
    }

    ScreenCapturer(const ScreenCapturer&) = delete;
    ScreenCapturer& operator=(const ScreenCapturer&) = delete;

    bool CaptureFrame(cv::Mat& frame, const cv::Rect& crop_region) {
        // 高精度分步计时
        auto t_total_start = std::chrono::high_resolution_clock::now();
        double t_acquire_ms = 0.0, t_qi_ms = 0.0, t_copy_ms = 0.0, t_map_ms = 0.0, t_cvt_ms = 0.0, t_unmap_ms = 0.0, t_total_ms = 0.0;

        IDXGIResource* pDesktopResource = nullptr;
        DXGI_OUTDUPL_FRAME_INFO frameInfo;

        auto t_acq_s = std::chrono::high_resolution_clock::now();
        HRESULT hr = pDuplicator->AcquireNextFrame(cfg.capture_timeout_ms, &frameInfo, &pDesktopResource);
        auto t_acq_e = std::chrono::high_resolution_clock::now();
        t_acquire_ms = std::chrono::duration<double, std::milli>(t_acq_e - t_acq_s).count();

        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
            // 无更新：尝试复用上一帧缓存
            if (try_serve_cache(frame)) {
                ++stat_cache_reused_;
                return true;
            }
            // 仅在需要时统计失败帧
            if (cfg.enable_capture_step_timing && !cfg.capture_timing_only_on_success) {
                t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
                accumulate_and_maybe_print(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
            }
            return false;
        }
        if (FAILED(hr)) {
            std::cerr << "AcquireNextFrame failed. HRESULT: 0x" << std::hex << hr << std::endl;
            if (cfg.enable_capture_step_timing && !cfg.capture_timing_only_on_success) {
                t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
                accumulate_and_maybe_print(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
            }
            return false;
        }

        ID3D11Texture2D* pAcquiredDesktopImage = nullptr;
        auto t_qi_s = std::chrono::high_resolution_clock::now();
        hr = pDesktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&pAcquiredDesktopImage);
        auto t_qi_e = std::chrono::high_resolution_clock::now();
        t_qi_ms = std::chrono::duration<double, std::milli>(t_qi_e - t_qi_s).count();
        SafeRelease(&pDesktopResource);
        if (FAILED(hr)) {
            if (cfg.enable_capture_step_timing && !cfg.capture_timing_only_on_success) {
                t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
                accumulate_and_maybe_print(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
            }
            return false;
        }

        // Desktop Duplication 脏矩形检测：若无变化，跳过本帧（复用上一帧）
        UINT dirty_size = 0;
        HRESULT hr_dirty = pDuplicator->GetFrameDirtyRects(0, nullptr, &dirty_size);
        if (SUCCEEDED(hr_dirty) && dirty_size == 0) {
            // 无脏矩形：复用上一帧缓存，释放当前帧
            bool served = try_serve_cache(frame);
            SafeRelease(&pDesktopResource); // 保险起见
            SafeRelease(&pAcquiredDesktopImage);
            pDuplicator->ReleaseFrame();
            if (served) {
                ++stat_dirty_skips_;
                ++stat_cache_reused_;
                return true;
            }
            t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
            if (cfg.enable_capture_step_timing && !cfg.capture_timing_only_on_success) {
                accumulate_and_maybe_print(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
            }
            return false;
        }

        D3D11_BOX sourceRegion;
        sourceRegion.left = crop_region.x;
        sourceRegion.right = crop_region.x + crop_region.width;
        sourceRegion.top = crop_region.y;
        sourceRegion.bottom = crop_region.y + crop_region.height;
        sourceRegion.front = 0;
        sourceRegion.back = 1;

        // 双缓冲 staging + EVENT 查询：本帧拷贝到 curr，上一帧 prev 做 Map 读取
        const size_t curr = frame_index_ % kStagingCount;
        const size_t prev = (frame_index_ + kStagingCount - 1) % kStagingCount;

        auto t_copy_s = std::chrono::high_resolution_clock::now();
        pContext->CopySubresourceRegion(staging_[curr], 0, 0, 0, 0, pAcquiredDesktopImage, 0, &sourceRegion);
        auto t_copy_e = std::chrono::high_resolution_clock::now();
        t_copy_ms = std::chrono::duration<double, std::milli>(t_copy_e - t_copy_s).count();

        // 在 curr 上标记拷贝完成事件
        pContext->End(copy_done_queries_[curr]);

        // 释放当前获取的桌面帧（Duplication 侧）
        SafeRelease(&pAcquiredDesktopImage);
        pDuplicator->ReleaseFrame();

        // 仅在上一帧拷贝完成时尝试 Map 读取 prev；否则直接返回 false（上层复用上一帧）
        BOOL copy_done = FALSE;
        HRESULT qhr = pContext->GetData(copy_done_queries_[prev], &copy_done, sizeof(copy_done), 0);
        if (qhr != S_OK || !copy_done) {
            // 未就绪：不阻塞等待，直接跳过本帧的 Map（复用上一帧缓存）
            ++stat_event_not_ready_;
            bool served = try_serve_cache(frame);
            if (served) ++stat_cache_reused_;
            t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
            if (cfg.enable_capture_step_timing && !cfg.capture_timing_only_on_success) {
                accumulate_and_maybe_print(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
            }
            frame_index_++;
            return served; // 若有缓存则返回 true，保证上层有图像
        }

        // 非阻塞 Map 读取上一帧
        D3D11_MAPPED_SUBRESOURCE mappedResource;
        auto t_map_s = std::chrono::high_resolution_clock::now();
        hr = pContext->Map(staging_[prev], 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mappedResource);
        auto t_map_e = std::chrono::high_resolution_clock::now();
        t_map_ms = std::chrono::duration<double, std::milli>(t_map_e - t_map_s).count();
        if (hr == DXGI_ERROR_WAS_STILL_DRAWING) {
            // 未就绪：不阻塞等待，直接跳过本帧 Map（复用上一帧缓存）
            ++stat_map_still_drawing_;
            bool served = try_serve_cache(frame);
            if (served) ++stat_cache_reused_;
            t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
            if (cfg.enable_capture_step_timing && !cfg.capture_timing_only_on_success) {
                accumulate_and_maybe_print(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
            }
            frame_index_++;
            return served;
        }
        if (FAILED(hr)) {
            // 其它错误
            t_total_ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_total_start).count();
            if (cfg.enable_capture_step_timing && !cfg.capture_timing_only_on_success) {
                accumulate_and_maybe_print(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
            }
            frame_index_++;
            return false;
        }

        cv::Mat bgra_frame(crop_region.height, crop_region.width, CV_8UC4, mappedResource.pData, mappedResource.RowPitch);
        auto t_cvt_s = std::chrono::high_resolution_clock::now();
        cv::cvtColor(bgra_frame, frame, cv::COLOR_BGRA2BGR);
        // 缓存当前 BGR 结果以便无变化/未就绪时复用
        frame.copyTo(last_bgr_cache_);
        auto t_cvt_e = std::chrono::high_resolution_clock::now();
        t_cvt_ms = std::chrono::duration<double, std::milli>(t_cvt_e - t_cvt_s).count();

        auto t_unmap_s = std::chrono::high_resolution_clock::now();
        pContext->Unmap(staging_[prev], 0);
        auto t_unmap_e = std::chrono::high_resolution_clock::now();
        t_unmap_ms = std::chrono::duration<double, std::milli>(t_unmap_e - t_unmap_s).count();

        t_total_ms = std::chrono::duration<double, std::milli>(t_unmap_e - t_total_start).count();
        frame_index_++;

        if (cfg.enable_capture_step_timing || (cfg.enable_optimization_report && cfg.enable_capture_opt_report)) {
            accumulate_and_maybe_print(t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms);
        }
        ++stat_success_;
        return true;
    }

    int getWidth() const { return width; }
    int getHeight() const { return height; }

private:
    // 返回缓存图像（若存在），用于无变化/未就绪时复用上一帧
    bool try_serve_cache(cv::Mat& out) {
        if (!last_bgr_cache_.empty()) {
            // 确保尺寸匹配
            if (last_bgr_cache_.rows == out.rows && last_bgr_cache_.cols == out.cols && last_bgr_cache_.type() == out.type()) {
                last_bgr_cache_.copyTo(out);
                return true;
            } else {
                // 直接克隆为所需尺寸（若上层传入空 Mat，这里可直接赋值）
                out = last_bgr_cache_.clone();
                return true;
            }
        }
        return false;
    }

    // 采集计时聚合与打印（平均值），满足频率后输出并清零
    void accumulate_and_maybe_print(double t_acquire_ms, double t_qi_ms, double t_copy_ms,
                                    double t_map_ms, double t_cvt_ms, double t_unmap_ms, double t_total_ms) {
       // 统计优化触发计数（按周期打印）
       static size_t opt_counter = 0;
       static size_t stat_dirty_skips = 0, stat_event_not_ready = 0, stat_map_still_drawing = 0, stat_cache_reused = 0, stat_success = 0;
       // 将成员级累积并入周期统计
       stat_dirty_skips += stat_dirty_skips_;
       stat_event_not_ready += stat_event_not_ready_;
       stat_map_still_drawing += stat_map_still_drawing_;
       stat_cache_reused += stat_cache_reused_;
       stat_success += stat_success_;
       // 清零成员计数，避免重复累加
       stat_dirty_skips_ = stat_event_not_ready_ = stat_map_still_drawing_ = stat_cache_reused_ = stat_success_ = 0;
       // 递增优化统计的帧计数
       opt_counter++;
        static size_t counter = 0;
        static double acc_acq = 0.0, acc_qi = 0.0, acc_copy = 0.0, acc_map = 0.0, acc_cvt = 0.0, acc_unmap = 0.0, acc_total = 0.0;
        counter++;
        acc_acq += t_acquire_ms;
        acc_qi += t_qi_ms;
        acc_copy += t_copy_ms;
        acc_map += t_map_ms;
        acc_cvt += t_cvt_ms;
        acc_unmap += t_unmap_ms;
        acc_total += t_total_ms;

        bool should_print = cfg.capture_timing_every_frame;
        if (!should_print) {
            int interval = std::max(1, cfg.capture_timing_interval_frames);
            if (interval <= 1) should_print = true;
            else if (counter % static_cast<size_t>(interval) == 0) should_print = true;
        }
        if (should_print) {
            double n = static_cast<double>(counter);
            std::cout << std::fixed << std::setprecision(3)
                      << "[CAPTURE PERF] over " << counter << " frames"
                      << " | acquire=" << (acc_acq / n) << " ms"
                      << " | qi=" << (acc_qi / n) << " ms"
                      << " | copy=" << (acc_copy / n) << " ms"
                      << " | map=" << (acc_map / n) << " ms"
                      << " | cvt=" << (acc_cvt / n) << " ms"
                      << " | unmap=" << (acc_unmap / n) << " ms"
                      << " | total=" << (acc_total / n) << " ms"
                      << std::endl;

            // 按独立周期打印优化触发统计（仅当启用报告开关）
            if (cfg.enable_optimization_report && cfg.enable_capture_opt_report) {
                bool should_print_opt = false;
                int opt_interval = std::max(1, cfg.optimization_report_interval_frames);
                if (opt_interval <= 1) should_print_opt = true;
                else if (opt_counter % static_cast<size_t>(opt_interval) == 0) should_print_opt = true;
                if (should_print_opt) {
                    size_t total = static_cast<size_t>(opt_counter);
                    auto pct = [&](size_t v){ return total? (100.0 * double(v) / double(total)) : 0.0; };
                    std::cout << std::fixed << std::setprecision(1)
                              << "[CAPTURE OPT] over " << total << " frames"
                              << " | dirty-skip=" << stat_dirty_skips << " (" << pct(stat_dirty_skips) << "%)"
                              << " | event-not-ready=" << stat_event_not_ready << " (" << pct(stat_event_not_ready) << "%)"
                              << " | map-still-drawing=" << stat_map_still_drawing << " (" << pct(stat_map_still_drawing) << "%)"
                              << " | cache-reused=" << stat_cache_reused << " (" << pct(stat_cache_reused) << "%)"
                              << " | success=" << stat_success << " (" << pct(stat_success) << "%)"
                              << std::endl;
                    // 清零周期统计
                    opt_counter = 0;
                    stat_dirty_skips = stat_event_not_ready = stat_map_still_drawing = stat_cache_reused = stat_success = 0;
                }
            }

            counter = 0;
            acc_acq = acc_qi = acc_copy = acc_map = acc_cvt = acc_unmap = acc_total = 0.0;
        }
    }

    // 配置引用
    const Config& cfg;

    // 多缓冲 staging 与事件查询
    static constexpr size_t kStagingCount = 2;
    std::array<ID3D11Texture2D*, kStagingCount> staging_{}; // CPU 读 staging textures
    std::array<ID3D11Query*,     kStagingCount> copy_done_queries_{}; // EVENT queries 标记拷贝完成
    size_t frame_index_ = 0;
    cv::Mat last_bgr_cache_; // 上一帧 BGR 缓存（与输出相同尺寸格式）

    // 捕获优化触发计数（成员级，周期内累加）
    size_t stat_dirty_skips_ = 0;
    size_t stat_event_not_ready_ = 0;
    size_t stat_map_still_drawing_ = 0;
    size_t stat_cache_reused_ = 0;
    size_t stat_success_ = 0;

    IDXGIFactory1* pFactory = nullptr;
    IDXGIAdapter1* pAdapter = nullptr;
    IDXGIOutput* pOutput = nullptr;
    IDXGIOutput1* pOutput1 = nullptr;
    ID3D11Device* pDevice = nullptr;
    ID3D11DeviceContext* pContext = nullptr;
    IDXGIOutputDuplication* pDuplicator = nullptr;
    int width = 0;
    int height = 0;
};

// --- 4. 目标检测器 (ObjectDetector) ---
class ObjectDetector {
public:
    explicit ObjectDetector(const Config& cfg) {
        // English logs: load TensorRT-YOLO engine
        const std::string engine_path = cfg.resolve_engine_path();
        std::cout << "[INFO] Loading TensorRT-YOLO engine from " << engine_path << " ..." << std::endl;
        trtyolo::InferOption option;
        option.enableSwapRB(); // Input is BGR, swap to RGB internally
        // Input from screen capture is 640x640 by config; fix input dims for performance (optional)
        option.setInputDimensions(cfg.crop_size, cfg.crop_size);
        // Enable internal performance report from TensorRT-YOLO if enabled in config
        if (cfg.enable_performance_report) {
            option.enablePerformanceReport();
        }

        // Pre-flight checks and diagnostics
        {
            // Print current working directory for path diagnostics
            char cwd_buffer[MAX_PATH];
            if (GetCurrentDirectoryA(MAX_PATH, cwd_buffer)) {
                std::cout << "[DEBUG] Current working directory: " << cwd_buffer << std::endl;
            }
            // Check engine file existence
            DWORD attrs = GetFileAttributesA(engine_path.c_str());
            if (attrs == INVALID_FILE_ATTRIBUTES) {
                std::cerr << "[ERROR] Engine file not found: '" << engine_path << "'."
                          << " Ensure the engine file is present or provide the correct path." << std::endl;
            } else if (attrs & FILE_ATTRIBUTE_DIRECTORY) {
                std::cerr << "[ERROR] Engine path is a directory, not a file: '" << engine_path << "'." << std::endl;
            } else {
                std::cout << "[DEBUG] Found engine file: '" << engine_path << "'." << std::endl;
            }
        }

        try {
            detector_ = std::make_unique<trtyolo::DetectModel>(engine_path.c_str(), option);
            std::cout << "[INFO] Engine loaded successfully." << std::endl;
        } catch (const std::exception& e) {
            std::ostringstream oss;
            oss << "[FATAL] Failed to construct DetectModel with engine '" << engine_path << "'.\n"
                << "Diagnostics: Please check that:\n"
                << "- The engine file path is correct and accessible.\n"
                << "- TensorRT runtime version matches engine build version.\n"
                << "- Required TensorRT plugins are correctly registered.\n"
                << "- The engine was built for this GPU/driver.\n"
                << "Underlying error: " << e.what();
            throw std::runtime_error(oss.str());
        }
    }

    ObjectDetector(const ObjectDetector&) = delete;
    ObjectDetector& operator=(const ObjectDetector&) = delete;

    void Detect(const cv::Mat& image, std::vector<Detection>& detections, const Config& cfg, TimingDetails& timings) {
        detections.clear();
        auto t0 = std::chrono::high_resolution_clock::now();

        // Wrap cv::Mat to trtyolo::Image (with pitch)
        trtyolo::Image input(const_cast<unsigned char*>(image.data), image.cols, image.rows, static_cast<size_t>(image.step));
        trtyolo::DetectRes res = detector_->predict(input);

        auto t1 = std::chrono::high_resolution_clock::now();
        timings.inference_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        auto t2 = std::chrono::high_resolution_clock::now();

        // Convert to our Detection structure with threshold filtering
        if (res.num > 0 && res.boxes.size() == static_cast<size_t>(res.num)) {
            for (int i = 0; i < res.num; ++i) {
                float score = (i < (int)res.scores.size() ? res.scores[i] : 0.0f);
                if (score < cfg.confidence_threshold) continue;
                int cls = (i < (int)res.classes.size() ? res.classes[i] : -1);
                const auto& b = res.boxes[i];
                int x = std::max(0, static_cast<int>(std::floor(b.left)));
                int y = std::max(0, static_cast<int>(std::floor(b.top)));
                int w = std::max(0, static_cast<int>(std::round(b.right - b.left)));
                int h = std::max(0, static_cast<int>(std::round(b.bottom - b.top)));
                // Clamp to image
                w = std::min(w, image.cols - x);
                h = std::min(h, image.rows - y);
                if (w > 0 && h > 0)
                    detections.emplace_back(Detection{ cv::Rect(x, y, w, h), score, cls });
            }
        }

        auto t3 = std::chrono::high_resolution_clock::now();
        timings.postprocess_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

        // Query and print TensorRT-YOLO performance report based on frequency settings
        if (cfg.enable_performance_report) {
            bool should_report = cfg.perf_report_every_frame;
            if (!should_report) {
                perf_frame_counter_++;
                if (cfg.perf_report_interval_frames <= 1) {
                    should_report = true;
                } else if (perf_frame_counter_ % static_cast<size_t>(cfg.perf_report_interval_frames) == 0) {
                    should_report = true;
                }
            }
            if (should_report) {
                auto report = detector_->performanceReport();
                const std::string& throughput = std::get<0>(report);
                if (!throughput.empty()) {
                    const std::string& cpu_latency = std::get<1>(report);
                    const std::string& gpu_latency = std::get<2>(report);
                    std::cout << "[TRT-YOLO PERF] " << throughput << '\n'
                              << "[TRT-YOLO PERF] " << cpu_latency << '\n'
                              << "[TRT-YOLO PERF] " << gpu_latency << std::endl;
                }
            }
        }
    }

private:
    std::unique_ptr<trtyolo::DetectModel> detector_;
    // 性能报告打印计数器
    size_t perf_frame_counter_ = 0;
};

// --- 5. 核心协调器 (AimAssistant) ---
class AimAssistant {
public:
    AimAssistant()
        : cfg(),
        capturer(cfg, cfg.crop_size, cfg.crop_size), // 传递 cfg + crop_size
        detector(cfg),
        mouse(),
        crop_region((capturer.getWidth() - cfg.crop_size) / 2, (capturer.getHeight() - cfg.crop_size) / 2, cfg.crop_size, cfg.crop_size),
        crop_center(cfg.crop_size / 2, cfg.crop_size / 2),
        is_visualizing(cfg.enable_visualization),
        aggregator_(static_cast<size_t>(std::max(1, cfg.latency_log_interval_frames)))
    {
        if (capturer.getWidth() < cfg.crop_size || capturer.getHeight() < cfg.crop_size) {
            throw std::runtime_error("Screen resolution is smaller than configured crop_size.");
        }
        mouse_correction_factor = (cfg.horizontal_fov / static_cast<double>(capturer.getWidth()))
            * (cfg.pixels_for_360_turn / 360.0);
        if (is_visualizing) {
            cv::namedWindow(cfg.window_name, cv::WINDOW_AUTOSIZE);
        }
        std::cout << "--- Detection will run on a centered " << cfg.crop_size << "x" << cfg.crop_size << " region. ---" << std::endl;
        print_instructions();

        // 初始化 PID 控制器（像素域）。输出再乘以 mouse_correction_factor 转为鼠标单位。
        pid_x.SetGains(cfg.pid_kp_x, cfg.pid_ki_x, cfg.pid_kd_x);
        pid_x.SetLimits(cfg.pid_output_limit, cfg.pid_integral_limit);
        pid_y.SetGains(cfg.pid_kp_y, cfg.pid_ki_y, cfg.pid_kd_y);
        pid_y.SetLimits(cfg.pid_output_limit, cfg.pid_integral_limit);
    }

    ~AimAssistant() {
        if (is_visualizing) cv::destroyAllWindows();
    }

    void Run() {
        while (true) {
            auto loop_start_time = std::chrono::high_resolution_clock::now();

            FrameLatency fl; // 本帧延迟统计

            // 1) 截屏计时
            auto t_cap_start = std::chrono::high_resolution_clock::now();
            bool got = capturer.CaptureFrame(captured_frame, crop_region);
            auto t_cap_end = std::chrono::high_resolution_clock::now();
            fl.capture_ms = std::chrono::duration<double, std::milli>(t_cap_end - t_cap_start).count();
            if (!got) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                // 仅统计成功截屏帧：直接继续下一帧
                if (cfg.enable_latency_log && !cfg.latency_only_on_captured) {
                    // 如果也要统计失败帧，则把仅有 capture_ms 的 fl 加入
                    aggregator_.add(fl);
                }
                continue;
            }

            // 2) 检测计时（外层整体）
            auto t_det_start = std::chrono::high_resolution_clock::now();
            detector.Detect(captured_frame, detections, cfg, timings);
            auto t_det_end = std::chrono::high_resolution_clock::now();
            fl.detect_ms = std::chrono::duration<double, std::milli>(t_det_end - t_det_start).count();

            // 3) 鼠标控制计时（整体阶段）
            auto t_mouse_start = std::chrono::high_resolution_clock::now();
            Detection* best_target = findBestTarget();
            handleMouseInput(best_target);
            auto t_mouse_end = std::chrono::high_resolution_clock::now();
            fl.mouse_ms = std::chrono::duration<double, std::milli>(t_mouse_end - t_mouse_start).count();

            auto loop_end_time = std::chrono::high_resolution_clock::now();
            timings.total_loop_ms = std::chrono::duration<double, std::milli>(loop_end_time - loop_start_time).count();
            fl.loop_ms = timings.total_loop_ms;

            // 计算 dt（秒）供 PID 使用
            if (cfg.pid_use_measured_dt) {
                pid_dt_s = std::max(0.001, timings.total_loop_ms / 1000.0);
            } else {
                pid_dt_s = std::max(0.001, cfg.pid_fixed_dt_ms / 1000.0);
            }
            updateAndPrintStats();

            if (cfg.enable_latency_log) {
                aggregator_.add(fl);
            }

            if (is_visualizing) {
                handleVisualization(best_target);
            }
            char key = static_cast<char>(cv::waitKey(1));
            if (key == 27) { // ESC
                std::cout << "\nESC pressed. Exiting..." << std::endl;
                break;
            }
            if (key == 'v' || key == 'V') {
                toggleVisualization();
            }
        }
    }

private:
    void print_instructions() {
        std::cout << "\n--- Controls ---" << std::endl;
        std::cout << "    [Hold Left Mouse Button] to smoothly lock onto a target." << std::endl;
        std::cout << "    [Press F8] to move mouse to target ONCE (for accuracy test)." << std::endl;
        std::cout << "    [Press V in visualization window] to toggle visualization ON/OFF." << std::endl;
        std::cout << "    [Press ESC in visualization window] to quit." << std::endl;
    }

    Detection* findBestTarget() {
        Detection* target = nullptr;
        double min_dist_to_center = cfg.max_lock_distance_pixels;
        for (auto& det : detections) {
            if (!cfg.is_target_class(det.class_id)) continue; // 过滤非目标类别
            cv::Point box_center(det.box.x + det.box.width / 2, det.box.y + det.box.height / 2);
            double dist = std::hypot(box_center.x - crop_center.x, box_center.y - crop_center.y);
            if (dist < min_dist_to_center) {
                min_dist_to_center = dist;
                target = &det;
            }
        }
        return target;
    }

    void handleMouseInput(const Detection* target) {
        // 若无目标或未按住按键：释放 PID 积分，避免历史积分殘留
        bool smooth_pressed = (GetAsyncKeyState(cfg.smooth_aim_key) & 0x8000) != 0;
        if (!target || !smooth_pressed) {
            pid_x.Reset();
            pid_y.Reset();
            // 仍保留单次 F8 逻辑
            if (!target) return;
        }

        if (smooth_pressed) {
            cv::Point target_point(
                target->box.x + target->box.width / 2,
                target->box.y + static_cast<int>(target->box.height * cfg.target_y_ratio)
            );
            double err_x = static_cast<double>(target_point.x - crop_center.x);
            double err_y = static_cast<double>(target_point.y - crop_center.y);

            // PID 在像素域输出，再乘鼠标校正因子转为鼠标单位
            double out_x_pix = pid_x.Update(err_x, pid_dt_s);
            double out_y_pix = pid_y.Update(err_y, pid_dt_s);
            int move_x = static_cast<int>(out_x_pix * mouse_correction_factor);
            int move_y = static_cast<int>(out_y_pix * mouse_correction_factor);

            if (move_x != 0 || move_y != 0) {
                mouse.MoveRelative(move_x, move_y);
            }
        }

        // 单发模式：仍使用原比例（便于测试 PID 与比例的差异）
        static bool f8_was_pressed = false;
        bool f8_is_pressed = (GetAsyncKeyState(cfg.single_shot_key) & 0x8000) != 0;
        if (f8_is_pressed && !f8_was_pressed) {
            cv::Point target_point(
                target->box.x + target->box.width / 2,
                target->box.y + static_cast<int>(target->box.height * cfg.target_y_ratio)
            );
            double dx_pixels = static_cast<double>(target_point.x - crop_center.x);
            double dy_pixels = static_cast<double>(target_point.y - crop_center.y);
            double corrected_dx = dx_pixels * mouse_correction_factor;
            double corrected_dy = dy_pixels * mouse_correction_factor;
            int move_x = static_cast<int>(corrected_dx);
            int move_y = static_cast<int>(corrected_dy);
            if (move_x != 0 || move_y != 0) {
                mouse.MoveRelative(move_x, move_y);
            }
        }
        f8_was_pressed = f8_is_pressed;
    }

    void updateAndPrintStats() {
        const double smoothing_factor = 0.05;
        if (is_first_frame) {
            smoothed_total = timings.total_loop_ms;
            smoothed_inf = timings.inference_ms;
            smoothed_post = timings.postprocess_ms;
            is_first_frame = false;
        }
        else {
            smoothed_total = smoothing_factor * timings.total_loop_ms + (1.0 - smoothing_factor) * smoothed_total;
            smoothed_inf = smoothing_factor * timings.inference_ms + (1.0 - smoothing_factor) * smoothed_inf;
            smoothed_post = smoothing_factor * timings.postprocess_ms + (1.0 - smoothing_factor) * smoothed_post;
        }

    }

    void handleVisualization(const Detection* best_target) {
        cv::circle(captured_frame, crop_center, static_cast<int>(cfg.max_lock_distance_pixels), cv::Scalar(255, 255, 0), 1);
        for (const auto& det : detections) {
            // 可视化：所有检测都显示；追踪类加星号，当前目标红框
            cv::Scalar color = (&det == best_target) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
            cv::rectangle(captured_frame, det.box, color, 2);
            std::string label = cfg.class_name_or_id(det.class_id) + (cfg.is_target_class(det.class_id) ? "*" : "") +
                                 " " + cv::format("%.2f", det.confidence);
            cv::putText(captured_frame, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }
        if (smoothed_total > 0) {
            double smoothed_fps = 1000.0 / smoothed_total;
            std::ostringstream stats_stream;
            stats_stream << std::fixed << std::setprecision(1) << "FPS: " << smoothed_fps;
            cv::putText(captured_frame, stats_stream.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
        }
        cv::imshow(cfg.window_name, captured_frame);
    }

    void toggleVisualization() {
        is_visualizing = !is_visualizing;
        if (is_visualizing) {
            cv::namedWindow(cfg.window_name, cv::WINDOW_AUTOSIZE);
            std::cout << "\n[INFO] Visualization toggled ON." << std::endl;
        }
        else {
            cv::destroyAllWindows();
            std::cout << "\n[INFO] Visualization toggled OFF. Console stats will continue." << std::endl;
        }
    }

    Config cfg;
    ScreenCapturer capturer;
    ObjectDetector detector;
    MouseController mouse;
    const cv::Rect crop_region;
    const cv::Point crop_center;
    double mouse_correction_factor = 1.0;
    cv::Mat captured_frame;
    std::vector<Detection> detections;
    TimingDetails timings;
    bool is_visualizing;
    double smoothed_total = 0.0, smoothed_inf = 0.0, smoothed_post = 0.0;
    bool is_first_frame = true;
    // 延迟统计
    LatencyAggregator aggregator_;

    // PID 控制
    PidController pid_x;
    PidController pid_y;
    double pid_dt_s = 1.0/60.0;
};

// --- 主函数 ---
int main() {
    Config boot_cfg; // KEEP
    // 从环境变量读取可选亲和配置
    boot_cfg.load_affinity_from_env();
    if (boot_cfg.enable_thread_affinity && boot_cfg.main_thread_affinity_mask) {
        ApplyThreadAffinityMask(boot_cfg.main_thread_affinity_mask,
            boot_cfg.enable_optimization_report && boot_cfg.enable_system_opt_report);
    }

    OverlayAnimator overlay;
    // Disable process/thread power throttling and background limits as early as possible
    DisablePowerThrottlingForProcessAndThread();
    // Raise process and main thread priority early
    RaiseProcessPrioritySafely();
    RaiseThisThreadPrioritySafely();

    // Enable MMCSS for main loop thread based on default Config values
   
    HANDLE hMmcss = EnableMmcssIfConfigured(boot_cfg.enable_mmcss_for_main_loop,
                                            boot_cfg.mmcss_task_name.c_str(),
                                            boot_cfg.mmcss_priority);

   // 提升主线程内存优先级 & 禁用进程级 PriorityBoost
   if (boot_cfg.enable_thread_memory_priority) {
       SetThisThreadMemoryPriorityNormal();
       DisableProcessPriorityBoost();
   }

    // Start tiny overlay animator (forces DWM updates) if enabled
    if (boot_cfg.enable_overlay_animator) {
        overlay.start(boot_cfg);
        if (boot_cfg.enable_optimization_report && boot_cfg.enable_system_opt_report) {
            std::cout << "[BOOT] Overlay animator=ON (size=" << boot_cfg.overlay_pixel_size
                      << ", interval=" << boot_cfg.overlay_update_interval_100ns << "x100ns, spin="
                      << boot_cfg.overlay_spin_budget_us << "us, pos="
                      << boot_cfg.overlay_pos_x << "," << boot_cfg.overlay_pos_y << ")" << std::endl;
        }
    } else if (boot_cfg.enable_optimization_report && boot_cfg.enable_system_opt_report) {
        std::cout << "[BOOT] Overlay animator=OFF" << std::endl;
    }

    if (boot_cfg.enable_optimization_report && boot_cfg.enable_system_opt_report) {
        std::wcout << L"[BOOT] Scheduling: MMCSS=" << (hMmcss?L"ON":L"OFF")
                   << L" (task=" << boot_cfg.mmcss_task_name << L", prio=" << (int)boot_cfg.mmcss_priority << L")" << std::endl;
    }

    try {
        AimAssistant assistant;
        assistant.Run();
    }
    catch (const std::exception& e) {
        DisableMmcss(hMmcss);
        std::cerr << "\n\n[FATAL ERROR] An unrecoverable error occurred: " << e.what() << std::endl;
        std::cerr << "Please check the following:\n"
            << "Press Enter to exit." << std::endl;
        std::cin.get();
        return -1;
    }
    std::cout << "\nProgram finished successfully." << std::endl;
    overlay.stop();
    DisableMmcss(hMmcss);
    return 0;
}
