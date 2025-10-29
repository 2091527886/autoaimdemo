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

// --- Priority helpers moved to separate modules ---
/* moved */


// 将当前线程内存优先级显式设置为 NORMAL（保证不被降级）
/* moved */
// 禁用进程 PriorityBoost，避免在系统内存压力下被降低优先级
/* moved */

// Forward declare Config for use in OverlayAnimator
class Config;
/* moved: MMCSS utilities now in MmcssUtil */
/* moved */
void DisableMmcss(HANDLE h);
/* moved */

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
#include "Timing.hpp"
#include "PidController.hpp"
#include "SystemPriority.hpp"
#include "PowerThrottling.hpp"
#include "MmcssUtil.hpp"
#include "DxgiTuning.hpp"
#include "ThreadAffinity.hpp"
#include "metrics/CaptureReporting.hpp"
#include "metrics/TrtPerfReporting.hpp"
#include "metrics/SystemBootReporting.hpp"
#include "ComHelpers.hpp"
#include "Detection.hpp"

// Forward Declarations
struct TimingDetails;
class Config;
class ScreenCapturer;
class ObjectDetector;
class MouseController;
class AimAssistant;

// --- 辅助宏 ---
// --- 数据结构定义 ---



// --- PID 控制器 (moved to PidController.hpp/.cpp) ---


// --- 2. 鼠标控制器 (MouseController) ---
#include "OverlayAnimator.hpp"
#include "MouseController.hpp"
#include "HotkeyManager.hpp"
#include "ObjectDetector.hpp"

// --- 3. 屏幕捕捉器 (ScreenCapturer) - 优化版本 ---
class ScreenCapturer {
public:
    // Reporting helper
    metrics::CaptureReporting capture_reporter_;
public:
    ScreenCapturer(const Config& cfg_ref, int crop_width, int crop_height) : cfg(cfg_ref), capture_reporter_(cfg_ref) {
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
        capture_reporter_.BootBanner(kStagingCount);
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
        // Delegate to metrics reporter to preserve behavior and frequency/format
        capture_reporter_.ReportStepTimings(
            t_acquire_ms, t_qi_ms, t_copy_ms, t_map_ms, t_cvt_ms, t_unmap_ms, t_total_ms,
            stat_dirty_skips_, stat_event_not_ready_, stat_map_still_drawing_, stat_cache_reused_, stat_success_
        );
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
/* moved to ObjectDetector.hpp/.cpp */

// --- 5. 核心协调器 (AimAssistant) ---
class AimAssistant {
public:
    AimAssistant()
        : cfg(),
        capturer(cfg, cfg.crop_size, cfg.crop_size), // 传递 cfg + crop_size
        detector(cfg),
        mouse(),
        hotkeys(cfg),
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
        bool smooth_pressed = hotkeys.SmoothPressed();
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

        // 单发模式：保持边沿触发语义
        if (hotkeys.SingleShotJustPressed()) {
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
    HotkeyManager hotkeys;
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
        metrics::SystemBootReporting::PrintOverlayBanner(boot_cfg, true);
    } else {
        metrics::SystemBootReporting::PrintOverlayBanner(boot_cfg, false);
    }

    metrics::SystemBootReporting::PrintSchedulingBanner(boot_cfg, hMmcss);

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
