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
// Windows & DirectX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

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
    double preprocess_ms = 0;
    double inference_ms = 0;
    double postprocess_ms = 0;
    double total_loop_ms = 0;
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
class Config {
public:
    // Configuration center: resolve engine file path via Config only
    std::string engine_path = "best.engine"; // default value, can be edited in code
    std::string resolve_engine_path() const { return engine_path; }
    const int crop_size = 640;
    const int max_lock_distance_pixels = 320;
    const float confidence_threshold = 0.6f;
    const float nms_threshold = 0.4f;
    const int smooth_aim_key = VK_RBUTTON;
    const int single_shot_key = VK_F8;
    const double target_y_ratio = 0.6;
    const double pixels_for_360_turn = 3000;
    const double horizontal_fov = 75.0;
    const bool enable_visualization = true;
    const std::string window_name = "YOLO Real-time Detection";

    // 是否启用 TensorRT-YOLO 内部性能报告
    bool enable_performance_report = true;
    // 性能日志打印频率控制：默认每 30 帧打印一次；当 perf_report_every_frame=true 时每帧打印
    int  perf_report_interval_frames = 30;
    bool perf_report_every_frame = false;

    // PID 参数（像素域），输出会再乘 mouse_correction_factor
    const double pid_kp_x = 0.54;
    const double pid_ki_x = 0.001;
    const double pid_kd_x = 0.00600;
    const double pid_kp_y = 0.54;
    const double pid_ki_y = 0.001;
    const double pid_kd_y = 0.00600;
    const double pid_output_limit = 400.0;   // 每帧最大像素输出
    const double pid_integral_limit = 300.0; // 积分限幅
    const bool   pid_use_measured_dt = true; // 用实测 dt 计算；不稳定时可改成 false
    const double pid_fixed_dt_ms = 16.0;     // 固定 dt（毫秒），当 pid_use_measured_dt=false 时生效

    // 类别可读名称与追踪 ID 都在此配置，不从外部读取
    std::vector<std::string> class_names = { "enemy","teammate"};
    std::vector<int> target_class_ids = {0};

    bool is_target_class(int cid) const {
        return std::find(target_class_ids.begin(), target_class_ids.end(), cid) != target_class_ids.end();
    }
    std::string class_name_or_id(int cid) const {
        if (cid >= 0 && cid < (int)class_names.size() && !class_names[cid].empty()) return class_names[cid];
        return std::string("id=") + std::to_string(cid);
    }
};

// --- 2. 鼠标控制器 (MouseController) ---
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
    ScreenCapturer(int crop_width, int crop_height) {
        std::cout << "--- Initializing D3D for screen capture... ---" << std::endl;
        std::cout << "Crop size: " << crop_width << "x" << crop_height << std::endl;
        HRESULT hr;
        std::cout << "Init pointers: m_pStagingTexture=" << m_pStagingTexture
                  << ", pFactory=" << pFactory
                  << ", pAdapter=" << pAdapter
                  << ", pOutput=" << pOutput
                  << ", pOutput1=" << pOutput1
                  << ", pDevice=" << pDevice
                  << ", pContext=" << pContext
                  << ", pDuplicator=" << pDuplicator
                  << std::endl;

        hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&pFactory);
        std::cout << "CreateDXGIFactory1 hr=0x"
                  << std::hex << std::uppercase << static_cast<unsigned long>(hr)
                  << std::dec << std::nouppercase << std::endl;
        std::cout << "pFactory (IDXGIFactory1*)=" << pFactory << std::endl;
        if (FAILED(hr)) throw std::runtime_error("Failed to create DXGI Factory.");
        {
            HRESULT hr_enum_adapters = pFactory->EnumAdapters1(0, &pAdapter);
            std::cout << "EnumAdapters1 hr=0x"
                      << std::hex << std::uppercase << static_cast<unsigned long>(hr_enum_adapters)
                      << std::dec << std::nouppercase << std::endl;
            std::cout << "pAdapter (IDXGIAdapter1*)=" << pAdapter << std::endl;
            if (FAILED(hr_enum_adapters)) throw std::runtime_error("Failed to enumerate adapters.");
        }
        {
            HRESULT hr_enum_outputs = pAdapter->EnumOutputs(0, &pOutput);
            std::cout << "EnumOutputs hr=0x"
                      << std::hex << std::uppercase << static_cast<unsigned long>(hr_enum_outputs)
                      << std::dec << std::nouppercase << std::endl;
            std::cout << "pOutput (IDXGIOutput*)=" << pOutput << std::endl;
            if (FAILED(hr_enum_outputs)) throw std::runtime_error("Failed to enumerate outputs.");
        }

        DXGI_OUTPUT_DESC outputDesc;
        pOutput->GetDesc(&outputDesc);
        width = outputDesc.DesktopCoordinates.right - outputDesc.DesktopCoordinates.left;
        height = outputDesc.DesktopCoordinates.bottom - outputDesc.DesktopCoordinates.top;
        std::cout << "Output desktop size: " << width << "x" << height << std::endl;

        {
            HRESULT hr_create_device = D3D11CreateDevice(pAdapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &pDevice, nullptr, &pContext);
            std::cout << "D3D11CreateDevice hr=0x"
                      << std::hex << std::uppercase << static_cast<unsigned long>(hr_create_device)
                      << std::dec << std::nouppercase << std::endl;
            std::cout << "pDevice (ID3D11Device*)=" << pDevice << ", pContext (ID3D11DeviceContext*)=" << pContext << std::endl;
            if (FAILED(hr_create_device)) {
                throw std::runtime_error("Failed to create D3D11 device.");
            }
        }
        {
            HRESULT hr_qi_output1 = pOutput->QueryInterface(__uuidof(IDXGIOutput1), (void**)&pOutput1);
            std::cout << "QueryInterface(IDXGIOutput1) hr=0x"
                      << std::hex << std::uppercase << static_cast<unsigned long>(hr_qi_output1)
                      << std::dec << std::nouppercase << std::endl;
            std::cout << "pOutput1 (IDXGIOutput1*)=" << pOutput1 << std::endl;
            if (FAILED(hr_qi_output1)) {
                throw std::runtime_error("Failed to query IDXGIOutput1.");
            }
        }
        hr = pOutput1->DuplicateOutput(pDevice, &pDuplicator);
        std::cout << "pDuplicator (IDXGIOutputDuplication*)=" << pDuplicator << std::endl;
        if (FAILED(hr)) {
            std::cerr << "DuplicateOutput failed. HRESULT=0x"
                      << std::hex << std::uppercase << static_cast<unsigned long>(hr)
                      << std::dec << std::nouppercase << std::endl;
            throw std::runtime_error("Failed to create output duplication.");
        }

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

        hr = pDevice->CreateTexture2D(&desc, NULL, &m_pStagingTexture);
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to create staging texture for cropping.");
        }
        std::cout << "--- Screen capture initialized successfully (" << width << "x" << height << ") ---" << std::endl;
    }

    ~ScreenCapturer() {
        std::cout << "--- Cleaning up D3D resources... ---" << std::endl;
        SafeRelease(&m_pStagingTexture);
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
        IDXGIResource* pDesktopResource = nullptr;
        DXGI_OUTDUPL_FRAME_INFO frameInfo;
        HRESULT hr = pDuplicator->AcquireNextFrame(16, &frameInfo, &pDesktopResource);

        if (hr == DXGI_ERROR_WAIT_TIMEOUT) return false;
        if (FAILED(hr)) {
            pDuplicator->ReleaseFrame();
            std::cerr << "AcquireNextFrame failed. HRESULT: 0x" << std::hex << hr << std::endl;
            return false;
        }

        ID3D11Texture2D* pAcquiredDesktopImage = nullptr;
        hr = pDesktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&pAcquiredDesktopImage);
        SafeRelease(&pDesktopResource);
        if (FAILED(hr)) {
            pDuplicator->ReleaseFrame();
            return false;
        }

        D3D11_BOX sourceRegion;
        sourceRegion.left = crop_region.x;
        sourceRegion.right = crop_region.x + crop_region.width;
        sourceRegion.top = crop_region.y;
        sourceRegion.bottom = crop_region.y + crop_region.height;
        sourceRegion.front = 0;
        sourceRegion.back = 1;

        pContext->CopySubresourceRegion(m_pStagingTexture, 0, 0, 0, 0, pAcquiredDesktopImage, 0, &sourceRegion);

        D3D11_MAPPED_SUBRESOURCE mappedResource;
        hr = pContext->Map(m_pStagingTexture, 0, D3D11_MAP_READ, 0, &mappedResource);
        if (FAILED(hr)) {
            SafeRelease(&pAcquiredDesktopImage);
            pDuplicator->ReleaseFrame();
            return false;
        }

        cv::Mat bgra_frame(crop_region.height, crop_region.width, CV_8UC4, mappedResource.pData, mappedResource.RowPitch);
        cv::cvtColor(bgra_frame, frame, cv::COLOR_BGRA2BGR);

        pContext->Unmap(m_pStagingTexture, 0);
        SafeRelease(&pAcquiredDesktopImage);
        pDuplicator->ReleaseFrame();
        return true;
    }

    int getWidth() const { return width; }
    int getHeight() const { return height; }

private:
    ID3D11Texture2D* m_pStagingTexture = nullptr;
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
        timings.preprocess_ms = 0.0; // no explicit preprocess here
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
        capturer(cfg.crop_size, cfg.crop_size), // 传递 crop_size
        detector(cfg),
        mouse(),
        crop_region((capturer.getWidth() - cfg.crop_size) / 2, (capturer.getHeight() - cfg.crop_size) / 2, cfg.crop_size, cfg.crop_size),
        crop_center(cfg.crop_size / 2, cfg.crop_size / 2),
        is_visualizing(cfg.enable_visualization)
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
            if (!capturer.CaptureFrame(captured_frame, crop_region)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            detector.Detect(captured_frame, detections, cfg, timings);
            Detection* best_target = findBestTarget();
            handleMouseInput(best_target);
            auto loop_end_time = std::chrono::high_resolution_clock::now();
            timings.total_loop_ms = std::chrono::duration<double, std::milli>(loop_end_time - loop_start_time).count();
            // 计算 dt（秒）供 PID 使用
            if (cfg.pid_use_measured_dt) {
                pid_dt_s = std::max(0.001, timings.total_loop_ms / 1000.0);
            } else {
                pid_dt_s = std::max(0.001, cfg.pid_fixed_dt_ms / 1000.0);
            }
            updateAndPrintStats();
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
            smoothed_pre = timings.preprocess_ms;
            smoothed_inf = timings.inference_ms;
            smoothed_post = timings.postprocess_ms;
            is_first_frame = false;
        }
        else {
            smoothed_total = smoothing_factor * timings.total_loop_ms + (1.0 - smoothing_factor) * smoothed_total;
            smoothed_pre = smoothing_factor * timings.preprocess_ms + (1.0 - smoothing_factor) * smoothed_pre;
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
    double smoothed_total = 0.0, smoothed_pre = 0.0, smoothed_inf = 0.0, smoothed_post = 0.0;
    bool is_first_frame = true;
    int frame_count_for_console = 0;

    // PID 控制
    PidController pid_x;
    PidController pid_y;
    double pid_dt_s = 1.0/60.0;
};

// --- 主函数 ---
int main() {
    try {
        AimAssistant assistant;
        assistant.Run();
    }
    catch (const std::exception& e) {
        std::cerr << "\n\n[FATAL ERROR] An unrecoverable error occurred: " << e.what() << std::endl;
        std::cerr << "Please check the following:\n"
            << "Press Enter to exit." << std::endl;
        std::cin.get();
        return -1;
    }
    std::cout << "\nProgram finished successfully." << std::endl;
    return 0;
}