#include "ObjectDetector.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <algorithm>

#include <windows.h>
#include <trtyolo.hpp>
#include <opencv2/opencv.hpp>

#include "Config.hpp"
#include "Timing.hpp"

ObjectDetector::ObjectDetector(const Config& cfg)
    : trt_reporter_(cfg)
{
    const std::string engine_path = cfg.resolve_engine_path();
    std::cout << "[INFO] Loading TensorRT-YOLO engine from " << engine_path << " ..." << std::endl;

    trtyolo::InferOption option;
    option.enableSwapRB();
    option.setInputDimensions(cfg.crop_size, cfg.crop_size);
    if (cfg.enable_performance_report) {
        option.enablePerformanceReport();
    }

    // Diagnostics similar to original
    {
        char cwd_buffer[MAX_PATH];
        if (GetCurrentDirectoryA(MAX_PATH, cwd_buffer)) {
            std::cout << "[DEBUG] Current working directory: " << cwd_buffer << std::endl;
        }
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
            << "Underlying error: " << e.what();
        throw std::runtime_error(oss.str());
    }
}

ObjectDetector::~ObjectDetector() = default;

void ObjectDetector::Detect(const cv::Mat& image,
                            std::vector<Detection>& detections,
                            const Config& cfg,
                            TimingDetails& timings)
{
    detections.clear();
    auto t0 = std::chrono::high_resolution_clock::now();

    trtyolo::Image input(const_cast<unsigned char*>(image.data), image.cols, image.rows, static_cast<size_t>(image.step));
    trtyolo::DetectRes res = detector_->predict(input);

    auto t1 = std::chrono::high_resolution_clock::now();
    timings.inference_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto t2 = std::chrono::high_resolution_clock::now();

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
            w = std::min(w, image.cols - x);
            h = std::min(h, image.rows - y);
            if (w > 0 && h > 0)
                detections.emplace_back(Detection{ cv::Rect(x, y, w, h), score, cls });
        }
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    timings.postprocess_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    if (cfg.enable_performance_report) {
        if (trt_reporter_.ShouldReport()) {
            trt_reporter_.PrintReport(*detector_);
        }
    }
}
