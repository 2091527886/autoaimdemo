#include "ObjDet.hpp"

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

ObjDet::ObjDet(const Config& cfg)
    : perf_(cfg)
{
    const std::string engine = cfg.resolve_engine_path();
    std::cout << "[INFO] Loading TensorRT-YOLO engine from " << engine << " ..." << std::endl;

    trtyolo::InferOption opt;
    opt.enableSwapRB();
    opt.setInputDimensions(cfg.crop_size_px, cfg.crop_size_px);
    if (cfg.perf_report) {
        opt.enablePerformanceReport();
    }

    // Diagnostics similar to original
    {
        char cwd_buffer[MAX_PATH];
        if (GetCurrentDirectoryA(MAX_PATH, cwd_buffer)) {
            std::cout << "[DEBUG] Current working directory: " << cwd_buffer << std::endl;
        }
        DWORD attrs = GetFileAttributesA(engine.c_str());
        if (attrs == INVALID_FILE_ATTRIBUTES) {
            std::cerr << "[ERROR] Engine file not found: '" << engine << "'."
                      << " Ensure the engine file is present or provide the correct path." << std::endl;
        } else if (attrs & FILE_ATTRIBUTE_DIRECTORY) {
            std::cerr << "[ERROR] Engine path is a directory, not a file: '" << engine << "'." << std::endl;
        } else {
            std::cout << "[DEBUG] Found engine file: '" << engine << "'." << std::endl;
        }
    }

    try {
        model_ = std::make_unique<trtyolo::DetectModel>(engine.c_str(), opt);
        std::cout << "[INFO] Engine loaded successfully." << std::endl;
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "[FATAL] Failed to construct DetectModel with engine '" << engine << "'.\n"
            << "Underlying error: " << e.what();
        throw std::runtime_error(oss.str());
    }
}

ObjDet::~ObjDet() = default;

void ObjDet::detect(const cv::Mat& img,
                            std::vector<Detection>& out,
                            const Config& cfg,
                            TimingDetails& tms)
{
    out.clear();
    auto t0 = std::chrono::high_resolution_clock::now();

    trtyolo::Image input(const_cast<unsigned char*>(img.data), img.cols, img.rows, static_cast<size_t>(img.step));
    trtyolo::DetectRes res = model_->predict(input);

    auto t1 = std::chrono::high_resolution_clock::now();
    tms.inference_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    auto t2 = std::chrono::high_resolution_clock::now();

    if (res.num > 0 && res.boxes.size() == static_cast<size_t>(res.num)) {
        for (int i = 0; i < res.num; ++i) {
            float score = (i < (int)res.scores.size() ? res.scores[i] : 0.0f);
            if (score < cfg.conf_thresh) continue;
            int cls = (i < (int)res.classes.size() ? res.classes[i] : -1);
            const auto& b = res.boxes[i];
            int x = std::max(0, static_cast<int>(std::floor(b.left)));
            int y = std::max(0, static_cast<int>(std::floor(b.top)));
            int w = std::max(0, static_cast<int>(std::round(b.right - b.left)));
            int h = std::max(0, static_cast<int>(std::round(b.bottom - b.top)));
            w = std::min(w, img.cols - x);
            h = std::min(h, img.rows - y);
            if (w > 0 && h > 0)
                out.emplace_back(Detection{ cv::Rect(x, y, w, h), score, cls });
        }
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    tms.postprocess_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    if (cfg.perf_report) {
        if (perf_.ShouldReport()) {
            perf_.PrintReport(*model_);
        }
    }
}
