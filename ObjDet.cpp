#include "ObjDet.hpp"

#include "Log.hpp"
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <algorithm>

#include <windows.h>
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4251)
#endif
#include <trtyolo.hpp>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include <opencv2/opencv.hpp>

#include "Config.hpp"
#include "Timing.hpp"

ObjDet::ObjDet(const Config& cfg)
    : perf_(cfg)
{
    const std::string engine = cfg.resolve_engine_path();
    lg::o() << P_I "Load TRT: " << engine << lg::endl;

    trtyolo::InferOption opt;
    opt.set_rb();
    // Do not set fixed input dimensions to preserve BGRA pitch; use dynamic path
    // opt.setInputDimensions(cfg.crop_size_px, cfg.crop_size_px);
    if (cfg.perf_report) {
        opt.enablePerformanceReport();
    }

    // Diagnostics similar to original
    {
        char cwd_buffer[MAX_PATH];
        if (GetCurrentDirectoryA(MAX_PATH, cwd_buffer)) {
            lg::o() << P_D "cwd: " << cwd_buffer << lg::endl;
        }
        DWORD attrs = GetFileAttributesA(engine.c_str());
        if (attrs == INVALID_FILE_ATTRIBUTES) {
            lg::e() << P_E "Engine not found: '" << engine << "'" << lg::endl;
        } else if (attrs & FILE_ATTRIBUTE_DIRECTORY) {
            lg::e() << P_E "Engine is dir: '" << engine << "'" << lg::endl;
        } else {
            lg::o() << P_D "Engine found: '" << engine << "'" << lg::endl;
        }
    }

    try {
        model_ = std::make_unique<trtyolo::DetectModel>(engine.c_str(), opt);
        lg::o() << P_I "Engine OK" << lg::endl;
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << P_F "DetectModel failed: '" << engine << "'\nerr: " << e.what();
        throw std::runtime_error(oss.str());
    }
}

ObjDet::~ObjDet() = default;

void ObjDet::detect(const cv::Mat& img,
                            std::vector<Detection>& out,
                            const Config& cfg,
                            Timing& tm)
{
    out.clear();

    if (cfg.timing) {
        auto t0 = std::chrono::high_resolution_clock::now();
        trtyolo::Image input(const_cast<unsigned char*>(img.data), img.cols, img.rows, static_cast<size_t>(img.step));
        trtyolo::DetectRes res = model_->predict(input);
        auto t1 = std::chrono::high_resolution_clock::now();
        tm.infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
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
        tm.post_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    } else {
        trtyolo::Image input(const_cast<unsigned char*>(img.data), img.cols, img.rows, static_cast<size_t>(img.step));
        trtyolo::DetectRes res = model_->predict(input);
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
        tm.infer_ms = 0.0;
        tm.post_ms = 0.0;
    }

    if (cfg.timing && cfg.perf_report && perf_.ShouldReport()) {
        auto report = model_->performanceReport();
        const std::string& tp = std::get<0>(report);
        if (!tp.empty()) {
            tm.trt_throughput = tp;
            tm.trt_cpu_lat = std::get<1>(report);
            tm.trt_gpu_lat = std::get<2>(report);
        }
    }
}

void ObjDet::detect_view(unsigned char* data,
                         int width,
                         int height,
                         size_t pitch,
                         std::vector<Detection>& out,
                         const Config& cfg,
                         Timing& tm)
{
    out.clear();

    if (cfg.timing) {
        auto t0 = std::chrono::high_resolution_clock::now();
        trtyolo::Image input(data, width, height, pitch);
        trtyolo::DetectRes res = model_->predict(input);
        auto t1 = std::chrono::high_resolution_clock::now();
        tm.infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
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
                w = std::min(w, width - x);
                h = std::min(h, height - y);
                if (w > 0 && h > 0)
                    out.emplace_back(Detection{ cv::Rect(x, y, w, h), score, cls });
            }
        }

        auto t3 = std::chrono::high_resolution_clock::now();
        tm.post_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    } else {
        trtyolo::Image input(data, width, height, pitch);
        trtyolo::DetectRes res = model_->predict(input);
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
                w = std::min(w, width - x);
                h = std::min(h, height - y);
                if (w > 0 && h > 0)
                    out.emplace_back(Detection{ cv::Rect(x, y, w, h), score, cls });
            }
        }
        tm.infer_ms = 0.0;
        tm.post_ms = 0.0;
    }

    if (cfg.timing && cfg.perf_report && perf_.ShouldReport()) {
        auto report = model_->performanceReport();
        const std::string& tp = std::get<0>(report);
        if (!tp.empty()) {
            tm.trt_throughput = tp;
            tm.trt_cpu_lat = std::get<1>(report);
            tm.trt_gpu_lat = std::get<2>(report);
        }
    }
}
