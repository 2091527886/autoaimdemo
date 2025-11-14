#include "ObjDet.hpp"

#include "Log.hpp"
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <algorithm>
#include <fstream>

#include <windows.h>
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4251)
#endif
#include <trtyolo.hpp>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include "Config.hpp"
#include "Timing.hpp"

ObjDet::ObjDet(const Config& cfg)
    : perf_(cfg)
{
    const std::string engine = cfg.resolve_engine_path();
    lg::o() << P_I "Load TRT: " << engine << lg::endl;

    trtyolo::InferOption opt;
    if (cfg.trt_swap_rb) opt.set_rb();
    // 配置预处理参数
    opt.setBorderValue(cfg.trt_border_value);
    if (cfg.trt_use_normalize && cfg.trt_mean.size()==3 && cfg.trt_std.size()==3) {
        opt.setNormalizeParams(cfg.trt_mean, cfg.trt_std);
    }
    // 内存策略
    if (cfg.trt_cuda_mem) opt.enableCudaMem();
    if (cfg.trt_managed_mem) opt.enableManagedMemory();

    // 固定输入尺寸（使用裁剪窗口）以启用静态图与减少分配
    if (cfg.trt_input_w>0 && cfg.trt_input_h>0) {
        opt.setInputDimensions(cfg.trt_input_w, cfg.trt_input_h);
    } else {
        opt.setInputDimensions(cfg.crop_size_px, cfg.crop_size_px);
    }
    if (cfg.perf_report) {
        opt.enablePerformanceReport();
    }

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

   // 外部 block 配置覆盖（可选，先解析再一次性应用，避免顺序影响）
   int final_bx = cfg.lb_block_x;
   int final_by = cfg.lb_block_y;
   bool final_use_lb = cfg.lb_use_launch_bounds;
   int  final_minb   = cfg.lb_min_blocks_per_sm;
   bool final_use_occ= cfg.lb_use_occupancy;
   if (!cfg.lb_block_cfg_path.empty()) {
       std::ifstream fin(cfg.lb_block_cfg_path);
       if (fin) {
           std::string line;
           auto trim = [](std::string& s){
               size_t a = s.find_first_not_of(" \t\r\n");
               size_t b = s.find_last_not_of(" \t\r\n");
               if (a==std::string::npos) { s.clear(); return; }
               s = s.substr(a, b-a+1);
           };
           while (std::getline(fin, line)) {
               trim(line);
               if (line.empty() || line[0]=='#' || line[0]==';') continue;
               auto pos = line.find('=');
               if (pos==std::string::npos) continue;
               std::string k = line.substr(0,pos), v = line.substr(pos+1);
               trim(k); trim(v);
               if (k=="block_x") final_bx = std::stoi(v);
               else if (k=="block_y") final_by = std::stoi(v);
               else if (k=="use_launch_bounds") final_use_lb = (std::stoi(v)!=0);
               else if (k=="min_blocks_per_sm") final_minb = std::stoi(v);
               else if (k=="use_occupancy") final_use_occ = (std::stoi(v)!=0);
           }
       }
   }
   // 应用解析后的最终参数
   opt.setLetterboxBlockXY(final_bx, final_by);
   opt.setLetterboxLaunchBounds(final_use_lb, final_minb);
   opt.enableLetterboxOccupancy(final_use_occ);
   // 日志打印最终选择，便于核查
   lg::o() << P_I "Letterbox kernel config: block_x=" << final_bx
           << ", block_y=" << final_by
           << ", min_blocks_per_sm=" << final_minb
           << ", use_launch_bounds=" << (final_use_lb?"1":"0")
           << ", use_occupancy=" << (final_use_occ?"1":"0")
           << lg::endl;

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

void ObjDet::detect_view(unsigned char* data,
                         int width,
                         int height,
                         size_t pitch,
                         std::vector<Detection>& out,
                         const Config& cfg,
                         Timing& tm)
{
    out.clear();

    auto fill = [&](const trtyolo::DetectRes& res){
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
    };

    if (cfg.timing) {
        auto t0 = std::chrono::high_resolution_clock::now();
        trtyolo::Image input(data, width, height, pitch);
        trtyolo::DetectRes res = model_->predict(input);
        auto t1 = std::chrono::high_resolution_clock::now();
        tm.infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        auto t2 = std::chrono::high_resolution_clock::now();

        fill(res);

        auto t3 = std::chrono::high_resolution_clock::now();
        tm.post_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    } else {
        trtyolo::Image input(data, width, height, pitch);
        trtyolo::DetectRes res = model_->predict(input);
        fill(res);
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
