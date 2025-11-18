#include "Pose.hpp"
#include "Config.hpp"
#include "Timing.hpp"
#include "Detection.hpp"

#include "trtyolo_inc.hpp"

using Img = trtyolo::Image;
using PR = trtyolo::PoseRes;

PoseInfer::PoseInfer(const Config& cfg)
    : perf_(cfg) {
    if (!cfg.pose_enable) return;

    const std::string engine = cfg.pose_engine_path; // e.g. "spose.engine"
    LO << P_I "Load TRT Pose: " << engine << END;

    trtyolo::InferOption opt;
    if (cfg.pose_trt_swap_rb) opt.set_rb();
    opt.setBorderValue(cfg.trt_border_value); // reuse det border value
    if (cfg.pose_trt_use_normalize && cfg.pose_trt_mean.size()==3 && cfg.pose_trt_std.size()==3){
        opt.setNormalizeParams(cfg.pose_trt_mean, cfg.pose_trt_std);
    }
    if (cfg.pose_trt_cuda_mem) opt.enableCudaMem();
    if (cfg.pose_trt_managed_mem) opt.enableManagedMemory();

    opt.setInputDimensions(cfg.pose_trt_input_w, cfg.pose_trt_input_h);
    if (cfg.perf_report) opt.enablePerformanceReport();

    try{
        model_.reset(new trtyolo::PoseModel(engine.c_str(), opt));
        LO << P_I "Pose Engine OK" << END;
    }catch(const std::exception& e){
        LE << P_E "PoseModel failed: '" << engine << "' err: " << e.what() << END;
        model_.reset();
    }
}

PoseInfer::~PoseInfer() = default;

bool PoseInfer::run_on_view(unsigned char* data,
                            int view_w,
                            int view_h,
                            size_t pitch,
                            std::vector<std::vector<trtyolo::KeyPoint>>& out_kpts,
                            std::vector<cv::Rect>& out_boxes,
                            std::vector<int>& out_classes,
                            std::vector<float>& out_scores,
                            const Config& cfg,
                            Timing& tm){
    out_kpts.clear(); out_boxes.clear();
    if (!model_) return false;
    Img img(data, view_w, view_h, pitch);
    std::vector<Img> batch; batch.emplace_back(img);
    auto do_predict = [&](const std::vector<Img>& imgs){ return model_->predict(imgs); };
    if (cfg.timing){ auto t0=tn(); auto res=do_predict(batch); auto t1=tn(); tm.infer_ms += msd(t1-t0).count();
        if (res.empty()) return false; const auto& r = res[0];
        if (r.num<=0 || r.kpts.empty()) return false;
        out_kpts = r.kpts; // already in view coords
        out_boxes.clear(); out_boxes.reserve(r.boxes.size());
        for (const auto& b : r.boxes) out_boxes.emplace_back((int)b.left, (int)b.top, (int)(b.right - b.left), (int)(b.bottom - b.top));
        out_classes = r.classes;
        out_scores  = r.scores;
        // fetch TRT perf strings for this frame (if enabled)
        if (cfg.perf_report) {
            auto pr = model_->performanceReport();
            tm.trt_throughput = std::get<0>(pr);
            tm.trt_cpu_lat    = std::get<1>(pr);
            tm.trt_gpu_lat    = std::get<2>(pr);
        }
        return true;
    } else {
        auto res=do_predict(batch); if (res.empty()) return false; const auto& r = res[0];
        if (r.num<=0 || r.kpts.empty()) return false;
        out_kpts = r.kpts;
        out_boxes.clear(); out_boxes.reserve(r.boxes.size());
        for (const auto& b : r.boxes) out_boxes.emplace_back((int)b.left, (int)b.top, (int)(b.right - b.left), (int)(b.bottom - b.top));
        out_classes = r.classes;
        out_scores  = r.scores;
        if (cfg.perf_report) {
            auto pr = model_->performanceReport();
            tm.trt_throughput = std::get<0>(pr);
            tm.trt_cpu_lat    = std::get<1>(pr);
            tm.trt_gpu_lat    = std::get<2>(pr);
        }
        return true;
    }
}

