#include "Pose.hpp"
#include "Config.hpp"
#include "Timing.hpp"
#include "Detection.hpp"
#include "objdet_helpers.hpp"

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

    if (cfg.pose_trt_input_w>0 && cfg.pose_trt_input_h>0){
        opt.setInputDimensions(cfg.pose_trt_input_w, cfg.pose_trt_input_h);
    } else if (cfg.trt_input_w>0 && cfg.trt_input_h>0){
        opt.setInputDimensions(cfg.trt_input_w, cfg.trt_input_h);
    } else {
        opt.setInputDimensions(cfg.crop_size_px, cfg.crop_size_px);
    }
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
        return true;
    } else {
        auto res=do_predict(batch); if (res.empty()) return false; const auto& r = res[0];
        if (r.num<=0 || r.kpts.empty()) return false;
        out_kpts = r.kpts;
        out_boxes.clear(); out_boxes.reserve(r.boxes.size());
        for (const auto& b : r.boxes) out_boxes.emplace_back((int)b.left, (int)b.top, (int)(b.right - b.left), (int)(b.bottom - b.top));
        return true;
    }
}

void PoseInfer::run_on_crops(unsigned char* data,
                             int view_w,
                             int view_h,
                             size_t pitch,
                             const std::vector<Detection>& dets,
                             const std::vector<int>& sel_idx,
                             std::vector<std::vector<trtyolo::KeyPoint>>& out_kpts,
                             std::vector<bool>& out_ok,
                             const Config& cfg,
                             Timing& tm){
    out_kpts.assign(dets.size(), {});
    out_ok.assign(dets.size(), false);
    if (!model_) return;
    if (sel_idx.empty()) return;

    std::vector<Img> batch;
    batch.reserve(sel_idx.size());
    struct CropInfo{ int idx; int x; int y; int w; int h; size_t off; };
    std::vector<CropInfo> infos;

    for(int idx : sel_idx){
        if (idx < 0 || idx >= (int)dets.size()) continue;
        const auto& d = dets[idx];
        if (d.box.width < 8 || d.box.height < 8) continue; // tiny skip
        int x = d.box.x, y = d.box.y, w = d.box.width, h = d.box.height;
        // clamp to view
        x = MX(0, MN(x, view_w-1)); y = MX(0, MN(y, view_h-1));
        w = MX(0, MN(w, view_w - x)); h = MX(0, MN(h, view_h - y));
        if (w<=0 || h<=0) continue;
        unsigned char* ptr = data + (size_t)y * pitch + (size_t)x * 4;
        batch.emplace_back(ptr, w, h, pitch);
        infos.push_back({idx,x,y,w,h,(size_t)y * pitch + (size_t)x * 4});
    }

    if (batch.empty()) return;

    auto do_predict = [&](const std::vector<Img>& imgs){ return model_->predict(imgs); };

    if (cfg.timing){
        auto t0 = tn();
        auto results = do_predict(batch);
        auto t1 = tn();
        tm.infer_ms += msd(t1 - t0).count();
        for(size_t bi=0; bi<results.size() && bi<infos.size(); ++bi){
            const auto& r = results[bi];
            const auto& inf = infos[bi];
            if (r.num<=0 || r.kpts.empty()) { out_ok[inf.idx]=false; continue; }
            // take the first instance in crop
            const auto& kpts = r.kpts.front();
            auto& dst = out_kpts[inf.idx];
            dst.reserve(kpts.size());
            for(const auto& kp : kpts){
                dst.emplace_back(kp.x + inf.x, kp.y + inf.y, kp.conf);
            }
            out_ok[inf.idx]=true;
        }
    } else {
        auto results = do_predict(batch);
        for(size_t bi=0; bi<results.size() && bi<infos.size(); ++bi){
            const auto& r = results[bi];
            const auto& inf = infos[bi];
            if (r.num<=0 || r.kpts.empty()) { out_ok[inf.idx]=false; continue; }
            const auto& kpts = r.kpts.front();
            auto& dst = out_kpts[inf.idx];
            dst.reserve(kpts.size());
            for(const auto& kp : kpts){ dst.emplace_back(kp.x + inf.x, kp.y + inf.y, kp.conf); }
            out_ok[inf.idx]=true;
        }
    }
}
