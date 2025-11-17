#include "ObjDet.hpp"
#include "Config.hpp"
#include "Timing.hpp"
#include "sh.hpp"
#include "objdet_helpers.hpp"
#include <cuda_runtime.h>
#include "trtyolo_inc.hpp"
using R = cv::Rect;

cudaStream_t ObjDet::get_stream() const { return model_? model_->getStream() : 0; }

void objdet_perf(trtyolo::DetectModel* model,
                 metrics::TrtPerfReporting& perf,
                 bool timing,
                 bool perf_report,
                 Timing& tm);

using DR = trtyolo::DetectRes;
using Img = trtyolo::Image;

void ObjDet::detect_view(unsigned char *data,
                         int width,
                         int height,
                         size_t pitch,
                         std::vector<Detection> &out,
                         const Config &cfg,
                         Timing &tm) {
  out.clear();

  auto fill = [&](const trtyolo::DetectRes &res){
    if (res.num > 0 && res.boxes.size() == (size_t)res.num){
      for (int i=0;i<res.num;++i){
        float score = (i < (int)res.scores.size()? res.scores[i] : 0.0f);
        if (!accept_det(score, cfg.conf_thresh)) continue;
        int cls = (i < (int)res.classes.size()? res.classes[i] : -1);
        const auto &b = res.boxes[i];
        auto det = make_det(b, width, height, score, cls);
        if (det.box.width>0 && det.box.height>0) out.emplace_back(det);
      }
    }
  };

  // 清空设备侧视图缓存
  last_dev_ = {};

  if (cfg.timing) {
    auto t0 = tn();
    Img input(data, width, height, pitch);
    // 当开启导出设备侧结果时，优先走 predict_dev + OC-SORT；否则回退到 CPU res
    if (true) {
      auto res_dev = model_->predict_dev(input);
      last_dev_.xyxy = res_dev.xyxy; last_dev_.scores = res_dev.scores; last_dev_.classes = res_dev.classes; last_dev_.num = res_dev.num;
      auto t1 = tn();
      tm.infer_ms = msd(t1 - t0).count();
      // 将设备侧结果直接交给 Aim 层（OCSort 在 Aim_loop 中消费），这里维持 out 仅为 obs 以便 overlay
      // 兼容：也填充 out，便于其他模块使用（通过一次 D2H 拷贝，代价较小）
      if (res_dev.num > 0) {
        std::vector<float> h_xyxy(res_dev.num*4);
        std::vector<float> h_sc(res_dev.num);
        std::vector<int>   h_cl(res_dev.num);
        // Use async copies on the same stream to reduce blocking
        cudaMemcpyAsync(h_xyxy.data(), res_dev.xyxy, res_dev.num*4*sizeof(float), cudaMemcpyDeviceToHost, get_stream());
        cudaMemcpyAsync(h_sc.data(),   res_dev.scores, res_dev.num*sizeof(float), cudaMemcpyDeviceToHost, get_stream());
        cudaMemcpyAsync(h_cl.data(),   res_dev.classes, res_dev.num*sizeof(int), cudaMemcpyDeviceToHost, get_stream());
        cudaStreamSynchronize(get_stream());
        for (int i=0;i<res_dev.num;++i){
          float score = h_sc[i]; if (!accept_det(score, cfg.conf_thresh)) continue;
          int cls = h_cl[i];
          Detection det; det.confidence = score; det.class_id = cls;
          det.box.x = (int)h_xyxy[i*4+0]; det.box.y = (int)h_xyxy[i*4+1];
          det.box.width  = (int)(h_xyxy[i*4+2] - h_xyxy[i*4+0]);
          det.box.height = (int)(h_xyxy[i*4+3] - h_xyxy[i*4+1]);
          if (det.box.width>0 && det.box.height>0) out.emplace_back(det);
        }
      }
      auto t3 = tn();
      tm.post_ms = msd(t3 - t1).count();
    } else {
      DR res = model_->predict(input);
      auto t1 = tn(); tm.infer_ms = msd(t1 - t0).count();
      auto t2 = tn(); fill(res); auto t3 = tn(); tm.post_ms = msd(t3 - t2).count();
    }
  } else {
    Img input(data, width, height, pitch);
    auto res_dev = model_->predict_dev(input);
    last_dev_.xyxy = res_dev.xyxy; last_dev_.scores = res_dev.scores; last_dev_.classes = res_dev.classes; last_dev_.num = res_dev.num;
    if (res_dev.num > 0) {
      std::vector<float> h_xyxy(res_dev.num*4);
      std::vector<float> h_sc(res_dev.num);
      std::vector<int>   h_cl(res_dev.num);
      // Use async copies on the TRT stream, then sync once
      cudaMemcpyAsync(h_xyxy.data(), res_dev.xyxy, res_dev.num*4*sizeof(float), cudaMemcpyDeviceToHost, get_stream());
      cudaMemcpyAsync(h_sc.data(),   res_dev.scores, res_dev.num*sizeof(float), cudaMemcpyDeviceToHost, get_stream());
      cudaMemcpyAsync(h_cl.data(),   res_dev.classes, res_dev.num*sizeof(int), cudaMemcpyDeviceToHost, get_stream());
      cudaStreamSynchronize(get_stream());
      for (int i=0;i<res_dev.num;++i){
        float score = h_sc[i]; if (!accept_det(score, cfg.conf_thresh)) continue;
        int cls = h_cl[i];
        Detection det; det.confidence = score; det.class_id = cls;
        det.box.x = (int)h_xyxy[i*4+0]; det.box.y = (int)h_xyxy[i*4+1];
        det.box.width  = (int)(h_xyxy[i*4+2] - h_xyxy[i*4+0]);
        det.box.height = (int)(h_xyxy[i*4+3] - h_xyxy[i*4+1]);
        if (det.box.width>0 && det.box.height>0) out.emplace_back(det);
      }
    }
    tm.infer_ms = 0.0; tm.post_ms = 0.0;
  }

  objdet_perf(model_.get(), perf_, cfg.timing, cfg.perf_report, tm);
}
