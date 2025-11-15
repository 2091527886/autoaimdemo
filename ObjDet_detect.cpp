#include "ObjDet.hpp"
#include "Config.hpp"
#include "Timing.hpp"
#include "sh.hpp"
#include "objdet_helpers.hpp"

#include "trtyolo_inc.hpp"
using R = cv::Rect;

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
        if (score < cfg.conf_thresh) continue;
        int cls = (i < (int)res.classes.size()? res.classes[i] : -1);
        const auto &b = res.boxes[i];
        R r = rect_from_box_clamped(b, width, height);
        if (r.width>0 && r.height>0) out.emplace_back(Detection{r, score, cls});
      }
    }
  };

  if (cfg.timing) {
    auto t0 = tn();
    Img input(data, width, height, pitch);
    DR res = model_->predict(input);
    auto t1 = tn();
    tm.infer_ms = msd(t1 - t0).count();
    auto t2 = tn();
    fill(res);
    auto t3 = tn();
    tm.post_ms = msd(t3 - t2).count();
  } else {
    Img input(data, width, height, pitch);
    DR res = model_->predict(input);
    fill(res);
    tm.infer_ms = 0.0;
    tm.post_ms = 0.0;
  }

  objdet_perf(model_.get(), perf_, cfg.timing, cfg.perf_report, tm);
}
